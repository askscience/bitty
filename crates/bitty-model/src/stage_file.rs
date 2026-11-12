//! Per-stage model files.
//!
//! A node that serves layers `[start, end)` does not need the whole model. A stage file is a
//! normal GGUF with the full metadata (hyperparameters, tokenizer) and only the tensors of that
//! range, so it can be built by reading just those byte ranges from the published model: from a
//! local copy, or straight from an HTTP server that supports range requests.
//!
//! What goes into a stage:
//! - `blk.N.*` for every N in the range;
//! - `token_embd.weight` in the first stage, and in the last one when the model has no separate
//!   `output.weight` (tied output head);
//! - `output*` tensors (final norm, LM head) in the last stage;
//! - any other tensor that is not tied to a block, which keeps unknown small tensors available.

use crate::gguf::{parse_gguf_prefix, GgufError, GgufLayout, GgufMetadataValue, GgufTensorInfo};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use thiserror::Error;

/// Metadata keys a stage file carries on top of the model's own.
pub const STAGE_START_KEY: &str = "bitty.stage.layer_start";
pub const STAGE_END_KEY: &str = "bitty.stage.layer_end";

const FIRST_HEADER_READ: u64 = 16 << 20;
const MAX_HEADER_READ: u64 = 1 << 30;

#[derive(Debug, Error)]
pub enum StageFileError {
    #[error("stage file I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("model header is larger than {MAX_HEADER_READ} bytes")]
    HeaderTooLarge,
    #[error("model has no block count in its metadata")]
    NoBlockCount,
    #[error("layer range {start}..{end} is outside the model's {layers} blocks")]
    BadRange { start: u32, end: u32, layers: u32 },
    #[error("the source is already a stage file for layers {start}..{end}; build stages from the whole model")]
    SourceIsStage { start: u64, end: u64 },
    #[error("tensor {0} reaches past the end of the source")]
    TensorOutOfBounds(String),
    #[error("range request failed: {0}")]
    Fetch(String),
    #[error("the download was cancelled")]
    Cancelled,
    #[error("not enough disk for the layers: {needed_mb} MB needed, free_disk_mb={free_mb}")]
    NoSpace { needed_mb: u64, free_mb: u64 },
}

/// Space a download leaves alone, so the disk of the machine never fills to the last byte.
const DISK_MARGIN: u64 = 512 * 1024 * 1024;

/// The free bytes of the disk that holds `path`, or of its nearest folder that exists.
pub fn free_disk_bytes(path: &Path) -> Option<u64> {
    let mut existing = path;
    while !existing.exists() {
        existing = existing.parent()?;
    }
    let existing = without_long_path_mark(existing.canonicalize().ok()?);
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|disk| existing.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(|disk| disk.available_space())
}

/// The cores of this machine, without counting the second thread of each. `None` when the
/// system does not say.
pub fn physical_cores() -> Option<u32> {
    sysinfo::System::physical_core_count().map(|cores| cores as u32)
}

/// Windows gives a full path as `\\?\C:\...`, while its disks are named `C:\`: without the
/// mark in front the path is found on its disk.
fn without_long_path_mark(path: PathBuf) -> PathBuf {
    match path.to_str().and_then(|text| text.strip_prefix(r"\\?\")) {
        Some(plain) if !plain.starts_with("UNC\\") => PathBuf::from(plain),
        _ => path,
    }
}

/// The bytes a download into the folder of `path` may use.
pub fn usable_disk_bytes(path: &Path) -> Option<u64> {
    free_disk_bytes(path).map(|free| free.saturating_sub(DISK_MARGIN))
}

/// Asked while a download runs; `true` stops it.
pub type CancelCheck = std::sync::Arc<dyn Fn() -> bool + Send + Sync>;

/// Random access to the published model.
pub trait ByteSource {
    fn len(&self) -> Result<u64, StageFileError>;
    /// Writes exactly `len` bytes starting at `offset` into `out`.
    fn copy_range(&self, offset: u64, len: u64, out: &mut dyn Write) -> Result<(), StageFileError>;

    fn read_range(&self, offset: u64, len: u64) -> Result<Vec<u8>, StageFileError> {
        let mut bytes = Vec::with_capacity(len as usize);
        self.copy_range(offset, len, &mut bytes)?;
        Ok(bytes)
    }
}

/// A model file on disk.
pub struct FileSource {
    path: PathBuf,
}

impl FileSource {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl ByteSource for FileSource {
    fn len(&self) -> Result<u64, StageFileError> {
        Ok(std::fs::metadata(&self.path)?.len())
    }

    fn copy_range(&self, offset: u64, len: u64, out: &mut dyn Write) -> Result<(), StageFileError> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let copied = io::copy(&mut file.take(len), out)?;
        if copied != len {
            return Err(StageFileError::Io(io::ErrorKind::UnexpectedEof.into()));
        }
        Ok(())
    }
}

/// A model behind HTTP(S), read with range requests through `curl`.
pub struct HttpSource {
    url: String,
    cancel: Option<CancelCheck>,
}

impl HttpSource {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            cancel: None,
        }
    }

    /// Stops a running download as soon as `cancel` answers `true`.
    pub fn with_cancel(mut self, cancel: Option<CancelCheck>) -> Self {
        self.cancel = cancel;
        self
    }

    fn curl(&self) -> Command {
        let mut command = Command::new("curl");
        command
            .args(["--location", "--fail", "--silent", "--show-error"])
            .args([
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--proto",
                "=https,http",
            ])
            // a connection that stalls is ended, so the transfer can go on from where it was
            .args(["--speed-limit", "1024", "--speed-time", "60"]);
        command
    }
}

impl ByteSource for HttpSource {
    fn len(&self) -> Result<u64, StageFileError> {
        // a one-byte range answers with "Content-Range: bytes 0-0/<total>" and proves that the
        // server honours ranges at all
        let output = self
            .curl()
            .args(["--range", "0-0", "--dump-header", "-"])
            .args(["--output", if cfg!(windows) { "NUL" } else { "/dev/null" }])
            .arg(&self.url)
            .stderr(Stdio::piped())
            .output()?;
        if !output.status.success() {
            return Err(StageFileError::Fetch(
                String::from_utf8_lossy(&output.stderr).trim().into(),
            ));
        }
        let headers = String::from_utf8_lossy(&output.stdout);
        headers
            .lines()
            .filter_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.trim()
                    .eq_ignore_ascii_case("content-range")
                    .then_some(value)
            })
            .last()
            .and_then(|value| value.rsplit('/').next()?.trim().parse().ok())
            .ok_or_else(|| {
                StageFileError::Fetch("the server does not support range requests".into())
            })
    }

    fn copy_range(&self, offset: u64, len: u64, out: &mut dyn Write) -> Result<(), StageFileError> {
        // A transfer of several gigabytes over a home line breaks now and then. What arrived
        // is kept and the rest is asked again from where it stopped; only a row of attempts
        // that bring nothing ends the download.
        let mut done = 0u64;
        let mut empty_attempts = 0u32;
        while done < len {
            let received = self.copy_once(offset + done, len - done, out)?;
            if received > len - done {
                return Err(StageFileError::Fetch(format!(
                    "expected {len} bytes at offset {offset}, the server sent more"
                )));
            }
            done += received;
            if done == len {
                break;
            }
            if received == 0 {
                empty_attempts += 1;
                if empty_attempts >= RESUME_ATTEMPTS {
                    return Err(StageFileError::Fetch(format!(
                        "expected {len} bytes at offset {offset}, received {done}"
                    )));
                }
            } else {
                empty_attempts = 0;
            }
            std::thread::sleep(RESUME_PAUSE);
        }
        Ok(())
    }
}

/// Attempts in a row that may bring no byte before a download is given up.
const RESUME_ATTEMPTS: u32 = 5;
const RESUME_PAUSE: std::time::Duration = std::time::Duration::from_secs(3);

impl HttpSource {
    /// One range request. Returns how many bytes reached `out`, which is less than `len`
    /// when the connection broke on the way.
    fn copy_once(&self, offset: u64, len: u64, out: &mut dyn Write) -> Result<u64, StageFileError> {
        let mut child = self
            .curl()
            .arg("--range")
            .arg(format!("{}-{}", offset, offset + len - 1))
            .arg(&self.url)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let mut stdout = child.stdout.take().expect("piped stdout");
        // take(len + 1): a server that ignores the range would send the whole file
        let mut limited = (&mut stdout).take(len + 1);
        let mut buffer = vec![0u8; 1 << 20];
        let mut cancelled = false;
        let mut total = 0u64;
        let mut write_error = None;
        loop {
            if self.cancel.as_ref().is_some_and(|cancel| cancel()) {
                cancelled = true;
                break;
            }
            let read = match limited.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                // a broken pipe from curl: what arrived so far still counts
                Err(_) => break,
            };
            if total + read as u64 > len {
                total += read as u64;
                break;
            }
            if let Err(err) = out.write_all(&buffer[..read]) {
                write_error = Some(err);
                break;
            }
            total += read as u64;
        }
        drop(stdout);
        if cancelled || total > len || write_error.is_some() {
            let _ = child.kill();
        }
        let _ = child.wait()?;
        if cancelled {
            return Err(StageFileError::Cancelled);
        }
        if let Some(err) = write_error {
            return Err(err.into());
        }
        Ok(total)
    }
}

/// What a stage file is made of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageFileReport {
    pub layer_count: u32,
    pub tensors: usize,
    pub source_tensors: usize,
    /// Size of the written file.
    pub bytes: u64,
    pub source_bytes: u64,
    /// Bytes read from the source (header included).
    pub fetched_bytes: u64,
}

/// Whether a tensor belongs to the stage `[start, end)` of a model with `layers` blocks.
pub fn stage_holds_tensor(
    name: &str,
    start: u32,
    end: u32,
    layers: u32,
    has_output_weight: bool,
) -> bool {
    if start >= end {
        return false;
    }
    let first = start == 0;
    let last = end == layers;
    if let Some(block) = block_of(name) {
        // blocks past the layer count belong to heads that go with the output
        return if block < layers {
            (start..end).contains(&block)
        } else {
            last
        };
    }
    if name == "token_embd.weight" {
        return first || (last && !has_output_weight);
    }
    if name == "output.weight" || name == "output.bias" || name.starts_with("output_norm.") {
        return last;
    }
    true
}

fn block_of(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (digits, _) = rest.split_once('.')?;
    digits.parse().ok()
}

fn read_layout(
    source: &dyn ByteSource,
    source_len: u64,
) -> Result<(GgufLayout, Vec<u8>), StageFileError> {
    let mut want = FIRST_HEADER_READ;
    loop {
        let len = want.min(source_len);
        let prefix = source.read_range(0, len)?;
        match parse_gguf_prefix(&prefix, source_len) {
            Ok(layout) => return Ok((layout, prefix)),
            Err(GgufError::UnexpectedEof) if len < source_len => {
                if want >= MAX_HEADER_READ {
                    return Err(StageFileError::HeaderTooLarge);
                }
                want *= 2;
            }
            Err(err) => return Err(err.into()),
        }
    }
}

fn block_count(layout: &GgufLayout) -> Option<u32> {
    let metadata = &layout.header.metadata;
    let arch = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(arch)) => arch.as_str(),
        _ => return None,
    };
    match metadata.get(&format!("{arch}.block_count")) {
        Some(GgufMetadataValue::U64(count)) => u32::try_from(*count).ok(),
        _ => None,
    }
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_u32_entry(out: &mut Vec<u8>, key: &str, value: u32) {
    const GGUF_TYPE_UINT32: u32 = 4;
    put_string(out, key);
    out.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

/// The end of a partial file is not trusted: a machine that lost power may have left it
/// unwritten. This much is fetched again.
const RESUME_ROLLBACK: u64 = if cfg!(test) { 8 } else { 1 << 20 };

/// The length from which the partial file at `partial` can go on, when it starts with `header`.
fn resumable_len(partial: &Path, header: &[u8]) -> Option<u64> {
    let mut file = File::open(partial).ok()?;
    let len = file.metadata().ok()?.len();
    let mut start = vec![0u8; header.len()];
    file.read_exact(&mut start).ok()?;
    (start == header).then(|| len.saturating_sub(RESUME_ROLLBACK).max(header.len() as u64))
}

fn align_to(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// Builds the stage file for layers `[start, end)` at `dest`, reading only the header and the
/// tensors of the stage from `source`. The file appears at `dest` only when it is complete.
pub fn write_stage_file(
    source: &dyn ByteSource,
    start: u32,
    end: u32,
    dest: &Path,
) -> Result<StageFileReport, StageFileError> {
    let source_len = source.len()?;
    let (layout, prefix) = read_layout(source, source_len)?;
    let metadata = &layout.header.metadata;
    let mut source_metadata = &prefix[layout.metadata_start as usize..layout.metadata_end as usize];
    let mut source_metadata_count = layout.metadata_count;
    if let (Some(GgufMetadataValue::U64(from)), Some(GgufMetadataValue::U64(to))) =
        (metadata.get(STAGE_START_KEY), metadata.get(STAGE_END_KEY))
    {
        // A stage holds all of a narrower stage: layers a machine has need not come over the
        // network again when the plan gives it fewer of them. The two entries that say what
        // the source holds are the last ones this writer put, and they go.
        let mut said = Vec::new();
        put_u32_entry(&mut said, STAGE_START_KEY, *from as u32);
        put_u32_entry(&mut said, STAGE_END_KEY, *to as u32);
        let covers = *from <= u64::from(start) && u64::from(end) <= *to;
        if !covers || !source_metadata.ends_with(&said) {
            return Err(StageFileError::SourceIsStage {
                start: *from,
                end: *to,
            });
        }
        source_metadata = &source_metadata[..source_metadata.len() - said.len()];
        source_metadata_count -= 2;
    }
    let layers = block_count(&layout).ok_or(StageFileError::NoBlockCount)?;
    // 0..0 is the vocabulary file: every metadata value and no tensor
    let vocabulary_only = start == 0 && end == 0;
    if (start >= end && !vocabulary_only) || end > layers {
        return Err(StageFileError::BadRange { start, end, layers });
    }

    let alignment = layout.header.alignment.max(1);
    let has_output_weight = layout
        .header
        .tensors
        .iter()
        .any(|tensor| tensor.name == "output.weight");
    let mut kept: Vec<&GgufTensorInfo> = layout
        .header
        .tensors
        .iter()
        .filter(|tensor| stage_holds_tensor(&tensor.name, start, end, layers, has_output_weight))
        .collect();
    kept.sort_by_key(|tensor| tensor.offset);
    for tensor in &kept {
        let fits = layout
            .data_start
            .checked_add(tensor.offset)
            .and_then(|at| at.checked_add(tensor.byte_len))
            .is_some_and(|tensor_end| tensor_end <= source_len);
        if !fits {
            return Err(StageFileError::TensorOutOfBounds(tensor.name.clone()));
        }
    }

    // Tensors that sit next to each other in the source are copied as one run, padding included,
    // so a stage costs a handful of range requests instead of one per tensor.
    struct Run {
        source_offset: u64,
        len: u64,
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut new_offsets = Vec::with_capacity(kept.len());
    let mut cursor = 0u64;
    for tensor in &kept {
        match runs.last_mut() {
            Some(run) if run.source_offset + run.len == tensor.offset => {
                new_offsets.push(cursor);
                run.len += tensor.byte_len;
            }
            _ => {
                cursor = align_to(cursor, alignment);
                new_offsets.push(cursor);
                runs.push(Run {
                    source_offset: tensor.offset,
                    len: tensor.byte_len,
                });
            }
        }
        cursor += tensor.byte_len;
    }

    let mut header = Vec::new();
    header.extend_from_slice(crate::gguf::GGUF_MAGIC);
    header.extend_from_slice(&layout.header.version.to_le_bytes());
    header.extend_from_slice(&(kept.len() as u64).to_le_bytes());
    header.extend_from_slice(&(source_metadata_count + 2).to_le_bytes());
    header.extend_from_slice(source_metadata);
    put_u32_entry(&mut header, STAGE_START_KEY, start);
    put_u32_entry(&mut header, STAGE_END_KEY, end);
    for (tensor, offset) in kept.iter().zip(&new_offsets) {
        put_string(&mut header, &tensor.name);
        header.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            header.extend_from_slice(&dimension.to_le_bytes());
        }
        header.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
    }
    header.resize(align_to(header.len() as u64, alignment) as usize, 0);

    let mut partial = dest.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    // what a cut download left behind is room already taken
    let on_disk = std::fs::metadata(&partial).map_or(0, |partial| partial.len());
    let needed =
        (header.len() as u64 + cursor + runs.len() as u64 * alignment).saturating_sub(on_disk);
    if let Some(usable) = usable_disk_bytes(dest) {
        if needed > usable {
            return Err(StageFileError::NoSpace {
                needed_mb: needed.div_ceil(1024 * 1024),
                free_mb: usable / (1024 * 1024),
            });
        }
    }

    // A download that was cut (a restart, an update, a line that dropped) goes on from what is
    // already in the partial file. The file is the same byte for byte for the same model and
    // the same layers, and its header says which those are.
    let resume_at = resumable_len(&partial, &header);
    let result = (|| -> Result<u64, StageFileError> {
        let file = match resume_at {
            Some(len) => {
                let mut file = OpenOptions::new().write(true).open(&partial)?;
                file.set_len(len)?;
                file.seek(SeekFrom::End(0))?;
                file
            }
            None => File::create(&partial)?,
        };
        let mut out = BufWriter::with_capacity(1 << 20, file);
        if resume_at.is_none() {
            out.write_all(&header)?;
        }
        // where the file already ends, counted from the end of the header
        let have = resume_at.map_or(0, |len| len - header.len() as u64);
        let mut written = 0u64;
        let mut fetched = prefix.len() as u64;
        for run in &runs {
            let padding = align_to(written, alignment) - written;
            let run_start = written + padding;
            let run_end = run_start + run.len;
            if have < run_end {
                let missing_padding = run_start.saturating_sub(have.max(written));
                out.write_all(&vec![0u8; missing_padding as usize])?;
                let skip = have.saturating_sub(run_start);
                source.copy_range(
                    layout.data_start + run.source_offset + skip,
                    run.len - skip,
                    &mut out,
                )?;
                fetched += run.len - skip;
            }
            written = run_end;
        }
        out.into_inner()
            .map_err(|err| err.into_error())?
            .sync_all()?;
        Ok(fetched)
    })();
    let fetched = match result {
        Ok(fetched) => fetched,
        // what arrived stays on disk for the next attempt; a full disk needs the room back
        Err(err) => {
            if matches!(&err, StageFileError::Io(io) if io.kind() == io::ErrorKind::StorageFull) {
                let _ = std::fs::remove_file(&partial);
            }
            return Err(err);
        }
    };
    std::fs::rename(&partial, dest)?;

    Ok(StageFileReport {
        layer_count: layers,
        tensors: kept.len(),
        source_tensors: layout.header.tensors.len(),
        bytes: std::fs::metadata(dest)?.len(),
        source_bytes: source_len,
        fetched_bytes: fetched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parse_gguf_file;

    fn string(out: &mut Vec<u8>, value: &str) {
        put_string(out, value);
    }

    /// A tiny 3-block model: every tensor holds 40 bytes of its own marker value.
    fn sample_model(with_output_weight: bool) -> (Vec<u8>, Vec<String>) {
        let mut names = vec!["token_embd.weight".to_string()];
        for block in 0..3 {
            names.push(format!("blk.{block}.attn_q.weight"));
            names.push(format!("blk.{block}.ffn_up.weight"));
        }
        names.push("output_norm.weight".into());
        if with_output_weight {
            names.push("output.weight".into());
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(names.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, "qwen35moe");
        put_u32_entry(&mut bytes, "qwen35moe.block_count", 3);
        put_u32_entry(&mut bytes, "qwen35moe.embedding_length", 10);
        for (index, name) in names.iter().enumerate() {
            string(&mut bytes, name);
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&10u64.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes()); // f32
            bytes.extend_from_slice(&(index as u64 * 64).to_le_bytes());
        }
        bytes.resize(align_to(bytes.len() as u64, 32) as usize, 0);
        for index in 0..names.len() {
            let mut data = vec![index as u8 + 1; 40];
            // every tensor but the last is padded to the 32-byte alignment
            if index + 1 < names.len() {
                data.resize(64, 0);
            }
            bytes.extend_from_slice(&data);
        }
        (bytes, names)
    }

    fn build(
        with_output_weight: bool,
        start: u32,
        end: u32,
    ) -> (Vec<String>, Vec<u8>, StageFileReport, Vec<String>) {
        let dir = std::env::temp_dir().join(format!(
            "bitty-stage-file-{}-{with_output_weight}-{start}-{end}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, names) = sample_model(with_output_weight);
        let source_path = dir.join("model.gguf");
        std::fs::write(&source_path, &model).unwrap();
        let dest = dir.join("stage.gguf");
        let report = write_stage_file(&FileSource::new(&source_path), start, end, &dest).unwrap();
        let parsed = parse_gguf_file(&dest).unwrap();
        let bytes = std::fs::read(&dest).unwrap();
        let layout = parse_gguf_prefix(&bytes, bytes.len() as u64).unwrap();
        assert_eq!(
            parsed.metadata.get(STAGE_START_KEY),
            Some(&GgufMetadataValue::U64(start as u64))
        );
        assert_eq!(
            parsed.metadata.get(STAGE_END_KEY),
            Some(&GgufMetadataValue::U64(end as u64))
        );
        // every kept tensor still holds its own marker bytes at an aligned offset
        for tensor in &parsed.tensors {
            let marker = names.iter().position(|name| name == &tensor.name).unwrap() as u8 + 1;
            let at = (layout.data_start + tensor.offset) as usize;
            assert_eq!(tensor.offset % 32, 0);
            assert_eq!(&bytes[at..at + 40], &[marker; 40][..], "{}", tensor.name);
        }
        assert!(!dest.with_extension("gguf.partial").exists());
        let kept = parsed
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        std::fs::remove_dir_all(&dir).unwrap();
        (kept, bytes, report, names)
    }

    #[test]
    fn a_wider_stage_gives_a_narrower_one() {
        let dir =
            std::env::temp_dir().join(format!("bitty-stage-file-{}-narrow", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, _) = sample_model(true);
        let source_path = dir.join("model.gguf");
        std::fs::write(&source_path, &model).unwrap();
        let wide = dir.join("wide.gguf");
        write_stage_file(&FileSource::new(&source_path), 1, 3, &wide).unwrap();
        let direct = dir.join("direct.gguf");
        write_stage_file(&FileSource::new(&source_path), 2, 3, &direct).unwrap();
        let narrow = dir.join("narrow.gguf");
        write_stage_file(&FileSource::new(&wide), 2, 3, &narrow).unwrap();

        let tensors = |path: &Path| {
            let bytes = std::fs::read(path).unwrap();
            let layout = parse_gguf_prefix(&bytes, bytes.len() as u64).unwrap();
            assert_eq!(
                layout.header.metadata.get(STAGE_START_KEY),
                Some(&GgufMetadataValue::U64(2))
            );
            assert_eq!(
                layout.header.metadata.get(STAGE_END_KEY),
                Some(&GgufMetadataValue::U64(3))
            );
            layout
                .header
                .tensors
                .iter()
                .map(|tensor| {
                    let at = (layout.data_start + tensor.offset) as usize;
                    (
                        tensor.name.clone(),
                        bytes[at..at + tensor.byte_len as usize].to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(tensors(&narrow), tensors(&direct));
        assert_eq!(
            parse_gguf_file(&narrow).unwrap().metadata.len(),
            parse_gguf_file(&direct).unwrap().metadata.len()
        );
        // the layers a stage does not hold cannot come out of it
        assert!(matches!(
            write_stage_file(&FileSource::new(&wide), 0, 2, &dir.join("no.gguf")),
            Err(StageFileError::SourceIsStage { start: 1, end: 3 })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_cut_download_goes_on_from_where_it_was() {
        let dir = std::env::temp_dir().join(format!("bitty-stage-resume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (model, _) = sample_model(true);
        let source_path = dir.join("model.gguf");
        std::fs::write(&source_path, &model).unwrap();
        let source = FileSource::new(&source_path);
        let whole_path = dir.join("whole.gguf");
        let whole_report = write_stage_file(&source, 0, 3, &whole_path).unwrap();
        let whole = std::fs::read(&whole_path).unwrap();

        let dest = dir.join("stage.gguf");
        let partial = dir.join("stage.gguf.partial");
        // cut at every length: inside the header, inside the padding, inside a tensor
        for cut in 0..whole.len() {
            std::fs::write(&partial, &whole[..cut]).unwrap();
            let report = write_stage_file(&source, 0, 3, &dest).unwrap();
            assert_eq!(std::fs::read(&dest).unwrap(), whole, "cut at {cut}");
            assert!(!partial.exists());
            assert!(report.fetched_bytes <= whole_report.fetched_bytes);
            std::fs::remove_file(&dest).unwrap();
        }
        // most of the file was there: only the end is fetched again
        std::fs::write(&partial, &whole[..whole.len() - 4]).unwrap();
        let report = write_stage_file(&source, 0, 3, &dest).unwrap();
        assert!(report.fetched_bytes < whole_report.fetched_bytes);

        // a partial file of other layers is not continued
        std::fs::write(&partial, &whole[..whole.len() - 4]).unwrap();
        write_stage_file(&source, 1, 2, &dest).unwrap();
        let other = dir.join("other.gguf");
        write_stage_file(&source, 1, 2, &other).unwrap();
        assert_eq!(
            std::fs::read(&dest).unwrap(),
            std::fs::read(&other).unwrap()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_range_gives_a_vocabulary_file_without_tensors() {
        let (kept, bytes, report, _) = build(true, 0, 0);
        assert!(kept.is_empty());
        assert_eq!(report.tensors, 0);
        assert_eq!(report.bytes, bytes.len() as u64);
    }

    #[test]
    fn middle_stage_holds_only_its_blocks() {
        let (kept, bytes, report, names) = build(true, 1, 2);
        assert_eq!(kept, ["blk.1.attn_q.weight", "blk.1.ffn_up.weight"]);
        assert_eq!(report.tensors, 2);
        assert_eq!(report.source_tensors, names.len());
        assert_eq!(report.bytes, bytes.len() as u64);
        assert!(report.bytes < report.source_bytes);
    }

    #[test]
    fn first_stage_holds_the_embeddings_and_last_the_head() {
        let (first, ..) = build(true, 0, 1);
        assert_eq!(
            first,
            [
                "token_embd.weight",
                "blk.0.attn_q.weight",
                "blk.0.ffn_up.weight"
            ]
        );
        let (last, ..) = build(true, 2, 3);
        assert_eq!(
            last,
            [
                "blk.2.attn_q.weight",
                "blk.2.ffn_up.weight",
                "output_norm.weight",
                "output.weight"
            ]
        );
    }

    #[test]
    fn tied_output_keeps_the_embeddings_in_the_last_stage() {
        let (last, ..) = build(false, 2, 3);
        assert_eq!(
            last,
            [
                "token_embd.weight",
                "blk.2.attn_q.weight",
                "blk.2.ffn_up.weight",
                "output_norm.weight"
            ]
        );
    }

    #[test]
    fn rejects_bad_ranges_and_stage_sources() {
        let dir =
            std::env::temp_dir().join(format!("bitty-stage-file-{}-reject", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, _) = sample_model(true);
        let source_path = dir.join("model.gguf");
        std::fs::write(&source_path, &model).unwrap();
        let source = FileSource::new(&source_path);
        let stage = dir.join("stage.gguf");
        assert!(matches!(
            write_stage_file(&source, 2, 4, &stage),
            Err(StageFileError::BadRange { .. })
        ));
        assert!(matches!(
            write_stage_file(&source, 2, 2, &stage),
            Err(StageFileError::BadRange { .. })
        ));
        write_stage_file(&source, 0, 1, &stage).unwrap();
        assert!(matches!(
            write_stage_file(&FileSource::new(&stage), 1, 2, &dir.join("again.gguf")),
            Err(StageFileError::SourceIsStage { start: 0, end: 1 })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod resume_tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    /// A server that honours ranges but closes every connection after `cut` body bytes.
    fn flaky_server(data: Vec<u8>, cut: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut range = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        let (start, end) = value.trim().split_once('-').unwrap();
                        range = Some((
                            start.parse::<usize>().unwrap(),
                            end.parse::<usize>().unwrap(),
                        ));
                    }
                }
                let Some((start, end)) = range else { continue };
                let body = &data[start..=end];
                let head = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                    body.len(), start, end, data.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body[..body.len().min(cut)]);
            }
        });
        url
    }

    #[test]
    fn a_download_that_breaks_goes_on_from_where_it_stopped() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let source = HttpSource::new(flaky_server(data.clone(), 4_000));
        let bytes = source.read_range(100, 9_000).unwrap();
        assert_eq!(bytes, &data[100..9_100]);
    }
}

#[cfg(test)]
mod long_path_tests {
    use super::without_long_path_mark;
    use std::path::PathBuf;

    #[test]
    fn a_windows_full_path_loses_its_mark_and_others_stay() {
        let plain = |text: &str| without_long_path_mark(PathBuf::from(text));
        assert_eq!(
            plain(r"\\?\C:\Users\a\.bitty"),
            PathBuf::from(r"C:\Users\a\.bitty")
        );
        assert_eq!(
            plain(r"\\?\UNC\server\share"),
            PathBuf::from(r"\\?\UNC\server\share")
        );
        assert_eq!(plain("/home/a/.bitty"), PathBuf::from("/home/a/.bitty"));
    }
}
