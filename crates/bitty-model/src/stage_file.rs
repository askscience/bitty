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
use std::fs::File;
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
        Self { url: url.into(), cancel: None }
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
            .args(["--retry", "5", "--retry-delay", "2", "--proto", "=https,http"]);
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
            return Err(StageFileError::Fetch(String::from_utf8_lossy(&output.stderr).trim().into()));
        }
        let headers = String::from_utf8_lossy(&output.stdout);
        headers
            .lines()
            .filter_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.trim().eq_ignore_ascii_case("content-range").then_some(value)
            })
            .last()
            .and_then(|value| value.rsplit('/').next()?.trim().parse().ok())
            .ok_or_else(|| StageFileError::Fetch("the server does not support range requests".into()))
    }

    fn copy_range(&self, offset: u64, len: u64, out: &mut dyn Write) -> Result<(), StageFileError> {
        if len == 0 {
            return Ok(());
        }
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
        let copied = (|| -> io::Result<u64> {
            let mut total = 0u64;
            loop {
                if self.cancel.as_ref().is_some_and(|cancel| cancel()) {
                    cancelled = true;
                    return Ok(total);
                }
                let read = match limited.read(&mut buffer) {
                    Ok(0) => return Ok(total),
                    Ok(read) => read,
                    Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                    Err(err) => return Err(err),
                };
                out.write_all(&buffer[..read])?;
                total += read as u64;
            }
        })();
        drop(stdout);
        if cancelled || matches!(copied, Ok(n) if n > len) {
            let _ = child.kill();
        }
        let status = child.wait()?;
        if cancelled {
            return Err(StageFileError::Cancelled);
        }
        let copied = copied?;
        if copied != len || !status.success() {
            return Err(StageFileError::Fetch(format!(
                "expected {len} bytes at offset {offset}, received {copied}"
            )));
        }
        Ok(())
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
pub fn stage_holds_tensor(name: &str, start: u32, end: u32, layers: u32, has_output_weight: bool) -> bool {
    if start >= end {
        return false;
    }
    let first = start == 0;
    let last = end == layers;
    if let Some(block) = block_of(name) {
        // blocks past the layer count belong to heads that go with the output
        return if block < layers { (start..end).contains(&block) } else { last };
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

fn read_layout(source: &dyn ByteSource, source_len: u64) -> Result<(GgufLayout, Vec<u8>), StageFileError> {
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
    if let (Some(GgufMetadataValue::U64(from)), Some(GgufMetadataValue::U64(to))) =
        (metadata.get(STAGE_START_KEY), metadata.get(STAGE_END_KEY))
    {
        return Err(StageFileError::SourceIsStage { start: *from, end: *to });
    }
    let layers = block_count(&layout).ok_or(StageFileError::NoBlockCount)?;
    // 0..0 is the vocabulary file: every metadata value and no tensor
    let vocabulary_only = start == 0 && end == 0;
    if (start >= end && !vocabulary_only) || end > layers {
        return Err(StageFileError::BadRange { start, end, layers });
    }

    let alignment = layout.header.alignment.max(1);
    let has_output_weight = layout.header.tensors.iter().any(|tensor| tensor.name == "output.weight");
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
    header.extend_from_slice(&(layout.metadata_count + 2).to_le_bytes());
    header.extend_from_slice(&prefix[layout.metadata_start as usize..layout.metadata_end as usize]);
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
    let result = (|| -> Result<u64, StageFileError> {
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&partial)?);
        out.write_all(&header)?;
        let mut written = 0u64;
        let mut fetched = prefix.len() as u64;
        for run in &runs {
            let padding = align_to(written, alignment) - written;
            out.write_all(&vec![0u8; padding as usize])?;
            source.copy_range(layout.data_start + run.source_offset, run.len, &mut out)?;
            written += padding + run.len;
            fetched += run.len;
        }
        out.into_inner().map_err(|err| err.into_error())?.sync_all()?;
        Ok(fetched)
    })();
    let fetched = match result {
        Ok(fetched) => fetched,
        Err(err) => {
            let _ = std::fs::remove_file(&partial);
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

    fn build(with_output_weight: bool, start: u32, end: u32) -> (Vec<String>, Vec<u8>, StageFileReport, Vec<String>) {
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
        assert_eq!(parsed.metadata.get(STAGE_START_KEY), Some(&GgufMetadataValue::U64(start as u64)));
        assert_eq!(parsed.metadata.get(STAGE_END_KEY), Some(&GgufMetadataValue::U64(end as u64)));
        // every kept tensor still holds its own marker bytes at an aligned offset
        for tensor in &parsed.tensors {
            let marker = names.iter().position(|name| name == &tensor.name).unwrap() as u8 + 1;
            let at = (layout.data_start + tensor.offset) as usize;
            assert_eq!(tensor.offset % 32, 0);
            assert_eq!(&bytes[at..at + 40], &[marker; 40][..], "{}", tensor.name);
        }
        assert!(!dest.with_extension("gguf.partial").exists());
        let kept = parsed.tensors.iter().map(|tensor| tensor.name.clone()).collect();
        std::fs::remove_dir_all(&dir).unwrap();
        (kept, bytes, report, names)
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
        assert_eq!(first, ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_up.weight"]);
        let (last, ..) = build(true, 2, 3);
        assert_eq!(
            last,
            ["blk.2.attn_q.weight", "blk.2.ffn_up.weight", "output_norm.weight", "output.weight"]
        );
    }

    #[test]
    fn tied_output_keeps_the_embeddings_in_the_last_stage() {
        let (last, ..) = build(false, 2, 3);
        assert_eq!(
            last,
            ["token_embd.weight", "blk.2.attn_q.weight", "blk.2.ffn_up.weight", "output_norm.weight"]
        );
    }

    #[test]
    fn rejects_bad_ranges_and_stage_sources() {
        let dir = std::env::temp_dir().join(format!("bitty-stage-file-{}-reject", std::process::id()));
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
            write_stage_file(&FileSource::new(&stage), 0, 1, &dir.join("again.gguf")),
            Err(StageFileError::SourceIsStage { start: 0, end: 1 })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
