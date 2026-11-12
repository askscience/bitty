//! The graphics cards of this machine, as the engine sees them.
//!
//! A release carries the engine once more, built with llama.cpp's Vulkan backend:
//! `bitty-stage-vulkan`. It is a file of its own because it needs the Vulkan library of the
//! system to start at all, and a machine without a card or without drivers must still run the
//! plain engine. Whether it works here is not guessed from the hardware: bitty starts it with
//! `--list-devices` and reads what it found.

use crate::stage_process::{
    cpu_level, find_engine_variant, CpuLevel, StageProcess, StageProcessError,
};
use bitty_model::TensorMetadata;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long the engine may take to list its cards. A driver that hangs must not hang the node.
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// Video memory left alone on top of the weights: the context of the sessions, the buffers of
/// the computation and whatever else the desktop asks for while the node runs.
/// `BITTY_GPU_RESERVE_MB` changes it.
const DEFAULT_RESERVE_MB: u64 = 512;

/// How much faster than the processor a card computes a layer that is wholly on it, where
/// nothing was measured: 27 ms against 11 for five layers of Qwen3.5-35B on a Radeon RX 5500M.
/// It only weighs the node for the leader's plan; whether this card is used at all is decided
/// by timing it.
pub const ASSUMED_GPU_SPEEDUP: f64 = 2.5;

/// The card must beat the processor by this much to be kept: a tie is not worth the video
/// memory, and the timing is short.
const GPU_MUST_WIN_BY: f64 = 0.9;

/// Tokens the timing decodes after the first, which is left out: on a card it builds the
/// programs of every operation.
const TIMED_TOKENS: u32 = 4;

/// The session the timing uses. Requests count their sessions up from 1.
const TIMING_SESSION: u64 = u64::MAX;

/// What `BITTY_GPU` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuMode {
    /// Never start the engine for graphics cards.
    Off,
    /// Use a card of its own when there is one. The default.
    Auto,
    /// Also use a card built into the processor when there is no other.
    Integrated,
}

pub fn gpu_mode() -> GpuMode {
    parse_gpu_mode(std::env::var("BITTY_GPU").ok().as_deref())
}

fn parse_gpu_mode(value: Option<&str>) -> GpuMode {
    match value
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("off" | "0" | "no" | "false" | "cpu" | "none") => GpuMode::Off,
        Some("igpu" | "integrated" | "all") => GpuMode::Integrated,
        _ => GpuMode::Auto,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuDevice {
    /// The name the backend gives it, such as `Vulkan0`.
    pub name: String,
    /// The name of the card.
    pub description: String,
    /// Built into the processor: it shares the memory of the machine.
    pub integrated: bool,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

impl GpuDevice {
    /// A driver that computes on the processor under the name of a card. It is slower than
    /// the plain engine.
    fn is_software(&self) -> bool {
        let description = self.description.to_ascii_lowercase();
        ["llvmpipe", "lavapipe", "swiftshader", "software"]
            .iter()
            .any(|name| description.contains(name))
    }
}

/// An engine that can compute on a graphics card, and the cards it found.
#[derive(Clone, Debug)]
pub struct GpuEngine {
    pub path: PathBuf,
    pub devices: Vec<GpuDevice>,
}

impl GpuEngine {
    /// The cards llama.cpp loads layers into when it is left to choose: every card of its
    /// own, and a built-in one only when there is no other.
    pub fn used_devices(&self) -> Vec<&GpuDevice> {
        let own: Vec<&GpuDevice> = self.devices.iter().filter(|gpu| !gpu.integrated).collect();
        if own.is_empty() {
            self.devices.iter().collect()
        } else {
            own
        }
    }

    /// Whether bitty may put layers on these cards without being told to.
    pub fn usable(&self, mode: GpuMode) -> bool {
        match mode {
            GpuMode::Off => false,
            GpuMode::Auto => self.devices.iter().any(|gpu| !gpu.integrated),
            GpuMode::Integrated => !self.devices.is_empty(),
        }
    }

    /// The video memory free right now on the cards that would be used.
    pub fn free_bytes(&self) -> u64 {
        self.used_devices().iter().map(|gpu| gpu.free_bytes).sum()
    }

    /// The video memory of the cards that would be used.
    pub fn total_bytes(&self) -> u64 {
        self.used_devices().iter().map(|gpu| gpu.total_bytes).sum()
    }

    /// What the weights of a stage may take of it: the free memory less the reserve.
    pub fn weight_budget_bytes(&self) -> u64 {
        let reserve_mb = std::env::var("BITTY_GPU_RESERVE_MB")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_RESERVE_MB);
        self.free_bytes()
            .saturating_sub(reserve_mb.saturating_mul(1024 * 1024))
    }

    /// One line for the log.
    pub fn summary(&self) -> String {
        self.devices
            .iter()
            .map(|gpu| {
                format!(
                    "{}{}, {} of {} MB free",
                    gpu.description,
                    if gpu.integrated { " (built in)" } else { "" },
                    gpu.free_bytes / 1024 / 1024,
                    gpu.total_bytes / 1024 / 1024
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Finds the engine built for graphics cards: `BITTY_STAGE_BIN` when it is set (a build of
/// one's own may have any backend), else `bitty-stage-vulkan` next to this executable or on
/// `PATH`. `bitty-stage-vulkan` needs AVX2 for the layers it leaves on the processor,
/// `bitty-stage-vulkan-avx` only AVX, `bitty-stage-vulkan-sse` runs on every x86-64 processor.
pub fn find_gpu_engine() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("BITTY_STAGE_BIN") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    match cpu_level() {
        CpuLevel::Avx2 => find_engine_variant(&["-vulkan", "-vulkan-avx", "-vulkan-sse"]),
        CpuLevel::Avx => find_engine_variant(&["-vulkan-avx", "-vulkan-sse"]),
        CpuLevel::Sse => find_engine_variant(&["-vulkan-sse"]),
    }
}

/// Asks the engine for graphics cards which cards it can compute on. `None` when there is no
/// such engine, when it does not start here (no Vulkan library, no driver) or when it finds
/// no card. Cards that are the processor under another name are left out.
///
/// This starts a process and takes a moment; the free memory it reports is the one of that
/// moment, so ask again before every load.
pub fn probe_gpu_engine() -> Option<GpuEngine> {
    let path = find_gpu_engine()?;
    // Windows answers a program that lacks a library with a window on the screen, which
    // nobody is there to close: the engine of the release is not started without its library.
    #[cfg(windows)]
    if std::env::var_os("BITTY_STAGE_BIN").is_none() {
        let system = std::env::var_os("SystemRoot").map(PathBuf::from)?;
        if !system.join("System32").join("vulkan-1.dll").is_file() {
            return None;
        }
    }
    let devices = list_devices(&path)?;
    (!devices.is_empty()).then_some(GpuEngine { path, devices })
}

fn list_devices(engine: &Path) -> Option<Vec<GpuDevice>> {
    let mut child = Command::new(engine)
        .arg("--list-devices")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // the list is a few lines: it fits in the pipe, so it can be read after the exit
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < PROBE_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(50));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    // an engine built before `--list-devices` ends with an error here
    if !status.success() {
        return None;
    }
    let mut output = String::new();
    child.stdout.take()?.read_to_string(&mut output).ok()?;
    Some(parse_devices(&output))
}

/// Reads the lines `device <kind> <free> <total> <name> <description>`, separated by tabs.
/// Anything else a driver may have printed is skipped.
fn parse_devices(output: &str) -> Vec<GpuDevice> {
    output
        .lines()
        .filter_map(|line| {
            let mut fields = line.trim_end().split('\t');
            if fields.next()? != "device" {
                return None;
            }
            let integrated = match fields.next()? {
                "gpu" => false,
                "igpu" => true,
                _ => return None,
            };
            let free_bytes = fields.next()?.parse().ok()?;
            let total_bytes = fields.next()?.parse().ok()?;
            let name = fields.next()?.to_string();
            let description = fields.next().unwrap_or_default().to_string();
            Some(GpuDevice {
                name,
                description,
                integrated,
                free_bytes,
                total_bytes,
            })
        })
        .filter(|gpu| !gpu.is_software())
        .collect()
}

/// Held while a process reads the free video memory and loads into it, so that two nodes of
/// one machine do not count the same memory twice. Released when the value is dropped, or
/// when the process ends. A machine where the file cannot be made goes on without it.
pub struct GpuLock(#[allow(dead_code)] Option<std::fs::File>);

pub fn gpu_lock() -> GpuLock {
    let path = std::env::temp_dir().join("bitty-gpu.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
        .ok()
        .filter(|file| file.lock().is_ok());
    GpuLock(file)
}

/// The time an engine takes for one token of its range, the best of a few. The input is
/// made up: the time of a layer does not depend on what the numbers are. Leaves no session
/// behind.
pub fn time_engine(process: &mut StageProcess) -> Result<Duration, StageProcessError> {
    let info = process.info();
    // small values of both signs; all zeros would be a special case for some operations
    let hidden: Vec<u8> = (0..info.hidden_size)
        .flat_map(|index| (((index % 7) as f32 - 3.0) * 0.01).to_le_bytes())
        .collect();
    let mut best = Duration::MAX;
    let mut run = |process: &mut StageProcess| {
        for position in 0..=TIMED_TOKENS {
            let started = Instant::now();
            if info.is_first() {
                process.decode_tokens(TIMING_SESSION, position, &[1], false)?;
            } else {
                process.decode_hidden(TIMING_SESSION, position, 1, &hidden, false)?;
            }
            if position > 0 {
                best = best.min(started.elapsed());
            }
        }
        Ok(())
    };
    let result = run(process);
    let ended = process.end_session(TIMING_SESSION);
    result.and(ended).map(|_| best)
}

/// Whether a card that took `gpu` for a token is kept over a processor that took `cpu`.
pub fn gpu_is_faster(cpu: Duration, gpu: Duration) -> bool {
    gpu.as_secs_f64() <= cpu.as_secs_f64() * GPU_MUST_WIN_BY
}

/// How much faster a node is than its processor alone when `share` of its layers (0 to 1)
/// is on a card: the times of the two parts add up.
pub fn speedup_with_share(share: f64) -> f64 {
    let share = share.clamp(0.0, 1.0);
    1.0 / ((1.0 - share) + share / ASSUMED_GPU_SPEEDUP)
}

/// What of a stage fits in the video memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuFit {
    /// Layers of the range, counted from its top.
    pub layers: u32,
    /// The output layer, which only the final stage has. llama.cpp places it before any block,
    /// so `layers > 0` on the final stage implies it.
    pub output: bool,
    /// The bytes of weights these take.
    pub bytes: u64,
}

/// How many layers of `[start, end)` fit in `budget` bytes, from the top of the range down,
/// by the sizes of the tensors in the file. On the final stage the output layer goes first:
/// it is the most expensive single step of the model. A layer the file says nothing about
/// ends the count, since its size is not known.
pub fn layers_that_fit(
    tensors: &[TensorMetadata],
    start: u32,
    end: u32,
    layer_count: u32,
    budget: u64,
) -> GpuFit {
    let mut fit = GpuFit::default();
    if end == layer_count {
        let named = |prefix: &str| -> u64 {
            tensors
                .iter()
                .filter(|tensor| tensor.layer_id.is_none() && tensor.name.starts_with(prefix))
                .map(|tensor| tensor.byte_len)
                .sum()
        };
        let mut output = named("output");
        // a model whose output shares the matrix of the token embedding
        if !tensors.iter().any(|tensor| tensor.name == "output.weight") {
            output += named("token_embd.weight");
        }
        if output == 0 || output > budget {
            return fit;
        }
        fit.output = true;
        fit.bytes = output;
    }
    for layer in (start..end).rev() {
        let bytes: u64 = tensors
            .iter()
            .filter(|tensor| tensor.layer_id == Some(layer))
            .map(|tensor| tensor.byte_len)
            .sum();
        if bytes == 0 || fit.bytes + bytes > budget {
            break;
        }
        fit.bytes += bytes;
        fit.layers += 1;
    }
    fit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, layer_id: Option<u32>, byte_len: u64) -> TensorMetadata {
        TensorMetadata {
            name: name.into(),
            layer_id,
            dimensions: Vec::new(),
            ggml_type: 0,
            byte_len,
            offset: 0,
        }
    }

    fn model() -> Vec<TensorMetadata> {
        let mut tensors = vec![
            tensor("token_embd.weight", None, 200),
            tensor("output_norm.weight", None, 1),
            tensor("output.weight", None, 199),
        ];
        for layer in 0..8 {
            tensors.push(tensor(
                &format!("blk.{layer}.attn_q.weight"),
                Some(layer),
                60,
            ));
            tensors.push(tensor(
                &format!("blk.{layer}.ffn_up.weight"),
                Some(layer),
                40,
            ));
        }
        tensors
    }

    #[test]
    fn the_engine_list_is_read_and_software_cards_are_dropped() {
        let output = "ggml_vulkan: Found 3 Vulkan devices:\n\
            device\tgpu\t4187593113\t8573157376\tVulkan0\tAMD Radeon RX 5500M (RADV NAVI14)\n\
            device\tigpu\t1000\t2000\tVulkan1\tIntel(R) UHD Graphics 630\r\n\
            device\tgpu\t1\t2\tVulkan2\tllvmpipe (LLVM 19.1.7, 256 bits)\n\
            device\tgpu\tmany\t2\tVulkan3\tbroken\n";
        let devices = parse_devices(output);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].description, "AMD Radeon RX 5500M (RADV NAVI14)");
        assert_eq!(devices[0].free_bytes, 4_187_593_113);
        assert!(!devices[0].integrated);
        assert!(devices[1].integrated);

        let engine = GpuEngine {
            path: PathBuf::from("bitty-stage-vulkan"),
            devices,
        };
        // the card of its own is the only one llama.cpp loads into
        assert_eq!(engine.free_bytes(), 4_187_593_113);
        assert!(engine.usable(GpuMode::Auto));
        assert!(!engine.usable(GpuMode::Off));
    }

    #[test]
    fn a_built_in_card_is_used_only_when_asked() {
        let engine = GpuEngine {
            path: PathBuf::from("bitty-stage-vulkan"),
            devices: parse_devices("device\tigpu\t1000\t2000\tVulkan0\tIntel(R) UHD Graphics\n"),
        };
        assert!(!engine.usable(GpuMode::Auto));
        assert!(engine.usable(GpuMode::Integrated));
        assert_eq!(engine.free_bytes(), 1000);
    }

    #[test]
    fn a_card_must_clearly_win_to_be_kept() {
        let ms = Duration::from_millis;
        assert!(gpu_is_faster(ms(100), ms(40)));
        assert!(gpu_is_faster(ms(100), ms(90)));
        assert!(!gpu_is_faster(ms(100), ms(95)));
        assert!(!gpu_is_faster(ms(100), ms(300)));
    }

    #[test]
    fn a_small_share_on_the_card_adds_little_speed() {
        assert_eq!(speedup_with_share(0.0), 1.0);
        assert!((speedup_with_share(1.0) - ASSUMED_GPU_SPEEDUP).abs() < 1e-9);
        // one layer of ten on the card: 6% faster, not 2.5 times
        assert!((speedup_with_share(0.1) - 1.0 / 0.94).abs() < 1e-9);
        assert_eq!(speedup_with_share(7.0), speedup_with_share(1.0));
    }

    #[test]
    fn the_mode_is_read_from_the_setting() {
        assert_eq!(parse_gpu_mode(None), GpuMode::Auto);
        assert_eq!(parse_gpu_mode(Some("auto")), GpuMode::Auto);
        assert_eq!(parse_gpu_mode(Some(" OFF ")), GpuMode::Off);
        assert_eq!(parse_gpu_mode(Some("0")), GpuMode::Off);
        assert_eq!(parse_gpu_mode(Some("igpu")), GpuMode::Integrated);
    }

    #[test]
    fn layers_are_counted_from_the_top_of_the_range() {
        let tensors = model();
        // a middle stage: 100 bytes a layer
        assert_eq!(layers_that_fit(&tensors, 2, 6, 8, 250).layers, 2);
        assert_eq!(layers_that_fit(&tensors, 2, 6, 8, 99).layers, 0);
        let all = layers_that_fit(&tensors, 2, 6, 8, u64::MAX);
        assert_eq!((all.layers, all.output, all.bytes), (4, false, 400));
    }

    #[test]
    fn the_final_stage_places_the_output_layer_first() {
        let tensors = model();
        let fit = layers_that_fit(&tensors, 4, 8, 8, 320);
        assert_eq!((fit.layers, fit.output, fit.bytes), (1, true, 300));
        let fit = layers_that_fit(&tensors, 4, 8, 8, 250);
        assert_eq!((fit.layers, fit.output), (0, true));
        // not even the output layer fits: nothing goes to the card
        assert_eq!(layers_that_fit(&tensors, 4, 8, 8, 150), GpuFit::default());
    }

    #[test]
    fn a_tied_output_counts_the_token_embedding() {
        let tensors: Vec<_> = model()
            .into_iter()
            .filter(|tensor| tensor.name != "output.weight")
            .collect();
        let fit = layers_that_fit(&tensors, 4, 8, 8, 301);
        assert_eq!((fit.layers, fit.output, fit.bytes), (1, true, 301));
    }

    #[test]
    fn a_file_without_the_layers_puts_nothing_on_the_card() {
        // the vocabulary file of a node that fetches its stages
        let tensors = vec![tensor("token_embd.weight", None, 200)];
        assert_eq!(layers_that_fit(&tensors, 2, 6, 8, u64::MAX).layers, 0);
    }
}
