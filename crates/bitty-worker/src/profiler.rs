use bitty_protocol::GpuInfo;
use bitty_protocol::{HardwareProfile, NodeId, NodeTier};
use nvml_wrapper::Nvml;
use std::time::Instant;
use sysinfo::System;

#[derive(Clone, Debug, Default)]
struct GpuProbe {
    gpus: Vec<GpuInfo>,
    backend_type: Option<String>,
}

#[derive(Clone, Debug)]
pub struct HardwareProfiler {
    node_id: NodeId,
}

impl HardwareProfiler {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: NodeId::new(node_id),
        }
    }

    pub fn profile(&self) -> HardwareProfile {
        let mut system = System::new_all();
        system.refresh_memory();
        system.refresh_cpu_all();

        let cpu_count = cpu_count(&system);
        let cpu_tflops = estimate_cpu_tflops(cpu_count);
        let ram_mb = std::env::var("BITTY_RAM_MB")
            .ok()
            .and_then(|value| value.parse().ok())
            .or_else(|| total_ram_mb(&system))
            .unwrap_or(4096);
        let available_ram_mb = std::env::var("BITTY_AVAILABLE_RAM_MB")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| system.available_memory() / 1024 / 1024)
            .min(ram_mb);
        let gpu_probe = configured_gpu_probe();
        let gpus = gpu_probe.gpus;
        let vram_mb = std::env::var("BITTY_VRAM_MB")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| gpus.iter().map(|gpu| gpu.vram_mb).max().unwrap_or(0));
        // The engine loads into the card only the layers `BITTY_GPU_LAYERS` names. Without it
        // the card does no work and holds nothing, so the node is planned as a CPU node.
        let engine_uses_gpu = std::env::var("BITTY_GPU_LAYERS")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .is_some_and(|layers| layers > 0);
        let gpu_tflops = std::env::var("BITTY_GPU_TFLOPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| {
                if engine_uses_gpu {
                    estimate_gpu_tflops(&gpus, vram_mb)
                } else {
                    0.0
                }
            });
        let tier = tier_for(cpu_count, ram_mb, gpu_tflops, vram_mb);
        let backend_type = std::env::var("BITTY_BACKEND")
            .ok()
            .or(gpu_probe.backend_type)
            .unwrap_or_else(|| if gpu_tflops > 0.0 { "gpu" } else { "cpu" }.into());
        let max_layers = std::env::var("BITTY_MAX_LAYERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(u32::MAX);
        let role = std::env::var("BITTY_NODE_ROLE").unwrap_or_else(|_| "worker".into());
        let disabled = std::env::var("BITTY_DISABLE_MODEL_LAYERS")
            .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let layer_eligible = !disabled
            && max_layers != 0
            && !matches!(role.as_str(), "coordinator" | "client")
            && (gpu_tflops > 0.0 || (ram_mb >= 6144 && cpu_count >= 4));

        HardwareProfile {
            node_id: self.node_id.clone(),
            cpu_tflops,
            gpu_tflops,
            memory_gb: ram_mb as f64 / 1024.0,
            memory_bandwidth_gbps: memory_bandwidth_gbps(&system, ram_mb, gpu_tflops),
            disk_bandwidth_mbps: detect_disk_bandwidth_mbps(),
            network_rtt_ms: std::env::var("BITTY_NETWORK_RTT_MS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(25.0),
            uplink_mbps: std::env::var("BITTY_UPLINK_MBPS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(50.0),
            os: std::env::consts::OS.into(),
            tier,
            ram_mb,
            available_ram_mb,
            free_disk_mb: 0,
            build: String::new(),
            vram_mb,
            architecture: architecture(&system),
            gpus,
            os_reclaim_score: 0.0,
            worker_endpoint: std::env::var("BITTY_WORKER_ENDPOINT").unwrap_or_default(),
            model_path: std::env::var("BITTY_MODEL_PATH").unwrap_or_default(),
            backend_type,
            layer_eligible,
            max_layers,
        }
    }
}

fn cpu_count(system: &System) -> usize {
    let detected = system.cpus().len();
    if detected > 0 {
        detected
    } else {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
    }
}

/// The memory free right now, in MB.
pub fn available_ram_mb_now() -> u64 {
    let mut system = System::new();
    system.refresh_memory();
    system.available_memory() / 1024 / 1024
}

fn total_ram_mb(system: &System) -> Option<u64> {
    let bytes = system.total_memory();
    (bytes > 0).then_some(bytes / 1024 / 1024)
}

fn architecture(system: &System) -> String {
    let cpu_brand = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim())
        .filter(|brand| !brand.is_empty())
        .unwrap_or(std::env::consts::ARCH);
    format!("{} ({cpu_brand})", std::env::consts::ARCH)
}

fn estimate_cpu_tflops(cpu_count: usize) -> f64 {
    let started = Instant::now();
    let mut acc = 0.0_f64;
    for index in 0..20_000 {
        let x = index as f64;
        acc += (x.sin() * x.cos()).abs();
    }
    let elapsed = started.elapsed().as_secs_f64().max(0.000_001);
    let synthetic_gflops = (20_000.0 / elapsed) / 1_000_000_000.0;
    (cpu_count as f64 * 0.05).max(synthetic_gflops + acc.min(1.0) * 0.0)
}

fn tier_for(cpu_count: usize, ram_mb: u64, gpu_tflops: f64, vram_mb: u64) -> NodeTier {
    if gpu_tflops >= 20.0 && vram_mb >= 16_384 {
        NodeTier::S
    } else if gpu_tflops >= 8.0 && vram_mb >= 8_192 {
        NodeTier::A
    } else if gpu_tflops > 0.0 || (cpu_count >= 16 && ram_mb >= 16_384) {
        NodeTier::B
    } else if cpu_count >= 8 && ram_mb >= 8_192 {
        NodeTier::C
    } else {
        NodeTier::D
    }
}

fn configured_gpu_probe() -> GpuProbe {
    if let Ok(name) = std::env::var("BITTY_GPU_NAME") {
        let vram_mb = std::env::var("BITTY_VRAM_MB")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        return GpuProbe {
            backend_type: Some(preferred_backend_for_name(&name)),
            gpus: vec![GpuInfo {
                name,
                vram_mb,
                compute_capability: 0,
            }],
        };
    }
    detect_nvml_gpus()
        .or_else(detect_system_gpus)
        .unwrap_or_default()
}

fn detect_nvml_gpus() -> Option<GpuProbe> {
    let nvml = Nvml::init().ok()?;
    let count = nvml.device_count().ok()?;
    let mut gpus = Vec::new();
    for index in 0..count {
        let device = nvml.device_by_index(index).ok()?;
        let name = device.name().ok()?;
        let memory = device.memory_info().ok();
        gpus.push(GpuInfo {
            name,
            vram_mb: memory.map(|info| info.total / 1024 / 1024).unwrap_or(0),
            compute_capability: 0,
        });
    }
    (!gpus.is_empty()).then(|| GpuProbe {
        gpus,
        backend_type: Some("nvidia".into()),
    })
}

/// GPU discovery without a graphics stack: DRM sysfs on Linux, the built-in GPU on Apple
/// silicon. NVIDIA cards are handled by NVML before this runs.
fn detect_system_gpus() -> Option<GpuProbe> {
    #[cfg(target_os = "linux")]
    {
        detect_drm_gpus(std::path::Path::new("/sys/class/drm"))
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Some(GpuProbe {
            gpus: vec![GpuInfo {
                name: "Apple GPU".into(),
                vram_mb: 0,
                compute_capability: 0x106b,
            }],
            backend_type: Some("metal:apple".into()),
        })
    }
    #[cfg(not(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))))]
    {
        None
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn detect_drm_gpus(drm_root: &std::path::Path) -> Option<GpuProbe> {
    let mut cards: Vec<_> = std::fs::read_dir(drm_root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            // `card0`, not connectors such as `card0-eDP-1`.
            entry.file_name().to_str().is_some_and(|name| {
                name.strip_prefix("card").is_some_and(|rest| {
                    !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
                })
            })
        })
        .map(|entry| entry.path())
        .collect();
    cards.sort();

    let mut gpus = Vec::new();
    let mut backend_type = None;
    for card in cards {
        let device = card.join("device");
        let Some(vendor_id) = std::fs::read_to_string(device.join("vendor"))
            .ok()
            .and_then(|raw| u32::from_str_radix(raw.trim().trim_start_matches("0x"), 16).ok())
        else {
            continue;
        };
        let vendor = vendor_name(vendor_id);
        // amdgpu exposes dedicated VRAM here; integrated GPUs share system memory and report none.
        let vram_mb = std::fs::read_to_string(device.join("mem_info_vram_total"))
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .map_or(0, |bytes| bytes / 1024 / 1024);
        if backend_type.is_none() {
            backend_type = Some(format!("vulkan:{vendor}"));
        }
        gpus.push(GpuInfo {
            name: format!("{vendor} GPU"),
            vram_mb,
            compute_capability: vendor_id as u64,
        });
    }
    (!gpus.is_empty()).then_some(GpuProbe { gpus, backend_type })
}

fn preferred_backend_for_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    if lower.contains("nvidia") || lower.contains("rtx") || lower.contains("gtx") {
        "nvidia".into()
    } else if lower.contains("amd") || lower.contains("radeon") {
        "amd".into()
    } else if lower.contains("apple") {
        "metal:apple".into()
    } else if lower.contains("intel") {
        "intel".into()
    } else {
        "gpu".into()
    }
}

fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x10de => "nvidia",
        0x1002 | 0x1022 => "amd",
        0x8086 => "intel",
        0x106b => "apple",
        _ => "other",
    }
}

fn estimate_gpu_tflops(gpus: &[GpuInfo], vram_mb: u64) -> f64 {
    if gpus.is_empty() && vram_mb == 0 {
        return 0.0;
    }
    let name = gpus
        .first()
        .map(|gpu| gpu.name.to_ascii_lowercase())
        .unwrap_or_default();
    if name.contains("nvidia") || name.contains("rtx") {
        10.0
    } else if name.contains("amd") || name.contains("radeon") {
        5.0
    } else if name.contains("apple") || name.contains("metal") {
        6.0
    } else if vram_mb >= 8_192 {
        4.0
    } else {
        1.0
    }
}

fn memory_bandwidth_gbps(system: &System, ram_mb: u64, gpu_tflops: f64) -> f64 {
    let cpu_count = cpu_count(system);
    let avg_frequency_mhz = average_cpu_frequency_mhz(system);
    let base = if gpu_tflops > 0.0 { 80.0 } else { 20.0 };
    base + (cpu_count as f64 * 1.5)
        + ((ram_mb as f64 / 1024.0).sqrt() * 2.0)
        + (avg_frequency_mhz / 1000.0).min(5.0)
}

fn average_cpu_frequency_mhz(system: &System) -> f64 {
    let cpus = system.cpus();
    if cpus.is_empty() {
        return 0.0;
    }
    cpus.iter().map(|cpu| cpu.frequency() as f64).sum::<f64>() / cpus.len() as f64
}

fn detect_disk_bandwidth_mbps() -> f64 {
    std::env::var("BITTY_DISK_MBPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(match std::env::consts::OS {
            "macos" => 900.0,
            "linux" => 500.0,
            _ => 300.0,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drm_probe_reads_cards_and_skips_connectors() {
        let root = std::env::temp_dir().join(format!("bitty-drm-{}", std::process::id()));
        let card = root.join("card1/device");
        std::fs::create_dir_all(&card).unwrap();
        std::fs::create_dir_all(root.join("card1-eDP-1")).unwrap();
        std::fs::write(card.join("vendor"), "0x1002\n").unwrap();
        std::fs::write(card.join("mem_info_vram_total"), "8589934592\n").unwrap();

        let probe = detect_drm_gpus(&root).unwrap();
        std::fs::remove_dir_all(&root).unwrap();

        assert_eq!(probe.gpus.len(), 1);
        assert_eq!(probe.gpus[0].vram_mb, 8192);
        assert_eq!(probe.backend_type.as_deref(), Some("vulkan:amd"));
    }

    #[test]
    fn ram_override_populates_profile() {
        std::env::set_var("BITTY_RAM_MB", "8192");
        let profile = HardwareProfiler::new("test-node").profile();
        std::env::remove_var("BITTY_RAM_MB");

        assert_eq!(profile.ram_mb, 8192);
        assert_eq!(profile.memory_gb, 8.0);
    }
}
