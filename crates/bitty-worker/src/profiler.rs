use bitty_protocol::{HardwareProfile, NodeId, NodeTier};
use std::time::Instant;

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
        let cpu_count = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let cpu_tflops = estimate_cpu_tflops(cpu_count);
        let tier = if cpu_count >= 16 {
            NodeTier::B
        } else if cpu_count >= 8 {
            NodeTier::C
        } else {
            NodeTier::D
        };
        let ram_mb = std::env::var("BITTY_RAM_MB")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4096);

        HardwareProfile {
            node_id: self.node_id.clone(),
            cpu_tflops,
            gpu_tflops: 0.0,
            memory_gb: ram_mb as f64 / 1024.0,
            memory_bandwidth_gbps: 20.0,
            disk_bandwidth_mbps: 300.0,
            network_rtt_ms: 25.0,
            uplink_mbps: 50.0,
            os: std::env::consts::OS.into(),
            tier,
            ram_mb,
            vram_mb: 0,
            architecture: std::env::consts::ARCH.into(),
            gpus: Vec::new(),
            os_reclaim_score: 0.0,
            worker_endpoint: std::env::var("BITTY_WORKER_ENDPOINT").unwrap_or_default(),
        }
    }
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
