use bitty_protocol::{HardwareProfile, NodeId, NodeTier};

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
        let cpu_tflops = cpu_count as f64 * 0.05;
        let tier = if cpu_count >= 16 {
            NodeTier::B
        } else if cpu_count >= 8 {
            NodeTier::C
        } else {
            NodeTier::D
        };

        HardwareProfile {
            node_id: self.node_id.clone(),
            cpu_tflops,
            gpu_tflops: 0.0,
            memory_gb: 4.0,
            memory_bandwidth_gbps: 20.0,
            disk_bandwidth_mbps: 300.0,
            network_rtt_ms: 25.0,
            uplink_mbps: 50.0,
            os: std::env::consts::OS.into(),
            tier,
        }
    }
}
