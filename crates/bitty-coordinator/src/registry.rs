use bitty_protocol::{HardwareProfile, Heartbeat, NodeId};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub enum NodeHealth {
    Healthy,
    Degraded,
    MissingHeartbeat,
}

#[derive(Clone, Debug)]
pub struct RegisteredNode {
    pub profile: HardwareProfile,
    pub last_heartbeat: Instant,
    pub baseline_tokens_per_second: f64,
    pub observed_tokens_per_second: f64,
    pub avg_forward_latency_ms: f64,
    pub activation_bytes_per_second: u64,
    pub backend_type: String,
    pub health: NodeHealth,
}

#[derive(Clone, Debug, Default)]
pub struct Registry {
    nodes: HashMap<NodeId, RegisteredNode>,
}

impl Registry {
    pub fn register(&mut self, profile: HardwareProfile) {
        let baseline = profile.effective_compute_score().max(0.1);
        let backend_type = node_backend(&profile);
        self.nodes.insert(
            profile.node_id.clone(),
            RegisteredNode {
                profile,
                last_heartbeat: Instant::now(),
                baseline_tokens_per_second: baseline,
                observed_tokens_per_second: baseline,
                avg_forward_latency_ms: 0.0,
                activation_bytes_per_second: 0,
                backend_type,
                health: NodeHealth::Healthy,
            },
        );
    }

    pub fn heartbeat(&mut self, heartbeat: Heartbeat) -> bool {
        let Some(node) = self.nodes.get_mut(&heartbeat.node_id) else {
            return false;
        };

        node.last_heartbeat = Instant::now();
        node.observed_tokens_per_second = heartbeat.observed_tokens_per_second;
        node.avg_forward_latency_ms = heartbeat.avg_forward_latency_ms;
        node.activation_bytes_per_second = heartbeat.activation_bytes_per_second;
        if !heartbeat.backend_type.is_empty() {
            node.backend_type = heartbeat.backend_type.clone();
            node.profile.backend_type = heartbeat.backend_type;
        }
        node.health =
            if heartbeat.observed_tokens_per_second < node.baseline_tokens_per_second * 0.5 {
                NodeHealth::Degraded
            } else {
                NodeHealth::Healthy
            };
        true
    }

    /// A node refused its layers and said what it has free now: the next plan counts with it.
    pub fn note_free(&mut self, node_id: &NodeId, memory_mb: Option<u64>, disk_mb: Option<u64>) {
        let Some(node) = self.nodes.get_mut(node_id) else {
            return;
        };
        if let Some(memory_mb) = memory_mb {
            node.profile.available_ram_mb = memory_mb.clamp(1, node.profile.ram_mb.max(1));
        }
        if let Some(disk_mb) = disk_mb {
            node.profile.free_disk_mb = disk_mb.max(1);
        }
    }

    pub fn profiles(&self) -> Vec<HardwareProfile> {
        self.nodes
            .values()
            .map(|node| {
                let mut profile = node.profile.clone();
                let ratio = (node.observed_tokens_per_second / node.baseline_tokens_per_second)
                    .clamp(0.10, 2.0);
                if node.observed_tokens_per_second > 0.0 {
                    profile.cpu_tflops *= ratio;
                    profile.gpu_tflops *= ratio;
                }
                if node.health == NodeHealth::Degraded && profile.gpu_tflops == 0.0 {
                    profile.layer_eligible = false;
                }
                profile
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn profile(&self, node_id: &NodeId) -> Option<HardwareProfile> {
        self.nodes.get(node_id).map(|node| node.profile.clone())
    }

    pub fn evict_missing(&mut self, timeout: Duration) -> Vec<NodeId> {
        let now = Instant::now();
        let missing = self
            .nodes
            .iter()
            .filter(|(_, node)| now.duration_since(node.last_heartbeat) > timeout)
            .map(|(node_id, _)| node_id.clone())
            .collect::<Vec<_>>();

        for node_id in &missing {
            self.nodes.remove(node_id);
        }

        missing
    }
}

fn node_backend(profile: &HardwareProfile) -> String {
    profile.backend_type().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_protocol::NodeTier;

    fn profile(ram_mb: u64, available_ram_mb: u64) -> HardwareProfile {
        HardwareProfile {
            node_id: NodeId::new("node-a"),
            cpu_tflops: 1.0,
            gpu_tflops: 0.0,
            memory_gb: ram_mb as f64 / 1024.0,
            memory_bandwidth_gbps: 20.0,
            disk_bandwidth_mbps: 500.0,
            network_rtt_ms: 10.0,
            uplink_mbps: 100.0,
            os: "linux".into(),
            tier: NodeTier::D,
            ram_mb,
            available_ram_mb,
            free_disk_mb: 0,
            build: String::new(),
            vram_mb: 0,
            architecture: "x86_64".into(),
            gpus: Vec::new(),
            os_reclaim_score: 0.0,
            worker_endpoint: String::new(),
            model_path: String::new(),
            backend_type: "cpu".into(),
            layer_eligible: true,
            max_layers: 0,
        }
    }

    #[test]
    fn a_refusal_changes_what_the_node_is_planned_with() {
        let mut registry = Registry::default();
        registry.register(profile(8192, 6000));
        let id = NodeId::new("node-a");

        registry.note_free(&id, Some(2700), None);
        let noted = registry.profile(&id).unwrap();
        assert_eq!(noted.available_ram_mb, 2700);
        assert_eq!(noted.free_disk_mb, 0);

        registry.note_free(&id, None, Some(4000));
        let noted = registry.profile(&id).unwrap();
        assert_eq!(noted.available_ram_mb, 2700);
        assert_eq!(noted.free_disk_mb, 4000);

        // Never more than the machine has, and never the zero that means "did not say".
        registry.note_free(&id, Some(99_000), Some(0));
        let noted = registry.profile(&id).unwrap();
        assert_eq!(noted.available_ram_mb, 8192);
        assert_eq!(noted.free_disk_mb, 1);

        registry.note_free(&NodeId::new("unknown"), Some(1), Some(1));
    }
}
