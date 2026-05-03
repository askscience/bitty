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
