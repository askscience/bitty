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
    pub health: NodeHealth,
}

#[derive(Clone, Debug, Default)]
pub struct Registry {
    nodes: HashMap<NodeId, RegisteredNode>,
}

impl Registry {
    pub fn register(&mut self, profile: HardwareProfile) {
        let baseline = profile.effective_compute_score().max(0.1);
        self.nodes.insert(
            profile.node_id.clone(),
            RegisteredNode {
                profile,
                last_heartbeat: Instant::now(),
                baseline_tokens_per_second: baseline,
                observed_tokens_per_second: baseline,
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
            .map(|node| node.profile.clone())
            .collect()
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
