use bitty_protocol::{
    AssignedLayerRange, HardwareProfile, LayerAssignment, LayerMetadata, ModelStage, NodeId,
    NodeTier, Quantization,
};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub memory_reserve_fraction: f64,
    pub critical_quantization: Quantization,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            memory_reserve_fraction: 0.15,
            critical_quantization: Quantization::Fp16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Halda {
    config: SchedulerConfig,
}

impl Halda {
    pub fn new(config: SchedulerConfig) -> Self {
        Self { config }
    }

    pub fn assign(
        &self,
        nodes: &[HardwareProfile],
        layers: &[LayerMetadata],
    ) -> Result<Vec<LayerAssignment>, HaldaError> {
        if nodes.is_empty() {
            return Err(HaldaError::NoNodes);
        }
        if layers.is_empty() {
            return Ok(Vec::new());
        }

        let mut ranked = nodes.to_vec();
        ranked.sort_by(|a, b| {
            b.effective_compute_score()
                .partial_cmp(&a.effective_compute_score())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node_id.0.cmp(&b.node_id.0))
        });

        let total_score = ranked
            .iter()
            .map(HardwareProfile::effective_compute_score)
            .sum::<f64>()
            .max(1.0);

        let total_layers = layers.len() as u32;
        let mut assignments = Vec::new();
        let mut cursor = 0_u32;

        for (index, node) in ranked.iter().enumerate() {
            if cursor >= total_layers {
                break;
            }

            let remaining_nodes = ranked.len() - index;
            let remaining_layers = total_layers - cursor;
            let score_share = node.effective_compute_score() / total_score;
            let target_len = ((layers.len() as f64) * score_share).round() as u32;
            let mut len = target_len.max(1).min(remaining_layers);

            if remaining_layers > remaining_nodes as u32 {
                len = len.min(remaining_layers.saturating_sub((remaining_nodes - 1) as u32));
            }

            let quantization = quantization_for(node, cursor, cursor + len, layers, &self.config);
            let memory_budget = available_memory_bytes(node, self.config.memory_reserve_fraction);

            while len > 1
                && assigned_weight_bytes_for_len(cursor, len, layers, quantization) > memory_budget
            {
                len -= 1;
            }

            let assigned_weight_bytes =
                assignment_weight_bytes(cursor, cursor + len, layers, quantization);
            if assigned_weight_bytes > memory_budget {
                return Err(HaldaError::InsufficientMemory {
                    node_id: node.node_id.clone(),
                    required: assigned_weight_bytes,
                    available: memory_budget,
                });
            }

            assignments.push(LayerAssignment {
                node_id: node.node_id.clone(),
                range: AssignedLayerRange {
                    start_layer: cursor,
                    end_layer_exclusive: cursor + len,
                    quantization,
                },
                assigned_weight_bytes,
                expected_latency_ms: estimate_latency_ms(node, cursor, cursor + len, layers),
                next_node_id: None,
                disk_offload_fraction: disk_offload_fraction(node, assigned_weight_bytes),
                model_stage: ModelStage::LayerRange,
            });
            cursor += len;
        }

        if cursor < total_layers {
            self.append_remaining_to_fastest(&mut assignments, &ranked[0], cursor, layers)?;
        }

        self.link_ring(&mut assignments);
        self.validate(&assignments, nodes, layers)?;
        Ok(assignments)
    }

    fn append_remaining_to_fastest(
        &self,
        assignments: &mut Vec<LayerAssignment>,
        node: &HardwareProfile,
        cursor: u32,
        layers: &[LayerMetadata],
    ) -> Result<(), HaldaError> {
        let end = layers.len() as u32;
        let quantization = quantization_for(node, cursor, end, layers, &self.config);
        let assigned_weight_bytes = assignment_weight_bytes(cursor, end, layers, quantization);
        let memory_budget = available_memory_bytes(node, self.config.memory_reserve_fraction);

        if assigned_weight_bytes > memory_budget {
            return Err(HaldaError::InsufficientMemory {
                node_id: node.node_id.clone(),
                required: assigned_weight_bytes,
                available: memory_budget,
            });
        }

        assignments.push(LayerAssignment {
            node_id: node.node_id.clone(),
            range: AssignedLayerRange {
                start_layer: cursor,
                end_layer_exclusive: end,
                quantization,
            },
            assigned_weight_bytes,
            expected_latency_ms: estimate_latency_ms(node, cursor, end, layers),
            next_node_id: None,
            disk_offload_fraction: disk_offload_fraction(node, assigned_weight_bytes),
            model_stage: ModelStage::LayerRange,
        });
        Ok(())
    }

    fn link_ring(&self, assignments: &mut [LayerAssignment]) {
        if assignments.is_empty() {
            return;
        }

        let node_ids: Vec<NodeId> = assignments.iter().map(|a| a.node_id.clone()).collect();
        let len = assignments.len();
        for (index, assignment) in assignments.iter_mut().enumerate() {
            assignment.next_node_id = Some(node_ids[(index + 1) % len].clone());
        }
        if let Some(first) = assignments.first_mut() {
            first.model_stage = ModelStage::EmbeddingAndLayers;
        }
        if let Some(last) = assignments.last_mut() {
            last.model_stage = ModelStage::FinalLayersAndLmHead;
        }
    }

    pub fn validate(
        &self,
        assignments: &[LayerAssignment],
        nodes: &[HardwareProfile],
        layers: &[LayerMetadata],
    ) -> Result<(), HaldaError> {
        let mut covered = HashSet::new();
        for assignment in assignments {
            if assignment.range.is_empty() {
                return Err(HaldaError::InvalidAssignment("empty layer range".into()));
            }

            let node = nodes
                .iter()
                .find(|node| node.node_id == assignment.node_id)
                .ok_or_else(|| HaldaError::InvalidAssignment("unknown node".into()))?;

            let memory_budget = available_memory_bytes(node, self.config.memory_reserve_fraction);
            if assignment.assigned_weight_bytes > memory_budget {
                return Err(HaldaError::InsufficientMemory {
                    node_id: node.node_id.clone(),
                    required: assignment.assigned_weight_bytes,
                    available: memory_budget,
                });
            }

            for layer_id in assignment.range.start_layer..assignment.range.end_layer_exclusive {
                if !covered.insert(layer_id) {
                    return Err(HaldaError::InvalidAssignment(format!(
                        "layer {layer_id} assigned twice"
                    )));
                }
            }
        }

        for layer in layers {
            if !covered.contains(&layer.layer_id) {
                return Err(HaldaError::InvalidAssignment(format!(
                    "layer {} was not assigned",
                    layer.layer_id
                )));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum HaldaError {
    #[error("cannot assign layers without nodes")]
    NoNodes,
    #[error("node {node_id} needs {required} bytes but only {available} bytes are available")]
    InsufficientMemory {
        node_id: NodeId,
        required: u64,
        available: u64,
    },
    #[error("invalid assignment: {0}")]
    InvalidAssignment(String),
}

fn quantization_for(
    node: &HardwareProfile,
    start: u32,
    end: u32,
    layers: &[LayerMetadata],
    config: &SchedulerConfig,
) -> Quantization {
    if layers[start as usize..end as usize]
        .iter()
        .any(|layer| layer.precision_critical)
    {
        return config.critical_quantization;
    }

    match node.tier {
        NodeTier::S | NodeTier::A => Quantization::Q4,
        NodeTier::B => Quantization::Q3,
        NodeTier::C | NodeTier::D => Quantization::Q2,
    }
}

fn assignment_weight_bytes(
    start: u32,
    end: u32,
    layers: &[LayerMetadata],
    quantization: Quantization,
) -> u64 {
    layers[start as usize..end as usize]
        .iter()
        .map(|layer| (layer.weight_bytes as f64 * quantization.bytes_per_weight()).ceil() as u64)
        .sum()
}

fn assigned_weight_bytes_for_len(
    start: u32,
    len: u32,
    layers: &[LayerMetadata],
    quantization: Quantization,
) -> u64 {
    assignment_weight_bytes(start, start + len, layers, quantization)
}

fn available_memory_bytes(node: &HardwareProfile, reserve_fraction: f64) -> u64 {
    (node.memory_bytes() as f64 * (1.0 - reserve_fraction)).max(0.0) as u64
}

fn estimate_latency_ms(
    node: &HardwareProfile,
    start: u32,
    end: u32,
    layers: &[LayerMetadata],
) -> f64 {
    let flops = layers[start as usize..end as usize]
        .iter()
        .map(|layer| layer.estimated_flops)
        .sum::<f64>();
    let tflops = node.gpu_tflops.max(node.cpu_tflops * 0.35).max(0.001);
    (flops / (tflops * 1e12)) * 1000.0 + node.network_rtt_ms
}

fn disk_offload_fraction(node: &HardwareProfile, assigned_weight_bytes: u64) -> f32 {
    let budget = available_memory_bytes(node, 0.15);
    if assigned_weight_bytes <= budget {
        0.0
    } else {
        1.0 - (budget as f32 / assigned_weight_bytes as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn node(index: usize, tier: NodeTier, memory_gb: f64) -> HardwareProfile {
        HardwareProfile {
            node_id: NodeId::new(format!("node-{index}")),
            cpu_tflops: 0.5 + index as f64,
            gpu_tflops: match tier {
                NodeTier::S => 30.0,
                NodeTier::A => 15.0,
                NodeTier::B => 5.0,
                NodeTier::C => 1.0,
                NodeTier::D => 0.0,
            },
            memory_gb,
            memory_bandwidth_gbps: 20.0 + index as f64,
            disk_bandwidth_mbps: 500.0,
            network_rtt_ms: 10.0,
            uplink_mbps: 100.0,
            os: "linux".into(),
            tier,
            ram_mb: (memory_gb * 1024.0) as u64,
            vram_mb: 0,
            architecture: "x86_64".into(),
            gpus: Vec::new(),
            os_reclaim_score: 0.0,
            worker_endpoint: String::new(),
            model_path: String::new(),
        }
    }

    fn layers(count: u32) -> Vec<LayerMetadata> {
        (0..count)
            .map(|layer_id| LayerMetadata {
                layer_id,
                weight_bytes: 1_000_000,
                activation_bytes: 4096,
                estimated_flops: 1e9,
                precision_critical: layer_id == 0 || layer_id + 1 == count,
            })
            .collect()
    }

    proptest! {
        #[test]
        fn halda_covers_each_layer_once(node_count in 1_usize..20, layer_count in 1_u32..80) {
            let tiers = [NodeTier::S, NodeTier::A, NodeTier::B, NodeTier::C, NodeTier::D];
            let nodes = (0..node_count)
                .map(|index| node(index, tiers[index % tiers.len()], 2.0))
                .collect::<Vec<_>>();
            let layers = layers(layer_count);

            let assignments = Halda::new(SchedulerConfig::default())
                .assign(&nodes, &layers)
                .unwrap();

            Halda::new(SchedulerConfig::default())
                .validate(&assignments, &nodes, &layers)
                .unwrap();
        }
    }

    #[test]
    fn critical_layers_remain_high_precision() {
        let nodes = vec![node(0, NodeTier::S, 2.0), node(1, NodeTier::D, 2.0)];
        let layers = layers(4);

        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &layers)
            .unwrap();

        assert!(assignments
            .iter()
            .filter(|assignment| assignment.range.contains(0) || assignment.range.contains(3))
            .all(|assignment| assignment.range.quantization == Quantization::Fp16));
    }
}
