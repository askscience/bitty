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

        let mut ranked = nodes
            .iter()
            .filter(|node| node.layer_eligible && node.max_layers != 0)
            .cloned()
            .collect::<Vec<_>>();
        if ranked.is_empty() {
            ranked = nodes.to_vec();
        }
        ranked.sort_by(|a, b| {
            b.effective_compute_score()
                .partial_cmp(&a.effective_compute_score())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node_id.0.cmp(&b.node_id.0))
        });
        if ranked.len() > 1 {
            let strongest = ranked[0].effective_compute_score().max(0.001);
            ranked.retain(|node| {
                node.gpu_tflops > 0.0 || node.effective_compute_score() >= strongest * 0.15
            });
        }
        if ranked.is_empty() {
            ranked = nodes.to_vec();
        }

        let total_score = ranked
            .iter()
            .map(HardwareProfile::effective_compute_score)
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);

        let total_layers = layers.len() as u32;
        let fits = |node: &HardwareProfile, start: u32, len: u32| {
            let quantization = quantization_for(node, start, start + len, layers, &self.config);
            assigned_weight_bytes_for_len(start, len, layers, quantization)
                <= available_memory_bytes(node, self.config.memory_reserve_fraction)
        };

        // First pass: a share of the layers in proportion to each node's speed.
        let mut lens: Vec<u32> = Vec::new();
        let mut cursor = 0_u32;
        for (index, node) in ranked.iter().enumerate() {
            if cursor >= total_layers {
                break;
            }
            let remaining_nodes = ranked.len() - index;
            let remaining_layers = total_layers - cursor;
            let score_share = node.effective_compute_score() / total_score;
            let target_len = ((layers.len() as f64) * score_share).round() as u32;
            let mut len = target_len
                .max(1)
                .min(node.max_layers.max(1))
                .min(remaining_layers);
            if remaining_layers > remaining_nodes as u32 {
                len = len.min(remaining_layers.saturating_sub((remaining_nodes - 1) as u32));
            }
            while len > 1 && !fits(node, cursor, len) {
                len -= 1;
            }
            lens.push(len);
            cursor += len;
        }

        // Rounding, layer caps and memory limits can leave layers over. A node runs one stage
        // engine, so it can hold only one range: the layers left over make the existing ranges
        // longer, fastest node first, and are never handed out as a second range.
        let mut left_over = total_layers - cursor;
        for index in 0..lens.len() {
            if left_over == 0 {
                break;
            }
            let node = &ranked[index];
            let start: u32 = lens[..index].iter().sum();
            while left_over > 0
                && lens[index] < node.max_layers.max(1)
                && fits(node, start, lens[index] + 1)
            {
                lens[index] += 1;
                left_over -= 1;
            }
        }
        if left_over > 0 {
            return Err(HaldaError::NotEnoughCapacity {
                layers: total_layers,
                unplaced: left_over,
                nodes: ranked.len(),
            });
        }

        let mut assignments = Vec::new();
        let mut cursor = 0_u32;
        for (node, len) in ranked.iter().zip(lens) {
            let end = cursor + len;
            let quantization = quantization_for(node, cursor, end, layers, &self.config);
            let assigned_weight_bytes = assignment_weight_bytes(cursor, end, layers, quantization);
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
            cursor = end;
        }

        self.link_ring(&mut assignments);
        self.validate(&assignments, nodes, layers)?;
        Ok(assignments)
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
        let mut holders = HashSet::new();
        for assignment in assignments {
            if !holders.insert(&assignment.node_id) {
                return Err(HaldaError::InvalidAssignment(format!(
                    "node {} was given two layer ranges",
                    assignment.node_id
                )));
            }
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
    #[error("{unplaced} of {layers} layers do not fit on the {nodes} available nodes")]
    NotEnoughCapacity {
        layers: u32,
        unplaced: u32,
        nodes: usize,
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
    let memory_bytes = if node.gpu_tflops > 0.0 && node.vram_mb > 0 {
        node.vram_mb.saturating_mul(1024 * 1024)
    } else {
        node.memory_bytes()
    };
    (memory_bytes as f64 * (1.0 - reserve_fraction)).max(0.0) as u64
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
            backend_type: if tier == NodeTier::D {
                "cpu".into()
            } else {
                "gpu".into()
            },
            layer_eligible: true,
            max_layers: u32::MAX,
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
    fn a_slow_node_alone_gets_every_layer_as_one_range() {
        let mut slow = node(0, NodeTier::D, 16.0);
        slow.cpu_tflops = 0.05;
        slow.gpu_tflops = 0.0;
        slow.vram_mb = 0;
        slow.backend_type = "cpu".into();
        assert!(slow.effective_compute_score() < 1.0);

        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&[slow], &layers(24))
            .unwrap();

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].range.len(), 24);
    }

    #[test]
    fn layers_that_fit_nowhere_are_an_error_not_a_second_range() {
        let mut capped = node(0, NodeTier::A, 16.0);
        capped.max_layers = 4;
        let result = Halda::new(SchedulerConfig::default()).assign(&[capped], &layers(24));
        assert!(matches!(
            result,
            Err(HaldaError::NotEnoughCapacity { unplaced: 20, .. })
        ));
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

    #[test]
    fn halda_skips_weak_cpu_node_when_stronger_node_is_available() {
        let mut weak = node(0, NodeTier::D, 8.0);
        weak.cpu_tflops = 0.25;
        weak.gpu_tflops = 0.0;
        weak.backend_type = "cpu".into();

        let mut strong = node(1, NodeTier::A, 32.0);
        strong.gpu_tflops = 12.0;
        strong.vram_mb = 16_384;
        strong.backend_type = "gpu".into();

        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&[weak.clone(), strong.clone()], &layers(30))
            .unwrap();

        assert!(assignments
            .iter()
            .all(|assignment| assignment.node_id != weak.node_id));
        assert_eq!(
            assignments
                .iter()
                .map(|assignment| assignment.range.len())
                .sum::<u32>(),
            30
        );
    }
}
