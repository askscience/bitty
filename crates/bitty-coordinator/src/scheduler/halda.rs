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
    /// Memory a stage engine needs on top of its layers: the embeddings or the output head, the
    /// context of every session, the engine itself.
    pub stage_overhead_bytes: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            memory_reserve_fraction: bitty_protocol::MEMORY_RESERVE_FRACTION,
            critical_quantization: Quantization::Fp16,
            stage_overhead_bytes: bitty_protocol::stage_overhead_bytes(),
        }
    }
}

/// A node that answers the leader faster than this is near: it is beside it, or a short
/// direct link away. A node that has not measured its link yet counts as near.
const NEAR_MS: f64 = 30.0;
/// Nodes that are far come in by how far they are, in steps of this size; inside a step
/// the stronger one comes first.
const FAR_BAND_MS: f64 = 20.0;

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
        let all_nodes = all_nodes_wanted();
        if ranked.len() > 1 && !all_nodes {
            let strongest = ranked[0].effective_compute_score().max(0.001);
            let strong = ranked
                .iter()
                .filter(|node| {
                    node.gpu_tflops > 0.0 || node.effective_compute_score() >= strongest * 0.15
                })
                .cloned()
                .collect::<Vec<_>>();
            // A slow node is left out only while the others can hold the model without it:
            // a slow ring is better than none.
            if self.lay_out(&strong, layers).is_ok() {
                ranked = strong;
            }
        }
        if ranked.is_empty() {
            ranked = nodes.to_vec();
        }

        // A node that cannot hold even one layer in the memory it has free would start to swap
        // and stall the whole ring: it stays out while other nodes can do the work.
        if ranked.len() > 1 {
            let largest = layers
                .iter()
                .map(|layer| layer.weight_bytes + layer.state_bytes)
                .max()
                .unwrap_or(0);
            let roomy = ranked
                .iter()
                .filter(|node| {
                    largest + self.config.stage_overhead_bytes
                        <= node.free_memory_bytes(self.config.memory_reserve_fraction)
                        && node.disk_bytes().is_none_or(|disk| largest <= disk)
                })
                .cloned()
                .collect::<Vec<_>>();
            if !roomy.is_empty() {
                ranked = roomy;
            }
        }

        // The token goes through the nodes one after the other: a node more does not make
        // the layers faster, it adds a trip. So the nodes that are near do the work, and
        // one that is far comes in only while the nearer ones cannot hold the model in the
        // memory they have free.
        let mut far = ranked
            .iter()
            .filter(|node| node.network_rtt_ms >= NEAR_MS)
            .cloned()
            .collect::<Vec<_>>();
        far.sort_by(|a, b| {
            let band = |node: &HardwareProfile| (node.network_rtt_ms / FAR_BAND_MS) as u64;
            band(a).cmp(&band(b))
        });
        if all_nodes {
            far.clear();
        }
        let mut lens = None;
        for count in 0..far.len() {
            let allowed = &far[..count];
            let near = ranked
                .iter()
                .filter(|node| {
                    node.network_rtt_ms < NEAR_MS
                        || allowed.iter().any(|far| far.node_id == node.node_id)
                })
                .cloned()
                .collect::<Vec<_>>();
            if near.is_empty() {
                continue;
            }
            if let Ok((found, false)) = self.lay_out(&near, layers) {
                lens = Some(found);
                ranked = near;
                break;
            }
        }
        let lens = match lens {
            Some(lens) => lens,
            None => self.lay_out(&ranked, layers)?.0,
        };

        let mut cursor = 0_u32;
        let ranges = ranked
            .iter()
            .zip(lens)
            .map(|(node, len)| {
                let start = cursor;
                cursor += len;
                (node, start, cursor)
            })
            .collect::<Vec<_>>();
        self.build(&ranges, nodes, layers)
    }

    /// How many layers each node takes, in the order of `ranked`, and whether a node had to
    /// go past the memory it has free.
    fn lay_out(
        &self,
        ranked: &[HardwareProfile],
        layers: &[LayerMetadata],
    ) -> Result<(Vec<u32>, bool), HaldaError> {
        let total_score = ranked
            .iter()
            .map(HardwareProfile::effective_compute_score)
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);

        let total_layers = layers.len() as u32;
        // Two limits. The free limit is what the node had free when it started: inside it the
        // node keeps working as it did before. The hard limit is its whole memory less a
        // reserve: past it the node swaps. Layers go inside the free limit first.
        let weights = |start: u32, len: u32| assignment_weight_bytes(start, start + len, layers);
        let memory = |start: u32, len: u32| assignment_memory_bytes(start, start + len, layers);
        let on_disk = |node: &HardwareProfile, start: u32, len: u32| {
            node.disk_bytes()
                .is_none_or(|disk| weights(start, len) <= disk)
        };
        let fits = |node: &HardwareProfile, start: u32, len: u32| {
            on_disk(node, start, len)
                && memory(start, len) + self.config.stage_overhead_bytes
                    <= node.free_memory_bytes(self.config.memory_reserve_fraction)
        };
        let fits_hard = |node: &HardwareProfile, start: u32, len: u32| {
            on_disk(node, start, len)
                && memory(start, len) + self.config.stage_overhead_bytes
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
        let mut past_free = false;
        for hard in [false, true] {
            for index in 0..lens.len() {
                if left_over == 0 {
                    break;
                }
                let node = &ranked[index];
                let start: u32 = lens[..index].iter().sum();
                while left_over > 0
                    && lens[index] < node.max_layers.max(1)
                    && if hard {
                        fits_hard(node, start, lens[index] + 1)
                    } else {
                        fits(node, start, lens[index] + 1)
                    }
                {
                    lens[index] += 1;
                    left_over -= 1;
                    past_free |= hard;
                }
            }
        }
        if left_over > 0 {
            return Err(HaldaError::NotEnoughCapacity {
                layers: total_layers,
                unplaced: left_over,
                nodes: ranked.len(),
            });
        }

        Ok((lens, past_free))
    }

    /// A plan made earlier, as it stands on the nodes of now: the same ranges on the same
    /// nodes, or an error when a node is gone or a range no longer fits where it was. A plan
    /// that still holds is worth keeping: every range that moves is a download of gigabytes.
    pub fn keep(
        &self,
        ranges: &[(NodeId, u32, u32)],
        nodes: &[HardwareProfile],
        layers: &[LayerMetadata],
    ) -> Result<Vec<LayerAssignment>, HaldaError> {
        let mut placed = Vec::new();
        for (node_id, start, end) in ranges {
            let node = nodes
                .iter()
                .find(|node| &node.node_id == node_id)
                .ok_or_else(|| HaldaError::InvalidAssignment(format!("node {node_id} is gone")))?;
            let len = end.saturating_sub(*start);
            if !node.layer_eligible || len > node.max_layers {
                return Err(HaldaError::InvalidAssignment(format!(
                    "node {node_id} no longer takes {len} layers"
                )));
            }
            let weights = assignment_weight_bytes(*start, *end, layers);
            if node.disk_bytes().is_some_and(|disk| weights > disk) {
                return Err(HaldaError::InvalidAssignment(format!(
                    "the disk of node {node_id} no longer holds its layers"
                )));
            }
            placed.push((node, *start, *end));
        }
        placed.sort_by_key(|(_, start, _)| *start);
        self.build(&placed, nodes, layers)
    }

    fn build(
        &self,
        ranges: &[(&HardwareProfile, u32, u32)],
        nodes: &[HardwareProfile],
        layers: &[LayerMetadata],
    ) -> Result<Vec<LayerAssignment>, HaldaError> {
        let mut assignments = Vec::new();
        for &(node, cursor, end) in ranges {
            let quantization = quantization_for(node, cursor, end, layers, &self.config);
            let assigned_weight_bytes = assignment_weight_bytes(cursor, end, layers);
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
                disk_offload_fraction: disk_offload_fraction(
                    node,
                    assignment_memory_bytes(cursor, end, layers) + self.config.stage_overhead_bytes,
                ),
                model_stage: ModelStage::LayerRange,
            });
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
            let required = assignment.assigned_weight_bytes + self.config.stage_overhead_bytes;
            if required > memory_budget {
                return Err(HaldaError::InsufficientMemory {
                    node_id: node.node_id.clone(),
                    required,
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

/// The bytes of the layers as they are in the model file: a stage file holds these very bytes,
/// whatever the tier of the node that loads it.
fn assignment_weight_bytes(start: u32, end: u32, layers: &[LayerMetadata]) -> u64 {
    layers[start as usize..end as usize]
        .iter()
        .map(|layer| layer.weight_bytes)
        .sum()
}

/// The memory a range takes on a node, less the fixed overhead of a stage: its weights and
/// the state its recurrent layers keep.
fn assignment_memory_bytes(start: u32, end: u32, layers: &[LayerMetadata]) -> u64 {
    layers[start as usize..end as usize]
        .iter()
        .map(|layer| layer.weight_bytes + layer.state_bytes)
        .sum()
}

fn available_memory_bytes(node: &HardwareProfile, reserve_fraction: f64) -> u64 {
    node.hard_memory_bytes(reserve_fraction)
}

/// `BITTY_ALL_NODES=1` on the leader gives layers to every machine that can hold one, near or
/// far, fast or slow. It is for tests: a far or slow machine usually makes the ring slower.
fn all_nodes_wanted() -> bool {
    std::env::var("BITTY_ALL_NODES")
        .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
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
            available_ram_mb: 0,
            free_disk_mb: 0,
            build: String::new(),
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
                state_bytes: 0,
                layer_id,
                weight_bytes: 1_000_000,
                activation_bytes: 4096,
                estimated_flops: 1e9,
                precision_critical: layer_id == 0 || layer_id + 1 == count,
            })
            .collect()
    }

    /// Forty layers of the size they have in the 35B model file.
    fn big_layers() -> Vec<LayerMetadata> {
        layers(40)
            .into_iter()
            .map(|mut layer| {
                layer.weight_bytes = 527 * 1024 * 1024;
                layer
            })
            .collect()
    }

    fn cpu_node(index: usize, ram_gb: u64, free_gb: u64) -> HardwareProfile {
        let mut profile = node(index, NodeTier::D, ram_gb as f64);
        profile.available_ram_mb = free_gb * 1024;
        profile
    }

    fn linked(mut profile: HardwareProfile, rtt_ms: f64) -> HardwareProfile {
        profile.network_rtt_ms = rtt_ms;
        profile
    }

    fn holders(assignments: &[LayerAssignment]) -> Vec<String> {
        let mut names = assignments
            .iter()
            .map(|assignment| assignment.node_id.0.clone())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    #[test]
    fn a_far_node_stays_out_while_the_near_ones_hold_the_model() {
        let nodes = vec![
            linked(cpu_node(0, 32, 28), 0.0),
            linked(cpu_node(1, 8, 6), 1.5),
            linked(cpu_node(2, 16, 14), 70.0),
        ];
        let plan = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        assert_eq!(holders(&plan), ["node-0", "node-1"]);
    }

    #[test]
    fn a_slow_node_is_kept_when_the_model_needs_its_memory() {
        // the leader can hold 14 layers, the fast far node about 20: the slow ones must help
        let mut leader = linked(cpu_node(0, 32, 28), 0.0);
        leader.max_layers = 14;
        leader.cpu_tflops = 4.0;
        let mut slow = linked(cpu_node(1, 8, 6), 1.5);
        slow.cpu_tflops = 0.1;
        let nodes = vec![leader, slow, linked(cpu_node(2, 16, 14), 70.0)];
        let plan = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        assert_eq!(holders(&plan), ["node-0", "node-1", "node-2"]);
    }

    #[test]
    fn the_nearest_far_node_comes_in_when_the_near_ones_are_too_small() {
        let nodes = vec![
            linked(cpu_node(0, 16, 12), 0.0),
            linked(cpu_node(1, 8, 6), 1.5),
            linked(cpu_node(2, 32, 28), 180.0),
            linked(cpu_node(3, 32, 28), 70.0),
        ];
        let plan = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        assert_eq!(holders(&plan), ["node-0", "node-1", "node-3"]);
    }

    #[test]
    fn every_node_works_when_only_all_of_them_hold_the_model() {
        let nodes = vec![
            linked(cpu_node(0, 16, 10), 0.0),
            linked(cpu_node(1, 16, 10), 90.0),
            linked(cpu_node(2, 16, 10), 200.0),
        ];
        let plan = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        assert_eq!(holders(&plan), ["node-0", "node-1", "node-2"]);
    }

    fn len_of(assignments: &[LayerAssignment], index: usize) -> u32 {
        assignments
            .iter()
            .find(|assignment| assignment.node_id.0 == format!("node-{index}"))
            .map(|assignment| assignment.range.end_layer_exclusive - assignment.range.start_layer)
            .unwrap_or(0)
    }

    #[test]
    fn a_small_machine_gets_only_what_its_free_memory_holds() {
        // 8 GB with 4 GB free: 4 GB * 0.9 less the overhead leaves room for 4 layers.
        let nodes = vec![cpu_node(0, 8, 4), cpu_node(1, 32, 24)];
        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        assert!(len_of(&assignments, 0) <= 4, "{assignments:?}");
        assert_eq!(len_of(&assignments, 0) + len_of(&assignments, 1), 40);
    }

    #[test]
    fn the_size_of_a_range_is_the_size_in_the_file_for_every_tier() {
        let nodes = vec![cpu_node(0, 16, 12), cpu_node(1, 32, 24)];
        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        for assignment in &assignments {
            let len = assignment.range.end_layer_exclusive - assignment.range.start_layer;
            assert_eq!(
                assignment.assigned_weight_bytes,
                len as u64 * 527 * 1024 * 1024
            );
        }
    }

    #[test]
    fn a_machine_with_no_room_for_one_layer_stays_out() {
        let nodes = vec![cpu_node(0, 8, 1), cpu_node(1, 32, 28)];
        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        assert_eq!(assignments.len(), 1);
        assert_eq!(len_of(&assignments, 1), 40);
    }

    #[test]
    fn a_lone_machine_goes_past_its_free_memory_up_to_the_hard_limit() {
        // 90% of 20 GB free is not enough for 40 layers, 125% is.
        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&[cpu_node(0, 32, 20)], &big_layers())
            .unwrap();
        assert_eq!(len_of(&assignments, 0), 40);
    }

    #[test]
    fn a_machine_never_goes_far_past_its_free_memory() {
        // 8 GB with 2.7 GB free, as the office machine: 125% of it holds 3 layers at most.
        let mut small = cpu_node(0, 8, 0);
        small.available_ram_mb = 2765;
        let nodes = vec![small, cpu_node(1, 16, 14)];
        let result = Halda::new(SchedulerConfig::default()).assign(&nodes, &big_layers());
        assert!(
            matches!(result, Err(HaldaError::NotEnoughCapacity { .. })),
            "{result:?}"
        );
    }

    #[test]
    fn a_small_disk_limits_the_layers() {
        let mut small_disk = cpu_node(1, 32, 28);
        small_disk.free_disk_mb = 3 * 1024;
        let nodes = vec![cpu_node(0, 32, 28), small_disk];
        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&nodes, &big_layers())
            .unwrap();
        assert!(len_of(&assignments, 1) <= 5, "{assignments:?}");
        assert_eq!(len_of(&assignments, 0) + len_of(&assignments, 1), 40);
    }

    #[test]
    fn a_plan_is_kept_while_its_nodes_hold_it() {
        let halda = Halda::new(SchedulerConfig::default());
        let layers = big_layers();
        let nodes = vec![cpu_node(0, 32, 24), cpu_node(1, 16, 12)];
        let plan = halda.assign(&nodes, &layers).unwrap();
        let ranges = plan
            .iter()
            .map(|assignment| {
                (
                    assignment.node_id.clone(),
                    assignment.range.start_layer,
                    assignment.range.end_layer_exclusive,
                )
            })
            .collect::<Vec<_>>();

        // the nodes come back in another order and with a little less memory free
        let again = vec![cpu_node(1, 16, 11), cpu_node(0, 32, 23)];
        let kept = halda.keep(&ranges, &again, &layers).unwrap();
        assert_eq!(
            kept.iter()
                .map(|a| (&a.node_id, &a.range))
                .collect::<Vec<_>>(),
            plan.iter()
                .map(|a| (&a.node_id, &a.range))
                .collect::<Vec<_>>()
        );

        // one node is gone, or has far too little memory left: the plan does not hold
        assert!(halda.keep(&ranges, &again[1..], &layers).is_err());
        let squeezed = vec![cpu_node(0, 32, 24), cpu_node(1, 16, 1)];
        assert!(halda.keep(&ranges, &squeezed, &layers).is_err());
    }

    #[test]
    fn machines_too_small_together_are_an_error() {
        let nodes = vec![cpu_node(0, 8, 6), cpu_node(1, 8, 6)];
        let result = Halda::new(SchedulerConfig::default()).assign(&nodes, &big_layers());
        assert!(
            matches!(result, Err(HaldaError::NotEnoughCapacity { .. })),
            "{result:?}"
        );
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
