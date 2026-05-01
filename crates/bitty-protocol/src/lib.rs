use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const PROTO_PATH: &str = "../../proto/bitty/v1/cluster.proto";

pub mod pb {
    tonic::include_proto!("bitty.v1");
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeTier {
    S,
    A,
    B,
    C,
    D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantization {
    Fp16,
    Q4,
    Q3,
    Q2,
    Bit1,
}

impl Quantization {
    pub fn bytes_per_weight(self) -> f64 {
        match self {
            Self::Fp16 => 2.0,
            Self::Q4 => 0.5,
            Self::Q3 => 0.375,
            Self::Q2 => 0.25,
            Self::Bit1 => 0.125,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub node_id: NodeId,
    pub cpu_tflops: f64,
    pub gpu_tflops: f64,
    pub memory_gb: f64,
    pub memory_bandwidth_gbps: f64,
    pub disk_bandwidth_mbps: f64,
    pub network_rtt_ms: f64,
    pub uplink_mbps: f64,
    pub os: String,
    pub tier: NodeTier,
}

impl HardwareProfile {
    pub fn memory_bytes(&self) -> u64 {
        (self.memory_gb * 1024.0 * 1024.0 * 1024.0).max(0.0) as u64
    }

    pub fn effective_compute_score(&self) -> f64 {
        let compute = self.gpu_tflops.max(self.cpu_tflops * 0.35);
        let memory = self.memory_bandwidth_gbps.max(1.0).sqrt();
        let network_penalty = 1.0 + (self.network_rtt_ms / 100.0).clamp(0.0, 10.0);
        (compute * memory) / network_penalty
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerMetadata {
    pub layer_id: u32,
    pub weight_bytes: u64,
    pub activation_bytes: u64,
    pub estimated_flops: f64,
    pub precision_critical: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssignedLayerRange {
    pub start_layer: u32,
    pub end_layer_exclusive: u32,
    pub quantization: Quantization,
}

impl AssignedLayerRange {
    pub fn contains(&self, layer_id: u32) -> bool {
        self.start_layer <= layer_id && layer_id < self.end_layer_exclusive
    }

    pub fn len(&self) -> u32 {
        self.end_layer_exclusive.saturating_sub(self.start_layer)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub node_id: NodeId,
    pub range: AssignedLayerRange,
    pub assigned_weight_bytes: u64,
    pub expected_latency_ms: f64,
    pub next_node_id: Option<NodeId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationDType {
    Fp16,
    Fp8,
    I8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActivationTensor {
    pub request_id: String,
    pub token_position: u32,
    pub source_layer: u32,
    pub target_layer: u32,
    pub shape: Vec<u32>,
    pub dtype: ActivationDType,
    pub payload: Vec<u8>,
    pub crc32: u32,
}

impl ActivationTensor {
    pub fn new(
        request_id: impl Into<String>,
        token_position: u32,
        source_layer: u32,
        target_layer: u32,
        shape: Vec<u32>,
        dtype: ActivationDType,
        payload: Vec<u8>,
    ) -> Self {
        let crc32 = checksum(&payload);
        Self {
            request_id: request_id.into(),
            token_position,
            source_layer,
            target_layer,
            shape,
            dtype,
            payload,
            crc32,
        }
    }

    pub fn verify_checksum(&self) -> bool {
        checksum(&self.payload) == self.crc32
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenOutput {
    pub request_id: String,
    pub token_position: u32,
    pub token_id: u32,
    pub text: String,
    pub finished: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyUpdate {
    pub topology_epoch: String,
    pub assignments: Vec<LayerAssignment>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub node_id: NodeId,
    pub observed_tokens_per_second: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestId(pub String);

fn checksum(payload: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(payload);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_checksum_detects_corruption() {
        let mut activation = ActivationTensor::new(
            "req",
            0,
            0,
            1,
            vec![1, 4],
            ActivationDType::Fp16,
            vec![1, 2, 3],
        );
        assert!(activation.verify_checksum());
        activation.payload[0] = 99;
        assert!(!activation.verify_checksum());
    }
}
