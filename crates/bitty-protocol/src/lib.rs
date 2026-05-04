use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

pub const PROTO_PATH: &str = "../../proto/bitty/v1/cluster.proto";

pub mod pb {
    tonic::include_proto!("bitty.v1");
}

pub mod cli;
pub mod endpoint;
pub mod iroh_transport;
pub mod logits_codec;
pub mod registration;
pub mod security;
pub mod validation;

/// Wire compatibility: bump when changing registration or tensor encoding contract.
pub const BITTY_PROTOCOL_VERSION: u32 = 1;

pub use registration::validate_register_worker;

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

impl NodeTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::S => "S",
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
        }
    }
}

impl TryFrom<&str> for NodeTier {
    type Error = ProtocolConversionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "S" => Ok(Self::S),
            "A" => Ok(Self::A),
            "B" => Ok(Self::B),
            "C" => Ok(Self::C),
            "D" => Ok(Self::D),
            other => Err(ProtocolConversionError::UnknownNodeTier(other.into())),
        }
    }
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

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q4 => "q4",
            Self::Q3 => "q3",
            Self::Q2 => "q2",
            Self::Bit1 => "bit1",
        }
    }
}

impl TryFrom<&str> for Quantization {
    type Error = ProtocolConversionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "fp16" => Ok(Self::Fp16),
            "q4" => Ok(Self::Q4),
            "q3" => Ok(Self::Q3),
            "q2" => Ok(Self::Q2),
            "bit1" => Ok(Self::Bit1),
            other => Err(ProtocolConversionError::UnknownQuantization(other.into())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionKind {
    None,
    Fp8,
    TopK,
    Delta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStage {
    LayerRange,
    EmbeddingAndLayers,
    FinalLayersAndLmHead,
}

impl ModelStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LayerRange => "layer_range",
            Self::EmbeddingAndLayers => "embedding_and_layers",
            Self::FinalLayersAndLmHead => "final_layers_and_lm_head",
        }
    }
}

impl TryFrom<&str> for ModelStage {
    type Error = ProtocolConversionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "" | "layer_range" => Ok(Self::LayerRange),
            "embedding_and_layers" => Ok(Self::EmbeddingAndLayers),
            "final_layers_and_lm_head" => Ok(Self::FinalLayersAndLmHead),
            other => Err(ProtocolConversionError::UnknownModelStage(other.into())),
        }
    }
}

impl CompressionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Fp8 => "fp8",
            Self::TopK => "topk",
            Self::Delta => "delta",
        }
    }
}

impl TryFrom<&str> for CompressionKind {
    type Error = ProtocolConversionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "" | "none" => Ok(Self::None),
            "fp8" => Ok(Self::Fp8),
            "topk" => Ok(Self::TopK),
            "delta" => Ok(Self::Delta),
            other => Err(ProtocolConversionError::UnknownCompression(other.into())),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
    pub compute_capability: u64,
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
    pub ram_mb: u64,
    pub vram_mb: u64,
    pub architecture: String,
    pub gpus: Vec<GpuInfo>,
    pub os_reclaim_score: f64,
    pub worker_endpoint: String,
    pub model_path: String,
    pub backend_type: String,
    pub layer_eligible: bool,
    pub max_layers: u32,
}

impl HardwareProfile {
    pub fn memory_bytes(&self) -> u64 {
        let reported = self.ram_mb.saturating_mul(1024 * 1024);
        if reported > 0 {
            reported
        } else {
            (self.memory_gb * 1024.0 * 1024.0 * 1024.0).max(0.0) as u64
        }
    }

    pub fn effective_compute_score(&self) -> f64 {
        if !self.layer_eligible || self.max_layers == 0 {
            return 0.0;
        }
        let compute = if self.gpu_tflops > 0.0 {
            self.gpu_tflops
        } else {
            self.cpu_tflops * 0.20
        };
        let memory = self.memory_bandwidth_gbps.max(1.0).sqrt();
        let rtt_penalty = 1.0 + (self.network_rtt_ms / 50.0).clamp(0.0, 20.0);
        let uplink_penalty = (100.0 / self.uplink_mbps.max(5.0)).sqrt().max(1.0);
        (compute * memory) / (rtt_penalty * uplink_penalty)
    }

    pub fn backend_type(&self) -> &str {
        if self.backend_type.is_empty() {
            if self.gpu_tflops > 0.0 {
                "gpu"
            } else {
                "cpu"
            }
        } else {
            &self.backend_type
        }
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
    pub disk_offload_fraction: f32,
    pub model_stage: ModelStage,
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
    pub compression: CompressionKind,
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
            compression: CompressionKind::None,
        }
    }

    pub fn with_compression(mut self, compression: CompressionKind) -> Self {
        self.compression = compression;
        self
    }

    pub fn verify_checksum(&self) -> bool {
        checksum(&self.payload) == self.crc32
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenOutput {
    pub request_id: String,
    pub token_position: u32,
    pub token_id: u32,
    pub text: String,
    pub finished: bool,
    pub log_prob: f32,
    pub gen_latency_us: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BitNetLogits {
    pub request_id: String,
    pub token_position: u32,
    pub logits: Vec<f32>,
    pub crc32: u32,
}

impl BitNetLogits {
    pub fn new(request_id: impl Into<String>, token_position: u32, logits: Vec<f32>) -> Self {
        let payload = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self {
            request_id: request_id.into(),
            token_position,
            logits,
            crc32: checksum(&payload),
        }
    }

    pub fn verify_checksum(&self) -> bool {
        let payload = self
            .logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        checksum(&payload) == self.crc32
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyUpdate {
    pub topology_epoch: String,
    pub assignments: Vec<LayerAssignment>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifestMessage {
    pub shard_id: String,
    pub node_id: NodeId,
    pub range: AssignedLayerRange,
    pub byte_len: u64,
    pub sha256_hex: String,
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub node_id: NodeId,
    pub observed_tokens_per_second: f64,
    pub avg_forward_latency_ms: f64,
    pub activation_bytes_per_second: u64,
    pub backend_type: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestId(pub String);

#[derive(Debug, Error)]
pub enum ProtocolConversionError {
    #[error("missing required protobuf field: {0}")]
    MissingField(&'static str),
    #[error("unknown node tier: {0}")]
    UnknownNodeTier(String),
    #[error("unknown quantization: {0}")]
    UnknownQuantization(String),
    #[error("unknown compression: {0}")]
    UnknownCompression(String),
    #[error("unknown activation dtype: {0}")]
    UnknownDType(String),
    #[error("unknown model stage: {0}")]
    UnknownModelStage(String),
    #[error("protocol validation failed: {0}")]
    Validation(String),
}

impl From<&HardwareProfile> for pb::HardwareProfile {
    fn from(profile: &HardwareProfile) -> Self {
        Self {
            node_id: profile.node_id.0.clone(),
            cpu_tflops: profile.cpu_tflops,
            gpu_tflops: profile.gpu_tflops,
            memory_gb: profile.memory_gb,
            memory_bandwidth_gbps: profile.memory_bandwidth_gbps,
            disk_bandwidth_mbps: profile.disk_bandwidth_mbps,
            network_rtt_ms: profile.network_rtt_ms,
            uplink_mbps: profile.uplink_mbps,
            os: profile.os.clone(),
            tier: profile.tier.as_str().into(),
            ram_mb: profile.ram_mb,
            vram_mb: profile.vram_mb,
            architecture: profile.architecture.clone(),
            gpus: profile.gpus.iter().map(Into::into).collect(),
            os_reclaim_score: profile.os_reclaim_score,
            worker_endpoint: profile.worker_endpoint.clone(),
            model_path: profile.model_path.clone(),
            backend_type: profile.backend_type().into(),
            layer_eligible: profile.layer_eligible,
            max_layers: profile.max_layers,
        }
    }
}

impl From<&GpuInfo> for pb::GpuInfo {
    fn from(gpu: &GpuInfo) -> Self {
        Self {
            name: gpu.name.clone(),
            vram_mb: gpu.vram_mb,
            compute_capability: gpu.compute_capability,
        }
    }
}

impl From<pb::GpuInfo> for GpuInfo {
    fn from(gpu: pb::GpuInfo) -> Self {
        Self {
            name: gpu.name,
            vram_mb: gpu.vram_mb,
            compute_capability: gpu.compute_capability,
        }
    }
}

impl TryFrom<pb::HardwareProfile> for HardwareProfile {
    type Error = ProtocolConversionError;

    fn try_from(profile: pb::HardwareProfile) -> Result<Self, Self::Error> {
        let ram_mb = if profile.ram_mb > 0 {
            profile.ram_mb
        } else {
            (profile.memory_gb * 1024.0).max(0.0) as u64
        };
        let profile = Self {
            node_id: NodeId::new(profile.node_id),
            cpu_tflops: profile.cpu_tflops,
            gpu_tflops: profile.gpu_tflops,
            memory_gb: profile.memory_gb,
            memory_bandwidth_gbps: profile.memory_bandwidth_gbps,
            disk_bandwidth_mbps: profile.disk_bandwidth_mbps,
            network_rtt_ms: profile.network_rtt_ms,
            uplink_mbps: profile.uplink_mbps,
            os: profile.os,
            tier: NodeTier::try_from(profile.tier.as_str())?,
            ram_mb,
            vram_mb: profile.vram_mb,
            architecture: if profile.architecture.is_empty() {
                "unknown".into()
            } else {
                profile.architecture
            },
            gpus: profile.gpus.into_iter().map(Into::into).collect(),
            os_reclaim_score: profile.os_reclaim_score,
            worker_endpoint: profile.worker_endpoint,
            model_path: profile.model_path,
            backend_type: profile.backend_type,
            layer_eligible: if profile.max_layers == 0 {
                false
            } else {
                profile.layer_eligible
            },
            max_layers: profile.max_layers,
        };
        validation::validate_model_path(&profile.model_path)?;
        Ok(profile)
    }
}

impl From<&AssignedLayerRange> for pb::AssignedLayerRange {
    fn from(range: &AssignedLayerRange) -> Self {
        Self {
            start_layer: range.start_layer,
            end_layer_exclusive: range.end_layer_exclusive,
            quantization: range.quantization.as_str().into(),
        }
    }
}

impl TryFrom<pb::AssignedLayerRange> for AssignedLayerRange {
    type Error = ProtocolConversionError;

    fn try_from(range: pb::AssignedLayerRange) -> Result<Self, Self::Error> {
        Ok(Self {
            start_layer: range.start_layer,
            end_layer_exclusive: range.end_layer_exclusive,
            quantization: Quantization::try_from(range.quantization.as_str())?,
        })
    }
}

impl From<&LayerAssignment> for pb::LayerAssignment {
    fn from(assignment: &LayerAssignment) -> Self {
        Self {
            node_id: assignment.node_id.0.clone(),
            range: Some((&assignment.range).into()),
            assigned_weight_bytes: assignment.assigned_weight_bytes,
            expected_latency_ms: assignment.expected_latency_ms,
            next_node_id: assignment
                .next_node_id
                .as_ref()
                .map(|node_id| node_id.0.clone())
                .unwrap_or_default(),
            disk_offload_fraction: assignment.disk_offload_fraction,
            model_stage: assignment.model_stage.as_str().into(),
        }
    }
}

impl TryFrom<pb::LayerAssignment> for LayerAssignment {
    type Error = ProtocolConversionError;

    fn try_from(assignment: pb::LayerAssignment) -> Result<Self, Self::Error> {
        Ok(Self {
            node_id: NodeId::new(assignment.node_id),
            range: assignment
                .range
                .ok_or(ProtocolConversionError::MissingField(
                    "LayerAssignment.range",
                ))?
                .try_into()?,
            assigned_weight_bytes: assignment.assigned_weight_bytes,
            expected_latency_ms: assignment.expected_latency_ms,
            next_node_id: (!assignment.next_node_id.is_empty())
                .then(|| NodeId::new(assignment.next_node_id)),
            disk_offload_fraction: assignment.disk_offload_fraction,
            model_stage: ModelStage::try_from(assignment.model_stage.as_str())?,
        })
    }
}

impl From<&ActivationTensor> for pb::ActivationTensor {
    fn from(activation: &ActivationTensor) -> Self {
        Self {
            request_id: activation.request_id.clone(),
            token_position: activation.token_position,
            source_layer: activation.source_layer,
            target_layer: activation.target_layer,
            shape: activation.shape.clone(),
            dtype: match activation.dtype {
                ActivationDType::Fp16 => "fp16",
                ActivationDType::Fp8 => "fp8",
                ActivationDType::I8 => "i8",
            }
            .into(),
            payload: activation.payload.clone(),
            crc32: activation.crc32,
            compression: activation.compression.as_str().into(),
        }
    }
}

impl TryFrom<pb::ActivationTensor> for ActivationTensor {
    type Error = ProtocolConversionError;

    fn try_from(activation: pb::ActivationTensor) -> Result<Self, Self::Error> {
        let dtype = match activation.dtype.as_str() {
            "" | "fp16" => ActivationDType::Fp16,
            "fp8" => ActivationDType::Fp8,
            "i8" => ActivationDType::I8,
            other => return Err(ProtocolConversionError::UnknownDType(other.into())),
        };
        let activation = Self {
            request_id: activation.request_id,
            token_position: activation.token_position,
            source_layer: activation.source_layer,
            target_layer: activation.target_layer,
            shape: activation.shape,
            dtype,
            payload: activation.payload,
            crc32: activation.crc32,
            compression: CompressionKind::try_from(activation.compression.as_str())?,
        };
        validation::validate_activation_tensor(&activation)?;
        Ok(activation)
    }
}

impl From<&TokenOutput> for pb::TokenOutput {
    fn from(token: &TokenOutput) -> Self {
        Self {
            request_id: token.request_id.clone(),
            token_position: token.token_position,
            token_id: token.token_id,
            text: token.text.clone(),
            finished: token.finished,
            log_prob: token.log_prob,
            gen_latency_us: token.gen_latency_us,
        }
    }
}

impl TryFrom<pb::GenerateRequest> for GenerateRequest {
    type Error = ProtocolConversionError;

    fn try_from(request: pb::GenerateRequest) -> Result<Self, Self::Error> {
        let request = Self {
            request_id: request.request_id,
            prompt_tokens: request.prompt_tokens,
            prompt: request.prompt,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
        };
        validation::validate_generate_request(&request)?;
        Ok(request)
    }
}

impl From<&BitNetLogits> for pb::BitNetLogits {
    fn from(logits: &BitNetLogits) -> Self {
        Self {
            request_id: logits.request_id.clone(),
            token_position: logits.token_position,
            logits: Vec::new(),
            crc32: logits.crc32,
            logits_f32_le: logits_codec::logits_f32_le_bytes(&logits.logits),
        }
    }
}

impl TryFrom<pb::BitNetLogits> for BitNetLogits {
    type Error = ProtocolConversionError;

    fn try_from(logits: pb::BitNetLogits) -> Result<Self, Self::Error> {
        let vec = if !logits.logits_f32_le.is_empty() {
            logits_codec::logits_from_f32_le_bytes(&logits.logits_f32_le)?
        } else {
            logits.logits
        };
        let logits = Self {
            request_id: logits.request_id,
            token_position: logits.token_position,
            logits: vec,
            crc32: logits.crc32,
        };
        validation::validate_logits(&logits)?;
        Ok(logits)
    }
}

impl From<&ShardManifestMessage> for pb::ShardManifest {
    fn from(manifest: &ShardManifestMessage) -> Self {
        Self {
            shard_id: manifest.shard_id.clone(),
            node_id: manifest.node_id.0.clone(),
            range: Some((&manifest.range).into()),
            byte_len: manifest.byte_len,
            sha256_hex: manifest.sha256_hex.clone(),
            path: manifest.path.clone(),
        }
    }
}

impl TryFrom<pb::ShardManifest> for ShardManifestMessage {
    type Error = ProtocolConversionError;

    fn try_from(manifest: pb::ShardManifest) -> Result<Self, Self::Error> {
        let manifest = Self {
            shard_id: manifest.shard_id,
            node_id: NodeId::new(manifest.node_id),
            range: manifest
                .range
                .ok_or(ProtocolConversionError::MissingField("ShardManifest.range"))?
                .try_into()?,
            byte_len: manifest.byte_len,
            sha256_hex: manifest.sha256_hex,
            path: manifest.path,
        };
        validation::validate_model_path(&manifest.path)?;
        Ok(manifest)
    }
}

impl From<&Heartbeat> for pb::HeartbeatRequest {
    fn from(heartbeat: &Heartbeat) -> Self {
        Self {
            node_id: heartbeat.node_id.0.clone(),
            observed_tokens_per_second: heartbeat.observed_tokens_per_second,
            avg_forward_latency_ms: heartbeat.avg_forward_latency_ms,
            activation_bytes_per_second: heartbeat.activation_bytes_per_second,
            backend_type: heartbeat.backend_type.clone(),
        }
    }
}

impl From<pb::HeartbeatRequest> for Heartbeat {
    fn from(heartbeat: pb::HeartbeatRequest) -> Self {
        Self {
            node_id: NodeId::new(heartbeat.node_id),
            observed_tokens_per_second: heartbeat.observed_tokens_per_second,
            avg_forward_latency_ms: heartbeat.avg_forward_latency_ms,
            activation_bytes_per_second: heartbeat.activation_bytes_per_second,
            backend_type: heartbeat.backend_type,
        }
    }
}

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

    #[test]
    fn layer_assignment_round_trips_through_proto() {
        let assignment = LayerAssignment {
            node_id: NodeId::new("node-a"),
            range: AssignedLayerRange {
                start_layer: 1,
                end_layer_exclusive: 3,
                quantization: Quantization::Q3,
            },
            assigned_weight_bytes: 42,
            expected_latency_ms: 1.5,
            next_node_id: Some(NodeId::new("node-b")),
            disk_offload_fraction: 0.0,
            model_stage: ModelStage::EmbeddingAndLayers,
        };

        let proto = pb::LayerAssignment::from(&assignment);
        let decoded = LayerAssignment::try_from(proto).unwrap();

        assert_eq!(decoded, assignment);
    }

    #[test]
    fn shard_manifest_round_trips_through_proto() {
        let manifest = ShardManifestMessage {
            shard_id: "shard-a".into(),
            node_id: NodeId::new("node-a"),
            range: AssignedLayerRange {
                start_layer: 0,
                end_layer_exclusive: 2,
                quantization: Quantization::Bit1,
            },
            byte_len: 128,
            sha256_hex: "abc123".into(),
            path: "/tmp/shard.bin".into(),
        };

        let proto = pb::ShardManifest::from(&manifest);
        let decoded = ShardManifestMessage::try_from(proto).unwrap();

        assert_eq!(decoded, manifest);
    }

    #[test]
    fn unknown_activation_dtype_reports_dtype_error() {
        let err = ActivationTensor::try_from(pb::ActivationTensor {
            request_id: "req".into(),
            token_position: 0,
            source_layer: 0,
            target_layer: 1,
            shape: vec![1],
            dtype: "bf16".into(),
            payload: Vec::new(),
            crc32: 0,
            compression: "none".into(),
        })
        .unwrap_err();

        assert!(matches!(err, ProtocolConversionError::UnknownDType(value) if value == "bf16"));
    }

    #[test]
    fn activation_payload_limit_is_enforced() {
        let err = ActivationTensor::try_from(pb::ActivationTensor {
            request_id: "req".into(),
            token_position: 0,
            source_layer: 0,
            target_layer: 1,
            shape: vec![1],
            dtype: "fp16".into(),
            payload: vec![0; validation::MAX_ACTIVATION_PAYLOAD_BYTES + 1],
            crc32: 0,
            compression: "none".into(),
        })
        .unwrap_err();

        assert!(matches!(err, ProtocolConversionError::Validation(_)));
    }

    #[test]
    fn bitnet_logits_round_trips_through_f32_le_wire() {
        let logits = BitNetLogits::new("r", 1, vec![1.0, -2.5, f32::NAN]);
        let proto = pb::BitNetLogits::from(&logits);
        assert!(proto.logits.is_empty());
        assert_eq!(proto.logits_f32_le.len(), 12);
        let decoded = BitNetLogits::try_from(proto).unwrap();
        assert_eq!(decoded.request_id, logits.request_id);
        assert_eq!(decoded.token_position, logits.token_position);
        assert_eq!(decoded.crc32, logits.crc32);
        assert_eq!(decoded.logits.len(), logits.logits.len());
        for (left, right) in decoded.logits.iter().zip(logits.logits.iter()) {
            if left.is_nan() {
                assert!(right.is_nan());
            } else {
                assert!((left - right).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn bitnet_logits_legacy_repeated_float_still_decodes() {
        let proto = pb::BitNetLogits {
            request_id: "x".into(),
            token_position: 0,
            logits: vec![1.0, 2.0],
            crc32: 0,
            logits_f32_le: Vec::new(),
        };
        let decoded = BitNetLogits::try_from(proto).unwrap();
        assert_eq!(decoded.logits, vec![1.0, 2.0]);
    }
}
