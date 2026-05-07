//! Core types for the CPU inference backend.

use std::collections::HashMap;

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct CpuModelMetadata {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub activation: ActivationFn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFn {
    Silu,
    Relu2,
    Gelu,
}

/// A weight tensor in its native packed GGML quantization format.
#[derive(Clone)]
pub struct PackedTensor {
    pub data: Vec<u8>,
    pub ggml_type: u32,
    pub shape: Vec<usize>,
    pub name: String,
}

impl PackedTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dummy() -> Self {
        Self { data: vec![0u8; 4], ggml_type: 0, shape: vec![1, 1], name: "dummy".into() }
    }
}

pub struct CpuWeights {
    pub embed_tokens: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<CpuLayer>,
    pub lm_head: Option<LmHead>,
}

pub enum LmHead {
    Tied,
    Packed(PackedTensor),
}

pub struct CpuLayer {
    pub kind: LayerKind,
    pub input_ln: Vec<f32>,
    pub post_attn_ln: Vec<f32>,
    pub mlp: MlpBlock,
    pub layer_idx: usize,
}

impl CpuLayer {
    pub fn attn_o_proj(&self) -> &PackedTensor {
        match &self.kind {
            LayerKind::Attention(w) => &w.o_proj,
            LayerKind::Ssm(_) => unreachable!(),
        }
    }
}

pub enum LayerKind {
    Attention(AttentionWeights),
    Ssm(SsmWeights),
}

pub struct AttentionWeights {
    pub q_proj: PackedTensor,
    pub k_proj: PackedTensor,
    pub v_proj: PackedTensor,
    pub o_proj: PackedTensor,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub is_fused_qkv: bool,
}

pub struct SsmWeights {
    pub in_proj: PackedTensor,
    pub conv1d_weight: PackedTensor,
    pub conv1d_bias: Option<Vec<f32>>,
    pub dt_proj_weight: PackedTensor,
    pub dt_proj_bias: Option<Vec<f32>>,
    pub a_log: Vec<f32>,
    pub d_param: Vec<f32>,
    pub out_proj: PackedTensor,
    pub ssm_norm: Vec<f32>,
    pub d_state: usize,
    pub d_inner: usize,
    pub kernel_size: usize,
}

pub struct MlpBlock {
    pub up_proj: PackedTensor,
    pub gate_proj: PackedTensor,
    pub down_proj: PackedTensor,
}

pub struct KvCache {
    pub keys: HashMap<usize, Vec<f32>>,
    pub values: HashMap<usize, Vec<f32>>,
    pub seq_len: usize,
}

impl KvCache {
    pub fn new() -> Self {
        Self { keys: HashMap::new(), values: HashMap::new(), seq_len: 0 }
    }

    pub fn reserve(&mut self, num_layers: usize, max_kv_dim: usize, max_seq: usize) {
        for layer in 0..num_layers {
            self.keys.entry(layer).or_insert_with(|| vec![0f32; max_kv_dim * max_seq]);
            self.values.entry(layer).or_insert_with(|| vec![0f32; max_kv_dim * max_seq]);
        }
    }
}

pub struct SsmState {
    pub h: Vec<f32>,
    pub conv_state: Vec<f32>,
    pub d_inner: usize,
    pub d_state: usize,
    pub kernel_size: usize,
}

impl SsmState {
    pub fn new(d_inner: usize, d_state: usize, kernel_size: usize) -> Self {
        Self {
            h: vec![0f32; d_inner * d_state],
            conv_state: vec![0f32; d_inner * (kernel_size.saturating_sub(1))],
            d_inner,
            d_state,
            kernel_size,
        }
    }
}

pub struct QuantInfo {
    pub tensor_types: HashMap<String, (u32, usize)>,
}
