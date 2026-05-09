//! Core types for the CPU inference backend.

use bytes::Bytes;
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
    /// Dimensions that receive RoPE (may be <= head_dim for partial RoPE).
    pub rope_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub activation: ActivationFn,
    // Qwen3.5 hybrid (gated delta net + full attention)
    pub is_qwen35: bool,
    pub full_attention_interval: u32,
    /// MRoPE / IMRoPE section sizes (4 ints); used when `is_qwen35` for full-attention layers.
    pub rope_sections: [u32; 4],
    pub ssm_d_conv: usize,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_dt_rank: usize,
    pub ssm_n_group: usize,
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
    pub data: Bytes,
    pub ggml_type: u32,
    pub shape: Vec<usize>,
    pub name: String,
}

impl PackedTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dummy() -> Self {
        Self {
            data: Bytes::from(vec![0u8; 4]),
            ggml_type: 0,
            shape: vec![1, 1],
            name: "dummy".into(),
        }
    }
}

/// Pre-computed RoPE cos/sin cache to avoid per-token trig calls.
#[derive(Debug, Clone)]
pub struct RopeCache {
    cos: Vec<f32>,
    sin: Vec<f32>,
    max_seq: usize,
    half_dim: usize,
}

impl RopeCache {
    pub fn new(max_seq_len: usize, rope_dim: usize, rope_theta: f32) -> Self {
        let half_dim = rope_dim / 2;
        let total = max_seq_len * half_dim;
        let mut cos = vec![0f32; total];
        let mut sin = vec![0f32; total];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let theta = 1.0 / rope_theta.powf((2 * i) as f32 / rope_dim as f32);
                let idx = pos * half_dim + i;
                cos[idx] = (pos as f32 * theta).cos();
                sin[idx] = (pos as f32 * theta).sin();
            }
        }
        Self {
            cos,
            sin,
            max_seq: max_seq_len,
            half_dim,
        }
    }

    #[inline]
    pub fn rope_pair_count(&self) -> usize {
        self.half_dim
    }

    #[inline]
    pub fn get(&self, pos: usize, i: usize) -> (f32, f32) {
        let idx = pos * self.half_dim + i;
        (self.cos[idx], self.sin[idx])
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
            LayerKind::Ssm(_) | LayerKind::LinearAttn(_) => unreachable!(),
        }
    }
}

pub enum LayerKind {
    Attention(AttentionWeights),
    Ssm(SsmWeights),
    /// Qwen3.5 recurrent layer (gated delta net), not Mamba.
    LinearAttn(LinearAttnWeights),
}

pub struct AttentionWeights {
    pub q_proj: PackedTensor,
    pub k_proj: PackedTensor,
    pub v_proj: PackedTensor,
    pub o_proj: PackedTensor,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub attn_gate: Option<PackedTensor>,
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

/// Qwen3.5 linear (recurrent) attention block weights.
#[derive(Clone)]
pub struct LinearAttnWeights {
    pub wqkv: PackedTensor,
    pub wqkv_gate: PackedTensor,
    pub ssm_conv1d: PackedTensor,
    pub ssm_dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub ssm_alpha: PackedTensor,
    pub ssm_beta: PackedTensor,
    pub ssm_norm: Vec<f32>,
    pub ssm_out: PackedTensor,
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
        Self {
            keys: HashMap::new(),
            values: HashMap::new(),
            seq_len: 0,
        }
    }

    pub fn reserve(&mut self, num_layers: usize, max_kv_dim: usize, max_seq: usize) {
        for layer in 0..num_layers {
            self.keys
                .entry(layer)
                .or_insert_with(|| vec![0f32; max_kv_dim * max_seq]);
            self.values
                .entry(layer)
                .or_insert_with(|| vec![0f32; max_kv_dim * max_seq]);
        }
    }
}

#[derive(Debug, Clone)]
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

/// Recurrent state for Qwen3.5 gated delta net (per layer).
#[derive(Debug, Clone)]
pub struct LinearAttnState {
    /// Previous conv ring: length `(kernel_size - 1) * conv_channels`.
    pub conv_state: Vec<f32>,
    /// Delta-net state `[S_v, S_v, H_v]` stored row-major flat.
    pub delta_state: Vec<f32>,
    pub kernel_size: usize,
    pub conv_channels: usize,
    pub s_v: usize,
    pub h_v: usize,
}

impl LinearAttnState {
    pub fn new(kernel_size: usize, conv_channels: usize, s_v: usize, h_v: usize) -> Self {
        let conv_len = kernel_size.saturating_sub(1) * conv_channels;
        Self {
            conv_state: vec![0f32; conv_len],
            delta_state: vec![0f32; s_v * s_v * h_v],
            kernel_size,
            conv_channels,
            s_v,
            h_v,
        }
    }
}

/// Per-layer recurrent backing storage (Mamba vs Qwen linear).
#[derive(Debug, Clone)]
pub enum RecurrentState {
    Mamba(SsmState),
    QwenLinear(LinearAttnState),
    None,
}

impl RecurrentState {
    pub fn new_mamba(d_inner: usize, d_state: usize, kernel_size: usize) -> Self {
        RecurrentState::Mamba(SsmState::new(d_inner, d_state, kernel_size))
    }

    pub fn new_qwen_linear(
        kernel_size: usize,
        conv_channels: usize,
        s_v: usize,
        h_v: usize,
    ) -> Self {
        RecurrentState::QwenLinear(LinearAttnState::new(
            kernel_size,
            conv_channels,
            s_v,
            h_v,
        ))
    }
}

pub struct QuantInfo {
    pub tensor_types: HashMap<String, (u32, usize)>,
}
