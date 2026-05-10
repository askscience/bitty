//! Layer dispatcher. Routes each layer's forward pass to the correct implementation.

pub mod attention;
pub mod linear_attn;
pub mod mlp;
pub mod ssm;

use crate::cpu_backend::matmul;
use crate::cpu_backend::ops;
use crate::cpu_backend::types::{CpuLayer, CpuModelMetadata, KvCache, LayerKind, RecurrentState, RopeCache};

type Result<T> = std::result::Result<T, String>;

impl CpuLayer {
    pub fn forward(
        &self,
        hidden: &[f32],
        pos: usize,
        kv_cache: &mut KvCache,
        recurrent: &mut Vec<RecurrentState>,
        meta: &CpuModelMetadata,
        rope_cache: &RopeCache,
    ) -> Result<Vec<f32>> {
        let d = meta.hidden_size;
        let inter = meta.intermediate_size;

        // ---- Pre-attention norm
        let normed = ops::rms_norm(hidden, &self.input_ln, meta.rms_norm_eps);

        // ---- Attention or SSM block
        let block_out = match &self.kind {
            LayerKind::Attention(ref w) => {
                attention::forward(&normed, w, pos, kv_cache, meta, self.layer_idx, rope_cache)?
            }
            LayerKind::Ssm(ref w) => {
                let RecurrentState::Mamba(state) = &mut recurrent[self.layer_idx] else {
                    return Err(format!(
                        "layer {}: expected Mamba recurrent state",
                        self.layer_idx
                    ));
                };
                ssm::forward(&normed, w, pos, state, meta)?
            }
            LayerKind::LinearAttn(ref w) => {
                let RecurrentState::QwenLinear(state) = &mut recurrent[self.layer_idx] else {
                    return Err(format!(
                        "layer {}: expected Qwen linear recurrent state",
                        self.layer_idx
                    ));
                };
                linear_attn::forward(&normed, w, meta, state)?
            }
        };

        // ---- O projection (only for attention layers; SSM / linear already fused)
        let block_proj = match &self.kind {
            LayerKind::Attention(_) => {
                let actual_dim = block_out.len();
                matmul::matmul(&block_out, self.attn_o_proj(), actual_dim, d)?
            }
            LayerKind::Ssm(_) | LayerKind::LinearAttn(_) => block_out,
        };

        // ---- First residual: x1 = hidden + block_proj
        let mut x1 = vec![0f32; d];
        for i in 0..d {
            x1[i] = hidden[i] + block_proj[i];
        }

        // Gemma3 applies an extra norm (post_attention_norm) on the residual sum
        // between attention and FFN.
        let ffn_input = if let Some(ref post_attn) = self.post_attention_norm {
            if post_attn.len() == d {
                ops::rms_norm(&x1, post_attn, meta.rms_norm_eps)
            } else {
                x1.clone()
            }
        } else {
            x1.clone()
        };

        // ---- Post-attention / pre-FFN norm
        let pre_ffn = if let Some(ref pre_f) = self.pre_ffn_norm {
            if pre_f.len() == d {
                ops::rms_norm(&ffn_input, pre_f, meta.rms_norm_eps)
            } else {
                ops::rms_norm(&ffn_input, &self.post_attn_ln, meta.rms_norm_eps)
            }
        } else {
            ops::rms_norm(&ffn_input, &self.post_attn_ln, meta.rms_norm_eps)
        };

        // ---- FFN
        let ffn_raw = mlp::forward(&pre_ffn, &self.mlp, d, inter)?;

        // Gemma3 applies a norm on the FFN output before the second residual.
        let ffn_out = if let Some(ref post_f) = self.post_ffn_norm {
            if post_f.len() == d {
                ops::rms_norm(&ffn_raw, post_f, meta.rms_norm_eps)
            } else {
                ffn_raw
            }
        } else {
            ffn_raw
        };

        // ---- Second residual: x2 = x1 + ffn_out
        let mut out = vec![0f32; d];
        for i in 0..d {
            out[i] = x1[i] + ffn_out[i];
        }
        Ok(out)
    }
}
