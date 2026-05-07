//! Layer dispatcher. Routes each layer's forward pass to the correct implementation.

pub mod attention;
pub mod mlp;
pub mod ssm;

use crate::cpu_backend::matmul;
use crate::cpu_backend::ops;
use crate::cpu_backend::types::*;

type Result<T> = std::result::Result<T, String>;

impl CpuLayer {
    /// Run one transformer/SSM layer forward pass.
    ///
    /// Standard Llama-style block:
    ///     x1 = x + o_proj(attn(rms_norm_1(x)))
    ///     x2 = x1 + mlp(rms_norm_2(x1))
    /// Earlier versions incorrectly normed `o_proj(attn(...))` (without the
    /// residual add) before the MLP, producing garbled logits.
    pub fn forward(
        &self,
        hidden: &[f32],
        pos: usize,
        kv_cache: &mut KvCache,
        ssm_states: &mut Vec<SsmState>,
        meta: &CpuModelMetadata,
    ) -> Result<Vec<f32>> {
        let d = meta.hidden_size;
        let inter = meta.intermediate_size;

        // ---- Pre-attention norm
        let normed = ops::rms_norm(hidden, &self.input_ln, meta.rms_norm_eps);

        // ---- Attention or SSM block
        let block_out = match &self.kind {
            LayerKind::Attention(ref w) => {
                attention::forward(&normed, w, pos, kv_cache, meta, self.layer_idx)?
            }
            LayerKind::Ssm(ref w) => {
                let state = &mut ssm_states[self.layer_idx];
                ssm::forward(&normed, w, pos, state, meta)?
            }
        };

        // ---- O projection (only for attention layers; SSM already fused)
        let block_proj = match &self.kind {
            LayerKind::Attention(_) => {
                let actual_dim = block_out.len();
                matmul::matmul(&block_out, self.attn_o_proj(), actual_dim, d)?
            }
            LayerKind::Ssm(_) => block_out,
        };

        // ---- First residual: x1 = x + block_proj
        let mut x1 = vec![0f32; d];
        for i in 0..d {
            x1[i] = hidden[i] + block_proj[i];
        }

        // ---- Post-attention norm on the residual sum (not on the raw block_proj!)
        let post_normed = ops::rms_norm(&x1, &self.post_attn_ln, meta.rms_norm_eps);

        // ---- FFN
        let ffn_out = mlp::forward(&post_normed, &self.mlp, d, inter)?;

        // ---- Second residual: x2 = x1 + ffn_out
        let mut out = vec![0f32; d];
        for i in 0..d {
            out[i] = x1[i] + ffn_out[i];
        }
        Ok(out)
    }
}
