//! Common Transformer and SSM operations for CPU inference.

use super::types::RopeCache;

/// RMS Normalization: out = (x / rms) * weight
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len().min(weight.len());
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let rms = (ms + eps).sqrt();
    let scale = 1.0 / rms;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(x[i] * scale * weight[i]);
    }
    out
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: ln(1 + e^x) — numerically stable
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Softmax over a slice (in-place).
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|&v| (v - max).exp()).sum();
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v = (*v - max).exp() * inv;
    }
}

/// Apply RoPE (Rotary Position Embedding) to Q and K in-place using pre-computed cache.
///
/// Supports two rotation styles detected from the model architecture:
/// - `Neox` (i, i+rp) — used by Gemma, Qwen, GPT-NeoX.
/// - `Interleaved` (2i, 2i+1) — used by Llama-family after llama.cpp GGUF conversion
///   (the converter permutes weights so that adjacent pairs are the real/imag components).
///
/// Under GQA (groups > 1), each KV head is rotated exactly once; Q heads are rotated
/// individually. This avoids the previous bug where every Q-head iteration also
/// re-rotated the shared K head, compounding RoPE `groups` times.
pub fn rope_apply(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    style: super::types::RopeStyle,
    rope_cache: &RopeCache,
) {
    if head_dim == 0 {
        return;
    }
    let rp = rope_cache.rope_pair_count().min(head_dim / 2);

    use super::types::RopeStyle;

    match style {
        RopeStyle::Neox => {
            let q_heads = (q.len() / head_dim.max(1)).min(num_heads);
            for h in 0..q_heads {
                let q_off = h * head_dim;
                if q_off + rp * 2 > q.len() {
                    continue;
                }
                for i in 0..rp {
                    let (cos, sin) = rope_cache.get(pos, i);
                    let q0 = q[q_off + i];
                    let q1 = q[q_off + i + rp];
                    q[q_off + i] = q0 * cos - q1 * sin;
                    q[q_off + i + rp] = q0 * sin + q1 * cos;
                }
            }
            // Each KV head rotated once (not groups times)
            let kv_heads = (k.len() / head_dim.max(1)).min(num_kv_heads);
            for h in 0..kv_heads {
                let k_off = h * head_dim;
                if k_off + rp * 2 > k.len() {
                    continue;
                }
                for i in 0..rp {
                    let (cos, sin) = rope_cache.get(pos, i);
                    let k0 = k[k_off + i];
                    let k1 = k[k_off + i + rp];
                    k[k_off + i] = k0 * cos - k1 * sin;
                    k[k_off + i + rp] = k0 * sin + k1 * cos;
                }
            }
        }
        RopeStyle::Interleaved => {
            let q_heads = (q.len() / head_dim.max(1)).min(num_heads);
            for h in 0..q_heads {
                let q_off = h * head_dim;
                if q_off + head_dim > q.len() {
                    continue;
                }
                for i in 0..rp {
                    let (cos, sin) = rope_cache.get(pos, i);
                    let q0 = q[q_off + 2 * i];
                    let q1 = q[q_off + 2 * i + 1];
                    q[q_off + 2 * i] = q0 * cos - q1 * sin;
                    q[q_off + 2 * i + 1] = q0 * sin + q1 * cos;
                }
            }
            let kv_heads = (k.len() / head_dim.max(1)).min(num_kv_heads);
            for h in 0..kv_heads {
                let k_off = h * head_dim;
                if k_off + head_dim > k.len() {
                    continue;
                }
                for i in 0..rp {
                    let (cos, sin) = rope_cache.get(pos, i);
                    let k0 = k[k_off + 2 * i];
                    let k1 = k[k_off + 2 * i + 1];
                    k[k_off + 2 * i] = k0 * cos - k1 * sin;
                    k[k_off + 2 * i + 1] = k0 * sin + k1 * cos;
                }
            }
        }
    }
}
