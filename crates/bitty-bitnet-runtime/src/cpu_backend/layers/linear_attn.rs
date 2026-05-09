//! Qwen3.5 recurrent block: short conv on QKV mix + gated delta net (autoregressive step).
//! Ported from llama.cpp `build_layer_attn_linear` + `build_delta_net_autoregressive`.

use crate::cpu_backend::matmul;
use crate::cpu_backend::ops::{self, rms_norm, silu};
use crate::cpu_backend::types::{CpuModelMetadata, LinearAttnState, LinearAttnWeights, PackedTensor};

type Result<T> = std::result::Result<T, String>;

/// One causal conv step matching `ggml_compute_forward_ssm_conv_f32` (single new token).
fn ssm_conv_one(
    conv_state: &mut [f32],
    qkv_col: &[f32],
    kernel: &[f32],
    d_conv: usize,
    conv_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(qkv_col.len(), conv_dim);
    debug_assert_eq!(kernel.len(), d_conv * conv_dim);
    let past_rows = d_conv.saturating_sub(1);
    debug_assert_eq!(conv_state.len(), past_rows * conv_dim);

    let mut out = vec![0f32; conv_dim];
    for ch in 0..conv_dim {
        let mut sum = 0f32;
        for t in 0..past_rows {
            let s_idx = t * conv_dim + ch;
            let k_idx = t + ch * d_conv;
            sum += conv_state[s_idx] * kernel[k_idx];
        }
        let t = d_conv - 1;
        let k_idx = t + ch * d_conv;
        sum += qkv_col[ch] * kernel[k_idx];
        out[ch] = sum;
    }

    if past_rows > 0 {
        conv_state.copy_within(conv_dim..past_rows * conv_dim, 0);
        conv_state[(past_rows - 1) * conv_dim..past_rows * conv_dim].copy_from_slice(qkv_col);
    }

    out
}

/// Autoregressive gated delta net (`build_delta_net_autoregressive`, n_tokens == 1).
/// Returns flattened `[head_v * num_v_heads]` core output before gated norm.
fn delta_net_ar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    s_dim: usize,
    h: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (s_dim as f32).sqrt();
    let mut q_sc = vec![0f32; q.len()];
    let mut k_sc = vec![0f32; k.len()];
    for i in 0..q.len() {
        q_sc[i] = q[i] * scale;
        k_sc[i] = k[i] * scale;
    }

    let mut g_exp = vec![0f32; h];
    for hi in 0..h {
        g_exp[hi] = gate[hi].exp();
    }
    for hi in 0..h {
        let off = hi * s_dim * s_dim;
        for i in 0..s_dim * s_dim {
            state[off + i] *= g_exp[hi];
        }
    }

    // sk[i,h] = sum_{i'} state[i', i, h] * k[i', h]
    let mut sk = vec![0f32; s_dim * h];
    for hi in 0..h {
        for i in 0..s_dim {
            let mut acc = 0f32;
            for ip in 0..s_dim {
                let sidx = hi * s_dim * s_dim + i * s_dim + ip;
                let kidx = hi * s_dim + ip;
                acc += state[sidx] * k_sc[kidx];
            }
            sk[hi * s_dim + i] = acc;
        }
    }

    let mut d = vec![0f32; s_dim * h];
    for hi in 0..h {
        for i in 0..s_dim {
            let idx = hi * s_dim + i;
            d[idx] = (v[idx] - sk[idx]) * beta[hi];
        }
    }

    // state[i,j,h] += k[i,h] * d[j,h]
    for hi in 0..h {
        for j in 0..s_dim {
            let dj = d[hi * s_dim + j];
            for i in 0..s_dim {
                let sidx = hi * s_dim * s_dim + j * s_dim + i;
                let kidx = hi * s_dim + i;
                state[sidx] += k_sc[kidx] * dj;
            }
        }
    }

    // out[j,h] = sum_i state[i,j,h] * q[i,h]
    let mut out = vec![0f32; s_dim * h];
    for hi in 0..h {
        for j in 0..s_dim {
            let mut acc = 0f32;
            for i in 0..s_dim {
                let sidx = hi * s_dim * s_dim + j * s_dim + i;
                let qidx = hi * s_dim + i;
                acc += state[sidx] * q_sc[qidx];
            }
            out[hi * s_dim + j] = acc;
        }
    }
    out
}

fn dequant_f32(t: &PackedTensor, elems: usize) -> Result<Vec<f32>> {
    use crate::cpu_backend::dequant::dequantize_slice;
    Ok(dequantize_slice(&t.data, t.ggml_type, elems))
}

fn l2_norm_slice(x: &mut [f32], start: usize, len: usize, eps: f32) {
    let slice = &mut x[start..start + len];
    let mut sum = 0f32;
    for &v in slice.iter() {
        sum += v * v;
    }
    let n = (sum + eps).sqrt();
    let inv = if n > 0.0 { 1.0 / n } else { 1.0 };
    for v in slice.iter_mut() {
        *v *= inv;
    }
}

/// Forward one recurrent (linear attention) layer.
pub fn forward(
    hidden: &[f32],
    w: &LinearAttnWeights,
    meta: &CpuModelMetadata,
    state: &mut LinearAttnState,
) -> Result<Vec<f32>> {
    let d = meta.hidden_size;
    let head_k = meta.ssm_d_state;
    let num_k = meta.ssm_n_group;
    let num_v = meta.ssm_dt_rank;
    let head_v = meta.ssm_d_inner / num_v.max(1);
    let key_dim = head_k * num_k;
    let value_dim = head_v * num_v;
    let conv_dim = key_dim * 2 + value_dim;
    let d_conv = meta.ssm_d_conv.max(1);

    let qkv = matmul::matmul(hidden, &w.wqkv, d, conv_dim)?;
    let z = matmul::matmul(hidden, &w.wqkv_gate, d, value_dim)?;

    let beta_raw = matmul::matmul(hidden, &w.ssm_beta, d, num_v)?;
    let mut beta = vec![0f32; num_v];
    for i in 0..num_v {
        beta[i] = 1.0 / (1.0 + (-beta_raw[i]).exp());
    }

    let alpha = matmul::matmul(hidden, &w.ssm_alpha, d, num_v)?;
    let mut gate = vec![0f32; num_v];
    for i in 0..num_v {
        let b = w.ssm_dt_bias.get(i).copied().unwrap_or(0.0);
        let a = w.ssm_a.get(i).copied().unwrap_or(0.0);
        let sp = ops::softplus(alpha[i] + b);
        gate[i] = sp * a;
    }

    let k_w = dequant_f32(&w.ssm_conv1d, d_conv * conv_dim)?;
    let conv_out = ssm_conv_one(&mut state.conv_state, &qkv, &k_w, d_conv, conv_dim);
    let mut conv_silu: Vec<f32> = conv_out.iter().map(|&v| silu(v)).collect();

    l2_norm_slice(&mut conv_silu, 0, key_dim, meta.rms_norm_eps);
    l2_norm_slice(&mut conv_silu, key_dim, key_dim, meta.rms_norm_eps);

    let mut q_flat = vec![0f32; head_k * num_v];
    let mut k_flat = vec![0f32; head_k * num_v];
    let mut v_flat = vec![0f32; head_v * num_v];

    if num_k != num_v && num_v % num_k == 0 {
        let rep = num_v / num_k;
        for vh in 0..num_v {
            let kh = vh / rep;
            for t in 0..head_k {
                q_flat[vh * head_k + t] = conv_silu[kh * head_k + t];
                k_flat[vh * head_k + t] = conv_silu[key_dim + kh * head_k + t];
            }
        }
        for vh in 0..num_v {
            for t in 0..head_v {
                v_flat[vh * head_v + t] = conv_silu[2 * key_dim + vh * head_v + t];
            }
        }
    } else {
        q_flat.copy_from_slice(&conv_silu[..key_dim]);
        k_flat.copy_from_slice(&conv_silu[key_dim..2 * key_dim]);
        v_flat.copy_from_slice(&conv_silu[2 * key_dim..]);
    }

    let s = head_v;
    debug_assert_eq!(state.s_v, s);
    debug_assert_eq!(state.h_v, num_v);

    let out_core = delta_net_ar(
        &q_flat,
        &k_flat,
        &v_flat,
        &gate,
        &beta,
        &mut state.delta_state,
        s,
        num_v,
    );

    let mut normed = vec![0f32; value_dim];
    for vh in 0..num_v {
        let off = vh * head_v;
        let slice = &out_core[off..off + head_v];
        let n = rms_norm(slice, &w.ssm_norm, meta.rms_norm_eps);
        let z_off = vh * head_v;
        for i in 0..head_v {
            normed[off + i] = n[i] * silu(z[z_off + i]);
        }
    }

    matmul::matmul(&normed, &w.ssm_out, value_dim, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_net_ar_smoke() {
        let s = 4;
        let h = 2;
        let mut state = vec![0f32; s * s * h];
        let q = vec![0.1f32; s * h];
        let k = vec![0.2f32; s * h];
        let v = vec![0.3f32; s * h];
        let gate = vec![0.0f32; h];
        let beta = vec![0.5f32; h];
        let out = delta_net_ar(&q, &k, &v, &gate, &beta, &mut state, s, h);
        assert_eq!(out.len(), s * h);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
