//! Standard multi-head attention with RoPE, KV cache, and GQA support.

use crate::cpu_backend::matmul;
use crate::cpu_backend::ops;
use crate::cpu_backend::types::{
    AttentionWeights, CpuModelMetadata, KvCache, RopeCache,
};

type Result<T> = std::result::Result<T, String>;

pub fn forward(
    hidden: &[f32],
    w: &AttentionWeights,
    pos: usize,
    cache: &mut KvCache,
    meta: &CpuModelMetadata,
    layer_idx: usize,
    rope_cache: &RopeCache,
) -> Result<Vec<f32>> {
    let d = meta.hidden_size;
    let n_heads = meta.num_heads;
    let n_kv = meta.num_kv_heads;
    let hd = meta.head_dim;
    let q_dim = n_heads * hd;
    let k_dim = n_kv * hd;

    let mut q_fused_gate: Option<Vec<f32>> = None;

    // Project to Q, K, V
    let (q, k, v) = if w.is_fused_qkv {
        let fused_out = *w.q_proj.shape.last().unwrap_or(&(q_dim + k_dim * 2));
        let qkv = matmul::matmul(hidden, &w.q_proj, d, fused_out)?;
        let q_out = fused_out / 3;
        let k_out = q_out;
        let q = qkv[..q_out.min(qkv.len())].to_vec();
        let k_start = q_out.min(qkv.len());
        let k = qkv[k_start..(k_start + k_out).min(qkv.len())].to_vec();
        let v = qkv[(k_start + k_out).min(qkv.len())..].to_vec();
        (q, k, v)
    } else {
        // Use actual tensor output dims (handles Qwen head_dim=256, LLaMA head_dim=64)
        let q_out = *w.q_proj.shape.last().unwrap_or(&q_dim);
        let k_out = *w.k_proj.shape.last().unwrap_or(&k_dim);
        let v_out = *w.v_proj.shape.last().unwrap_or(&k_dim);
        let k = matmul::matmul(hidden, &w.k_proj, d, k_out)?;
        let v = matmul::matmul(hidden, &w.v_proj, d, v_out)?;
        let q_full = matmul::matmul(hidden, &w.q_proj, d, q_out)?;
        let q = if q_full.len() == 2 * n_heads * hd {
            let half = n_heads * hd;
            q_fused_gate = Some(q_full[half..].to_vec());
            q_full[..half].to_vec()
        } else {
            q_full
        };
        (q, k, v)
    };

    // Compute actual per-head dims from raw projections
    let raw_q_dim = q.len();
    let raw_k_dim = k.len();
    let actual_hd = if n_heads > 0 { raw_q_dim / n_heads } else { hd };
    let actual_kv_hd = if n_kv > 0 { raw_k_dim / n_kv } else { 1 };

    // Apply QK norms — per-head for models like Qwen where norm weight == head_dim
    let mut q = q;
    let mut k = k;
    if let Some(ref qn) = w.q_norm {
        if !qn.is_empty() && qn.len() == actual_hd && actual_hd > 0 {
            let nh = raw_q_dim / actual_hd.max(1);
            let mut q_normed = Vec::with_capacity(raw_q_dim);
            for h in 0..nh {
                let start = h * actual_hd;
                let head = &q[start..(start + actual_hd).min(q.len())];
                q_normed.extend_from_slice(&ops::rms_norm(head, qn, 1e-5));
            }
            q = q_normed;
        } else {
            q = ops::rms_norm(&q, qn, 1e-5);
        }
    }
    if let Some(ref kn) = w.k_norm {
        if !kn.is_empty() && kn.len() == actual_kv_hd && actual_kv_hd > 0 {
            let nkh = raw_k_dim / actual_kv_hd.max(1);
            let mut k_normed = Vec::with_capacity(raw_k_dim);
            for h in 0..nkh {
                let start = h * actual_kv_hd;
                let head = &k[start..(start + actual_kv_hd).min(k.len())];
                k_normed.extend_from_slice(&ops::rms_norm(head, kn, 1e-5));
            }
            k = k_normed;
        } else {
            k = ops::rms_norm(&k, kn, 1e-5);
        }
    }

    // Compute actual dims after norms (norms preserve shape)
    let actual_q_dim = q.len();
    let actual_k_dim = k.len();

    // Apply RoPE — use actual dims and detected rope style
    ops::rope_apply(
        &mut q,
        &mut k,
        pos,
        actual_hd,
        n_heads,
        n_kv,
        meta.rope_style,
        rope_cache,
    );

    // Store K,V in cache
    {
        let kd = actual_k_dim;
        let max_seq = meta.max_seq_len;
        let key_buf = cache
            .keys
            .entry(layer_idx)
            .or_insert_with(|| vec![0f32; kd * max_seq.max(4096)]);
        let val_buf = cache
            .values
            .entry(layer_idx)
            .or_insert_with(|| vec![0f32; kd * max_seq.max(4096)]);
        let off = cache.seq_len * kd;
        if off + k.len() > key_buf.len() || off + v.len() > val_buf.len() {
            return Err(format!("KV cache overflow at layer {layer_idx}"));
        }
        key_buf[off..off + k.len()].copy_from_slice(&k);
        val_buf[off..off + v.len()].copy_from_slice(&v);
    }

    // Attention with cache
    let groups = n_heads / n_kv.max(1);
    let cur_len = cache.seq_len + 1;
    let mut out = vec![0f32; actual_q_dim];
    let kd = actual_k_dim;
    let key_cache = cache
        .keys
        .get(&layer_idx)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let val_cache = cache
        .values
        .get(&layer_idx)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    for h in 0..n_heads {
        let kv_h = h / groups.max(1);
        let q_start = h * actual_hd;
        let o_start = h * actual_hd;
        if q_start + actual_hd > actual_q_dim {
            continue;
        }
        let mut scores = vec![0f32; cur_len];
        for kj in 0..cur_len {
            let k_start = kv_h * actual_kv_hd + kj * kd;
            if k_start + actual_hd > key_cache.len() {
                continue;
            }
            let mut dot = 0f32;
            let dot_dim = actual_hd.min(actual_kv_hd);
            for d in 0..dot_dim {
                dot += q[q_start + d] * key_cache[k_start + d];
            }
            scores[kj] = dot / (dot_dim as f32).sqrt();
        }
        ops::softmax(&mut scores);
        for d in 0..actual_hd {
            let mut sum = 0f32;
            for kj in 0..cur_len {
                let v_start = kv_h * actual_kv_hd + kj * kd;
                if v_start + d < val_cache.len() {
                    sum += scores[kj] * val_cache[v_start + d];
                }
            }
            out[o_start + d] = sum;
        }
    }

    // Qwen3.5 MHSA: gate from second half of fused `attn_q` or separate `attn_gate` matmul
    if let Some(ref gate_sl) = q_fused_gate {
        for i in 0..actual_q_dim.min(gate_sl.len()) {
            out[i] *= 1.0 / (1.0 + (-gate_sl[i]).exp());
        }
    } else if let Some(ref gate_weight) = w.attn_gate {
        let gate_dim = gate_weight.shape.last().copied().unwrap_or(0);
        if gate_dim == actual_q_dim {
            let gate_out = matmul::matmul(hidden, gate_weight, d, gate_dim)?;
            for i in 0..actual_q_dim.min(gate_out.len()) {
                out[i] *= 1.0 / (1.0 + (-gate_out[i]).exp());
            }
        }
    }

    Ok(out)
}
