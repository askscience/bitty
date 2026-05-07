//! Common Transformer and SSM operations for CPU inference.

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
    if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }
}

/// Softmax over a slice (in-place).
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|&v| (v - max).exp()).sum();
    let inv = 1.0 / sum;
    for v in x.iter_mut() { *v = (*v - max).exp() * inv; }
}

/// Apply RoPE (Rotary Position Embedding) to Q and K in-place.
pub fn rope_apply(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, num_heads: usize, num_kv_heads: usize, rope_theta: f32) {
    if head_dim == 0 { return; }
    let half_dim = head_dim / 2;
    let groups = num_heads.max(1) / num_kv_heads.max(1);
    let max_h = (q.len() / head_dim.max(1)).min(num_heads);
    let max_kv_h = (k.len() / head_dim.max(1)).min(num_kv_heads);

    for h in 0..max_h {
        let q_off = h * head_dim;
        let k_off = if groups > 0 { (h / groups.max(1)) * head_dim } else { 0 };
        if q_off + half_dim * 2 > q.len() || k_off + half_dim * 2 > k.len() { continue; }
        for i in 0..half_dim {
            let theta = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
            let cos = (pos as f32 * theta).cos();
            let sin = (pos as f32 * theta).sin();
            let q0 = q[q_off + i];
            let q1 = q[q_off + i + half_dim];
            q[q_off + i] = q0 * cos - q1 * sin;
            q[q_off + i + half_dim] = q0 * sin + q1 * cos;
            if k_off + i + half_dim < k.len() {
                let k0 = k[k_off + i];
                let k1 = k[k_off + i + half_dim];
                k[k_off + i] = k0 * cos - k1 * sin;
                k[k_off + i + half_dim] = k0 * sin + k1 * cos;
            }
        }
    }
}
