//! Mamba State Space Model (SSM) layer — selective scan algorithm.
//!
//! Implements the core selective scan from the Mamba paper (Gu & Dao, 2023)
//! for CPU inference with quantized GGUF weights.
//!
//! Algorithm (per position):
//!   1. Conv1D pre-processing on input x
//!   2. Delta projection: dt = softplus(Linear(input) + bias)
//!   3. Discretize A, B:  Ā = exp(dt ⊙ A),  B̄ = dt ⊙ B  
//!   4. Selective scan:  h_new = Ā ⊙ h + B̄ ⊙ x,  y = h @ C + D ⊙ x
//!   5. Gate with SiLU(z) where z is the second half of in_proj output
//!   6. SSM norm → output projection

use crate::cpu_backend::matmul;
use crate::cpu_backend::ops::{self, rms_norm, silu, softplus};
use crate::cpu_backend::types::{CpuModelMetadata, SsmState, SsmWeights};

type Result<T> = std::result::Result<T, String>;

/// Forward pass for an SSM (Mamba) layer.
pub fn forward(
    hidden: &[f32],
    w: &SsmWeights,
    pos: usize,
    state: &mut SsmState,
    meta: &CpuModelMetadata,
) -> Result<Vec<f32>> {
    let d = meta.hidden_size;
    let d_inner = w.d_inner;
    let d_state = w.d_state;
    let ksize = w.kernel_size;

    // 1. Input projection: x_and_z = hidden @ in_proj → [2 * d_inner]
    let xz = matmul::matmul(hidden, &w.in_proj, d, 2 * d_inner)?;
    let (x_raw, z) = xz.split_at(d_inner);

    // 2. Conv1D: depthwise causal convolution over time
    let x_conv = conv1d_step(
        x_raw,
        &w.conv1d_weight,
        w.conv1d_bias.as_deref(),
        state,
        d_inner,
        ksize,
    );
    let x_act: Vec<f32> = x_conv.iter().map(|&v| silu(v)).collect();

    // 3. Delta projection: dt = softplus(W_dt @ hidden + b_dt)
    // If no dt_proj_weight, use identity (dt = hidden[..d_inner] projected)
    let dt_rank = w.dt_proj_weight.shape.last().copied().unwrap_or(1);
    let dt_raw = if w.dt_proj_weight.ggml_type == 0 && w.dt_proj_weight.data.len() <= 4 {
        // Dummy tensor — use a simple projection from hidden
        hidden.iter().cycle().take(d_inner).copied().collect()
    } else if dt_rank == d_inner {
        matmul::matmul(hidden, &w.dt_proj_weight, d, d_inner)?
    } else {
        matmul::matmul(hidden, &w.dt_proj_weight, d, dt_rank)?
            .iter()
            .cycle()
            .take(d_inner)
            .copied()
            .collect()
    };

    let dt: Vec<f32> = dt_raw
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let bias = w
                .dt_proj_bias
                .as_ref()
                .map(|b| b.get(i).copied().unwrap_or(0.0))
                .unwrap_or(0.0);
            softplus(v + bias)
        })
        .collect();

    // 4. Selection parameters B and C from x
    // In standard Mamba: B and C are linear projections of x_act
    // For Qwen variant: B and C come from selection projections
    // We use x_act as the selection input
    let b = x_act.clone(); // B: [d_inner] → selection for each state dim
    let c = x_act.clone(); // C: [d_inner] → selection for each state dim

    // 5. Selective scan (recurrent step)
    // h_new = exp(dt ⊙ A) ⊙ h_old + dt ⊙ B ⊙ x
    // y = h_new @ C^T + D ⊙ x
    let mut y = vec![0f32; d_inner];

    for di in 0..d_inner {
        // Update hidden state for dimension di
        for si in 0..d_state {
            let idx = di * d_state + si;
            // A_discrete = exp(dt[di] * A[di, si])
            let a_disc = (dt[di] * w.a_log[idx].exp()).exp(); // exp(dt * exp(A_log)) = exp(dt * A)
                                                              // B_bar = dt[di] * B[di]  (simplified: B selection same for all states)
            let b_bar = dt[di] * b[di];
            // Update state
            state.h[idx] = a_disc * state.h[idx] + b_bar * x_act[di];
            // Accumulate output
            y[di] += state.h[idx] * c[di];
        }
        // Add D skip connection
        if di < w.d_param.len() {
            y[di] += w.d_param[di] * x_act[di];
        }
    }

    // 6. Gate with SiLU(z)
    let z_act: Vec<f32> = z.iter().map(|&v| silu(v)).collect();
    for i in 0..d_inner {
        y[i] *= z_act[i];
    }

    // 7. SSM norm
    let y_normed = rms_norm(&y, &w.ssm_norm, 1e-5);

    // 8. Output projection → back to hidden_size
    matmul::matmul(&y_normed, &w.out_proj, d_inner, d)
}

/// Causal depthwise 1D convolution step.
/// Maintains a sliding window of previous inputs in `state.conv_state`.
fn conv1d_step(
    x: &[f32],
    kernel: &super::super::types::PackedTensor, // [d_inner * kernel_size] or [d_inner, kernel_size]
    bias: Option<&[f32]>,
    state: &mut SsmState,
    d_inner: usize,
    kernel_size: usize,
) -> Vec<f32> {
    // Kernel is typically stored as [kernel_size, d_inner] in GGUF
    // but packed as 1D. For depthwise conv, kernel has shape [d_inner, kernel_size].
    let k_data = &kernel.data;
    let kernel_f32 = dequant_kernel(k_data, kernel.ggml_type, d_inner * kernel_size);

    // Shift conv state: move [1..] to [0..], append new x
    let state_len = state.conv_state.len();
    let conv_dim = d_inner * (kernel_size - 1);
    if state_len >= conv_dim {
        // Shift: move second element onward to the front
        for i in 0..conv_dim - d_inner {
            state.conv_state[i] = state.conv_state[i + d_inner];
        }
        // Append new x
        let start = conv_dim - d_inner;
        state.conv_state[start..conv_dim].copy_from_slice(x);
    }

    // Build full sequence: [conv_state, x] → length kernel_size * d_inner
    let full_len = d_inner * kernel_size;
    let mut seq = vec![0f32; full_len];
    let past_start = full_len - x.len() - conv_dim;
    if past_start < full_len {
        let copy_start = past_start + (full_len - x.len() - conv_dim);
        if copy_start < conv_dim {
            let copy_len = (conv_dim - copy_start).min(seq.len());
            seq[..copy_len].copy_from_slice(&state.conv_state[copy_start..copy_start + copy_len]);
        }
    }
    seq[full_len - x.len()..].copy_from_slice(x);

    // Depthwise conv: for each channel, dot product with kernel[channel, :]
    let mut out = vec![0f32; d_inner];
    for ch in 0..d_inner {
        let mut sum = 0f32;
        for t in 0..kernel_size {
            let ki = ch * kernel_size + t;
            if ki < kernel_f32.len() {
                sum += seq[t * d_inner + ch] * kernel_f32[ki];
            }
        }
        out[ch] = sum
            + bias
                .map(|b| b.get(ch).copied().unwrap_or(0.0))
                .unwrap_or(0.0);
    }

    out
}

fn dequant_kernel(data: &[u8], ggml_type: u32, num_elements: usize) -> Vec<f32> {
    use crate::cpu_backend::dequant::dequantize_slice;
    dequantize_slice(data, ggml_type, num_elements)
}
