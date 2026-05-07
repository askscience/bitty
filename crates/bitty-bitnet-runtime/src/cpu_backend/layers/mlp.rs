//! Shared MLP (Feed-Forward Network) used by all layer types.

use crate::cpu_backend::matmul;
use crate::cpu_backend::ops::silu;
use crate::cpu_backend::types::MlpBlock;

type Result<T> = std::result::Result<T, String>;

pub fn forward(hidden: &[f32], mlp: &MlpBlock, hidden_size: usize, intermediate_size: usize) -> Result<Vec<f32>> {
    let up = matmul::matmul(hidden, &mlp.up_proj, hidden_size, intermediate_size)?;
    let gate = matmul::matmul(hidden, &mlp.gate_proj, hidden_size, intermediate_size)?;
    let gated: Vec<f32> = up.iter().zip(gate.iter())
        .map(|(&u, &g)| u * silu(g))
        .collect();
    matmul::matmul(&gated, &mlp.down_proj, intermediate_size, hidden_size)
}
