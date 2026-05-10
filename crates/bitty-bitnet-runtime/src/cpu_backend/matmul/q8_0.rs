use crate::cpu_backend::dequant::{Q8_0Block, QK8_0};
use crate::cpu_backend::matmul::Result;
use rayon::prelude::*;

/// Q8_0 matmul. Weight is stored as [out_dim, in_dim] row-major, matching the
/// layout convention used by Q4_K/Q4_0/Q5_K/Q6_K in this codebase.
pub fn matmul(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let bs = Q8_0Block::BLOCK_SIZE;
    let total_elements = in_dim.checked_mul(out_dim).unwrap_or(0);
    if total_elements == 0 {
        return Ok(output);
    }
    let blocks_per_row = (in_dim / QK8_0).max(1);

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, out)| {
            let mut sum = 0f32;
            let row_base = j * blocks_per_row * bs;
            for bi in 0..blocks_per_row {
                let off = row_base + bi * bs;
                if off + bs > data.len() {
                    break;
                }
                let block = Q8_0Block::new(&data[off..off + bs]);
                let x_start = bi * QK8_0;
                let n = QK8_0.min(in_dim - x_start);
                for k in 0..n {
                    sum += input[x_start + k] * block.get(k);
                }
            }
            *out = sum;
        });

    Ok(output)
}
