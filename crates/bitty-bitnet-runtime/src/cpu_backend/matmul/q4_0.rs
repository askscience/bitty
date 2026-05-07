use crate::cpu_backend::dequant::{Q4_0Block, QK4_0};
use crate::cpu_backend::matmul::Result;
use rayon::prelude::*;

pub fn matmul(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let bs = Q4_0Block::BLOCK_SIZE;
    let total_elements = in_dim.checked_mul(out_dim).unwrap_or(0);
    if total_elements == 0 {
        return Ok(output);
    }
    let blocks_per_row = (in_dim / QK4_0).max(1);

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, out)| {
            let mut sum = 0f32;
            let mut buf = [0f32; QK4_0];
            for bi in 0..blocks_per_row {
                let blk = j * blocks_per_row + bi;
                let off = blk * bs;
                if off + bs > data.len() {
                    break;
                }
                let block = Q4_0Block::new(&data[off..off + bs]);
                block.dequantize_into(&mut buf);
                let x_start = bi * QK4_0;
                let n = QK4_0.min(in_dim - x_start);
                for k in 0..n {
                    sum += input[x_start + k] * buf[k];
                }
            }
            *out = sum;
        });

    Ok(output)
}
