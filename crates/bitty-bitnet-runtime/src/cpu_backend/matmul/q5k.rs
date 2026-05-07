use crate::cpu_backend::dequant::{Q5KBlock, QK_K};
use crate::cpu_backend::matmul::Result;
use rayon::prelude::*;

pub fn matmul(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let bs = Q5KBlock::BLOCK_SIZE;
    let total_elements = in_dim.checked_mul(out_dim).unwrap_or(0);
    if total_elements == 0 {
        return Ok(output);
    }
    let blocks_per_row = (in_dim / QK_K).max(1);

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, out)| {
            let mut sum = 0f32;
            let mut buf = [0f32; QK_K];
            for bi in 0..blocks_per_row {
                let blk = j * blocks_per_row + bi;
                let off = blk * bs;
                if off + bs > data.len() {
                    break;
                }
                let block = Q5KBlock::new(&data[off..off + bs]);
                block.dequantize_into(&mut buf);
                let x_start = bi * QK_K;
                let n = QK_K.min(in_dim - x_start);
                for k in 0..n {
                    sum += input[x_start + k] * buf[k];
                }
            }
            *out = sum;
        });

    Ok(output)
}
