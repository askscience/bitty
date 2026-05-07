use crate::cpu_backend::dequant::Q8_0Block;
use crate::cpu_backend::matmul::Result;
use rayon::prelude::*;

/// Q8_0 matmul. Weight is stored as [in_dim, out_dim] row-major.
/// Blocks are 32 consecutive flat-position elements.
pub fn matmul(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let be = 32usize;
    let bs = Q8_0Block::BLOCK_SIZE;

    let total_elements = in_dim * out_dim;
    let total_blocks = total_elements.div_ceil(be);

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, out)| {
            let mut sum = 0f32;
            for block_idx in 0..total_blocks {
                let block_off = block_idx * bs;
                if block_off + bs > data.len() {
                    break;
                }
                let block = Q8_0Block::new(&data[block_off..block_off + bs]);
                let flat_base = block_idx * be;
                for local in 0..be {
                    let flat_idx = flat_base + local;
                    if flat_idx >= total_elements {
                        break;
                    }
                    let wj = flat_idx / in_dim;
                    if wj != j {
                        continue;
                    }
                    let i = flat_idx % in_dim;
                    sum += input[i] * block.get(local);
                }
            }
            *out = sum;
        });

    Ok(output)
}
