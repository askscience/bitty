use crate::cpu_backend::dequant::Q8_0Block;
use crate::cpu_backend::matmul::Result;

/// Q8_0 matmul. Weight is stored as [in_dim, out_dim] row-major.
/// Blocks are 32 consecutive flat-position elements.
pub fn matmul(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let be = 32usize;
    let bs = Q8_0Block::BLOCK_SIZE;

    let total_elements = in_dim * out_dim;
    let total_blocks = total_elements.div_ceil(be);

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

            let j = flat_idx / in_dim; // output index
            let i = flat_idx % in_dim; // input index

            output[j] += input[i] * block.get(local);
        }
    }

    Ok(output)
}
