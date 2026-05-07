use crate::cpu_backend::dequant::{Q4KBlock, QK_K};
use crate::cpu_backend::matmul::Result;

/// Q4_K matrix-vector multiply: `output[j] = sum_i input[i] * W[j, i]`.
///
/// Weight memory layout is `[out_dim, in_dim]` row-major (fastest-changing is
/// `in_dim`, matching llama.cpp), so each 256-element Q4_K block covers 256
/// consecutive `i` values within a single output row.
pub fn matmul(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let bs = Q4KBlock::BLOCK_SIZE;
    let total_elements = in_dim.checked_mul(out_dim).unwrap_or(0);
    if total_elements == 0 {
        return Ok(output);
    }
    let mut buf = [0f32; QK_K];

    // Blocks always cover exactly QK_K=256 consecutive flat elements. For
    // standard GGUF tensors `in_dim` is a multiple of 256, so each block is
    // fully inside one output row. The general code below handles blocks that
    // straddle a row boundary too (no-op for typical shapes).
    let total_blocks = total_elements.div_ceil(QK_K);
    for blk in 0..total_blocks {
        let off = blk * bs;
        if off + bs > data.len() {
            break;
        }
        let block = Q4KBlock::new(&data[off..off + bs]);
        block.dequantize_into(&mut buf);

        let flat_base = blk * QK_K;
        let end = (flat_base + QK_K).min(total_elements);
        let mut flat = flat_base;
        while flat < end {
            let j = flat / in_dim;
            let i = flat % in_dim;
            let remain_row = in_dim - i;
            let remain_block = end - flat;
            let n = remain_row.min(remain_block);
            let row = &buf[flat - flat_base..flat - flat_base + n];
            let x_row = &input[i..i + n];
            let mut sum = 0f32;
            for k in 0..n {
                sum += x_row[k] * row[k];
            }
            output[j] += sum;
            flat += n;
        }
    }

    Ok(output)
}
