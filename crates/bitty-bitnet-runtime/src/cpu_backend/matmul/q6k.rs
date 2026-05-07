use crate::cpu_backend::dequant::{Q6KBlock, QK_K};
use crate::cpu_backend::matmul::Result;

pub fn matmul(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let bs = Q6KBlock::BLOCK_SIZE;
    let total_elements = in_dim.checked_mul(out_dim).unwrap_or(0);
    if total_elements == 0 {
        return Ok(output);
    }
    let mut buf = [0f32; QK_K];

    let total_blocks = total_elements.div_ceil(QK_K);
    for blk in 0..total_blocks {
        let off = blk * bs;
        if off + bs > data.len() {
            break;
        }
        let block = Q6KBlock::new(&data[off..off + bs]);
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
