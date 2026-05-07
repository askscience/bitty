use crate::cpu_backend::matmul::Result;

pub fn matmul_f32(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    let weight: &[f32] = bytemuck::cast_slice(data);
    for j in 0..out_dim {
        let row = &weight[j * in_dim..(j + 1) * in_dim];
        output[j] = dot(input, row);
    }
    Ok(output)
}

pub fn matmul_f16(input: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
    let mut output = vec![0f32; out_dim];
    for j in 0..out_dim {
        let mut sum = 0f32;
        let row_start = j * in_dim * 2;
        for i in 0..in_dim {
            let h = u16::from_le_bytes([data[row_start + i * 2], data[row_start + i * 2 + 1]]);
            sum += input[i] * half::f16::from_bits(h).to_f32();
        }
        output[j] = sum;
    }
    Ok(output)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
