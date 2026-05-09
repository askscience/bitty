//! On-the-fly GGML quantized weight dequantization to f32.
//!
//! Handles K-quant formats (Q4_K, Q5_K, Q6_K, Q8_K) and fixed-size
//! quant types (Q4_0, Q8_0). Based on llama.cpp's ggml-quants.c block layouts.

use bitty_model::gguf;

const QK_K: usize = 256;
const QK4_0: usize = 32;
const QK8_0: usize = 32;

pub fn dequantize_tensor(raw: &[u8], ggml_type: u32, expected_elements: usize) -> Vec<f32> {
    match ggml_type {
        gguf::GGML_TYPE_F32 => {
            let floats: &[f32] = bytemuck::cast_slice(raw);
            floats[..expected_elements.min(floats.len())].to_vec()
        }
        gguf::GGML_TYPE_F16 => {
            let halfs: &[half::f16] = bytemuck::cast_slice(raw);
            halfs.iter()
                .take(expected_elements)
                .map(|h| h.to_f32())
                .collect()
        }
        gguf::GGML_TYPE_Q8_0 => dequant_q8_0(raw, expected_elements),
        gguf::GGML_TYPE_Q4_0 => dequant_q4_0(raw, expected_elements),
        gguf::GGML_TYPE_Q4_K => dequant_q4_k(raw, expected_elements),
        gguf::GGML_TYPE_Q5_K => dequant_q5_k(raw, expected_elements),
        gguf::GGML_TYPE_Q6_K => dequant_q6_k(raw, expected_elements),
        gguf::GGML_TYPE_Q8_K => dequant_q8_k(raw, expected_elements),
        gguf::GGML_TYPE_Q2_K => dequant_q2_k(raw, expected_elements),
        gguf::GGML_TYPE_Q3_K => dequant_q3_k(raw, expected_elements),
        _ => vec![0.0f32; expected_elements],
    }
}

fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let block_size = QK8_0 + 4; // 32 elements + 2-byte scale (f16) + 2 padding
    for (i, block) in data.chunks(block_size).enumerate() {
        if block.len() < 4 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let start = i * QK8_0;
        for j in 0..QK8_0.min(block.len().saturating_sub(4)) {
            if start + j >= n { break; }
            out[start + j] = (block[4 + j] as i8) as f32 * d;
        }
    }
    out
}

fn dequant_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let block_size = QK4_0 / 2 + 2; // 16 bytes weights + 2-byte scale (f16)
    for (i, block) in data.chunks(block_size).enumerate() {
        if block.len() < 2 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let start = i * QK4_0;
        for j in 0..(QK4_0 / 2).min(block.len().saturating_sub(2)) {
            let byte = block[2 + j];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = (byte >> 4) as i8 - 8;
            if start + j * 2 < n { out[start + j * 2] = q0 as f32 * d; }
            if start + j * 2 + 1 < n { out[start + j * 2 + 1] = q1 as f32 * d; }
        }
    }
    out
}

fn dequant_q8_k(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let block_size = QK_K + 10; // 256 elements + d(f16) + 8 sub-block scales
    for (i, block) in data.chunks(block_size).enumerate() {
        if block.len() < 5 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let start = i * QK_K;
        for j in 0..QK_K.min(block.len().saturating_sub(10)) {
            if start + j >= n { break; }
            let scale_idx = j / 32;
            let scale = if 4 + scale_idx < block.len() - 2 {
                half::f16::from_le_bytes([block[4 + scale_idx * 2], block[5 + scale_idx * 2]]).to_f32()
            } else { 1.0f32 };
            out[start + j] = (block[10 + j] as i8) as f32 * d * scale;
        }
    }
    out
}

fn dequant_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let block_size = QK_K / 2 + 2 + 12; // 128 bytes weights + d+dmin(f16 each) + 12 scale bytes
    for (i, block) in data.chunks(block_size).enumerate() {
        if block.len() < 4 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let start = i * QK_K;
        let scales_start = 4;
        let weights_start = 4 + 12;

        for j in 0..(QK_K / 2).min(block.len().saturating_sub(weights_start)) {
            let byte = block[weights_start + j];
            let q0 = (byte & 0x0F) as i8;
            let q1 = (byte >> 4) as i8;
            let sc = if j < QK_K / 4 {
                block[scales_start + j / 16] & 0x3F
            } else {
                block[scales_start + 6 + (j - QK_K / 4) / 16] & 0x3F
            };
            let scale = dmin + d * sc as f32;
            if start + j * 2 < n { out[start + j * 2] = q0 as f32 * scale; }
            if start + j * 2 + 1 < n { out[start + j * 2 + 1] = q1 as f32 * scale; }
        }
    }
    out
}

fn dequant_q5_k(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let block_size = QK_K / 8 + QK_K / 2 + 2 + 12; // high bits + low bits + d+dmin + scales
    for (i, block) in data.chunks(block_size).enumerate() {
        if block.len() < 4 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let start = i * QK_K;
        let scales_start = 4;
        let high_bits_start = 4 + 12;
        let low_bits_start = high_bits_start + QK_K / 8;

        for j in 0..(QK_K / 2).min(block.len().saturating_sub(low_bits_start)) {
            let qh_byte = block[high_bits_start + j / 8];
            let qh = ((qh_byte >> (j % 8)) & 1) as u8;
            let ql_byte = block[low_bits_start + j];
            let q0 = ((ql_byte & 0x0F) | (qh << 4)) as i8 - 16;
            let q1 = ((ql_byte >> 4) | (qh << 4)) as i8 - 16;
            let sc_idx = j / 16;
            let sc = if sc_idx < 6 {
                block[scales_start + sc_idx] & 0x3F
            } else {
                block[scales_start + sc_idx - 6] >> 4
            };
            let scale = dmin + d * sc as f32;
            if start + j * 2 < n { out[start + j * 2] = q0 as f32 * scale; }
            if start + j * 2 + 1 < n { out[start + j * 2 + 1] = q1 as f32 * scale; }
        }
    }
    out
}

fn dequant_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let per_block = QK_K / 2 + QK_K / 4 + 2 + 16; // ql + qh + d(f16) + per-sub-block scales
    for (i, block) in data.chunks(per_block).enumerate() {
        if block.len() < 4 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let start = i * QK_K;
        let ql_start = 2 + 16;
        let qh_start = ql_start + QK_K / 2;

        for j in 0..(QK_K / 2).min(block.len().saturating_sub(qh_start)) {
            let ql_byte = block[ql_start + j];
            let qh_byte = block[qh_start + j / 2];
            let qh = if j % 2 == 0 { qh_byte & 0x0F } else { qh_byte >> 4 };
            let q = ((ql_byte & 0x0F) | (qh << 4)) as i8 - 32;
            let scale = (block[2 + j / 16] as i8) as f32;
            if start + j < n { out[start + j] = q as f32 * d * scale; }
        }
    }
    out
}

fn dequant_q2_k(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let block_size = QK_K / 4 + QK_K / 16 + 4 + 16; // quants + extra bits + d+dmin + scales
    for (i, block) in data.chunks(block_size).enumerate() {
        if block.len() < 4 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let start = i * QK_K;
        let scales_off = 4;
        let extra_off = 4 + 16;
        let quants_off = 4 + 16 + QK_K / 16;

        for j in 0..(QK_K / 4).min(block.len().saturating_sub(quants_off)) {
            let byte = block[quants_off + j];
            for k in 0..4 {
                let q = ((byte >> (k * 2)) & 0x03) as i8;
                let sc_idx = j / 4;
                let sc = if sc_idx < 16 {
                    block[scales_off + sc_idx] & 0x0F
                } else { 0 };
                let idx = start + j * 4 + k;
                if idx < n {
                    let q_val = q as f32 - if q > 1 { 4.0 } else { 0.0 };
                    out[idx] = (dmin as f32 + d as f32 * sc as f32) * q_val;
                }
            }
        }
    }
    out
}

fn dequant_q3_k(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let block_size = QK_K / 4 + QK_K / 8 + 4 + 12; // quants + extra bits + d+dmin + scales
    for (i, block) in data.chunks(block_size).enumerate() {
        if block.len() < 4 { break; }
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let start = i * QK_K;
        let scales_off = 4;
        let high_bits_off = 4 + 12;
        let quants_off = 4 + 12 + QK_K / 8;

        for j in 0..(QK_K / 4).min(block.len().saturating_sub(quants_off)) {
            let byte = block[quants_off + j];
            let qh_byte = block[high_bits_off + j / 4];
            for k in 0..4 {
                let q = ((byte >> (k * 2)) & 0x03) as i8;
                let qh = ((qh_byte >> (j % 4 * 2 + k)) & 0x01) as i8;
                let q = q | (qh << 2);
                let sc_idx = j / 8;
                let sc_shift = (j % 8 / 2) * 3;
                let sc = (block[scales_off + sc_idx] >> sc_shift) & 0x07;
                let idx = start + j * 4 + k;
                if idx < n {
                    out[idx] = (dmin + d * sc as f32) * (q as f32 - 3.0);
                }
            }
        }
    }
    out
}
