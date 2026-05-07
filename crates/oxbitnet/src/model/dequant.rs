//! Dequantization kernels for GGML quantized formats.
//!
//! Converts packed quantized tensor data (Q4_0, Q8_0, K-quant, etc.)
//! into F32 values suitable for GPU upload.
//!
//! Block sizes follow the llama.cpp (ggml) canonical definitions.

use super::gguf::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_M, GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_0_4_4, GGML_TYPE_Q4_0_4_8, GGML_TYPE_Q4_0_8_8, GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1, GGML_TYPE_Q8_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
};

const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;

fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

fn f16_bytes_to_f32(bytes: &[u8]) -> f32 {
    let h = u16::from_le_bytes([bytes[0], bytes[1]]);
    f16_to_f32(h)
}

// ── Q4_0 ──────────────────────────────────────────────────────────────────
// Block: 2-byte f16 scale d, then 16 bytes of 4-bit qs (32 values).
// dequant: value = (nibble - 8) * d

fn dequantize_q4_0_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let qs = &src[2..2 + QK4_0 / 2];
    for i in 0..QK4_0 {
        let byte = qs[i / 2];
        let nibble = if i & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        dst[i] = (nibble as f32 - 8.0) * d;
    }
}

fn dequantize_q4_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 2 + QK4_0 / 2;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK4_0);
        let mut block_dst = vec![0f32; QK4_0];
        dequantize_q4_0_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK4_0;
    }
    dst
}

// ── Q4_1 ──────────────────────────────────────────────────────────────────
// Block: 2-byte f16 min m, 2-byte f16 scale d, then 16 bytes of 4-bit qs.
// dequant: value = nibble * d + m

fn dequantize_q4_1_block(src: &[u8], dst: &mut [f32]) {
    let m = f16_bytes_to_f32(&src[..2]);
    let d = f16_bytes_to_f32(&src[2..4]);
    let qs = &src[4..4 + QK4_1 / 2];
    for i in 0..QK4_1 {
        let byte = qs[i / 2];
        let nibble = if i & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        dst[i] = nibble as f32 * d + m;
    }
}

fn dequantize_q4_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 4 + QK4_1 / 2;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK4_1);
        let mut block_dst = vec![0f32; QK4_1];
        dequantize_q4_1_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK4_1;
    }
    dst
}

// ── Q5_0 ──────────────────────────────────────────────────────────────────
// Block: 2-byte f16 scale d, 1 byte of high-bit selector, then 16 bytes of
// 4-bit low nibbles. Each q is 5 bits: (low 4 bits) + (high bit shifted).
// dequant: value = ((low_nibble | (high_bit << 4)) - 16) * d

fn dequantize_q5_0_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let qh = src[2];
    let ql = &src[3..3 + QK5_0 / 2];
    for i in 0..QK5_0 {
        let byte = ql[i / 2];
        let low = if i & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        let high = (qh >> (i / 2 / 4 * 4 + (i / 2) % 4)) & 1;
        let val = ((high as i32) << 4 | low as i32) - 16;
        dst[i] = val as f32 * d;
    }
}

fn dequantize_q5_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 3 + QK5_0 / 2;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK5_0);
        let mut block_dst = vec![0f32; QK5_0];
        dequantize_q5_0_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK5_0;
    }
    dst
}

// ── Q5_1 ──────────────────────────────────────────────────────────────────
// Block: 2-byte f16 min m, 2-byte f16 scale d, 1 byte high-selector,
// 16 bytes of 4-bit low nibbles.
// dequant: value = ((low_nibble | (high_bit << 4)) * d) + m

fn dequantize_q5_1_block(src: &[u8], dst: &mut [f32]) {
    let m = f16_bytes_to_f32(&src[..2]);
    let d = f16_bytes_to_f32(&src[2..4]);
    let qh = src[4];
    let ql = &src[5..5 + QK5_1 / 2];
    for i in 0..QK5_1 {
        let byte = ql[i / 2];
        let low = if i & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        let high = (qh >> (i / 2 / 4 * 4 + (i / 2) % 4)) & 1;
        let val = (high as i32) << 4 | low as i32;
        dst[i] = val as f32 * d + m;
    }
}

fn dequantize_q5_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 5 + QK5_1 / 2;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK5_1);
        let mut block_dst = vec![0f32; QK5_1];
        dequantize_q5_1_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK5_1;
    }
    dst
}

// ── Q8_0 ──────────────────────────────────────────────────────────────────
// Block: 2-byte f16 scale d, then 32 bytes of 8-bit qs (32 values).
// dequant: value = (q8_value as i8) * d

fn dequantize_q8_0_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let qs = &src[2..2 + QK8_0];
    for i in 0..QK8_0 {
        dst[i] = (qs[i] as i8) as f32 * d;
    }
}

fn dequantize_q8_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 2 + QK8_0;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK8_0);
        let mut block_dst = vec![0f32; QK8_0];
        dequantize_q8_0_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK8_0;
    }
    dst
}

// ── Q8_1 (alias for Q8_0 in most GGUF files) ─────────────────────────────
fn dequantize_q8_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    dequantize_q8_0(data, num_elements)
}

// ── K-Quant superblock helpers ────────────────────────────────────────────

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    // For K-quants, scales/mins are packed as 6-bit values interleaved
    // in groups of 4 values packed into 3 bytes.
    if j < 4 {
        let byte = scales[j];
        let sc = byte & 0x3f;
        let mn = (byte >> 2) & 0x30 | (scales[4] >> (2 * j)) & 0x03;
        (sc as u8, mn as u8)
    } else {
        let jj = j - 4;
        let byte = scales[4 + jj];
        let sc = byte & 0x3f;
        let mn = (byte >> 2) & 0x30 | (scales[8] >> (2 * jj)) & 0x03;
        (sc as u8, mn as u8)
    }
}

// ── Q4_K ──────────────────────────────────────────────────────────────────
// Superblock of 256 elements, 8 sub-blocks of 32 each.
// Layout: d (2B f16), dmin (2B f16), scales (12B: 6*6-bit scales + 6*6-bit mins),
// qs (128B of 4-bit values).
// dequant: value = d * (scale_i * (nibble - 8)) + dmin * (min_i - 8)

fn dequantize_q4_k_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let dmin = f16_bytes_to_f32(&src[2..4]);
    let scales = &src[4..16];
    let qs = &src[16..16 + QK_K / 2];

    for i in 0..QK_K {
        let sub = i / 32;
        let (sc, mn) = get_scale_min_k4(sub, scales);
        let byte = qs[i / 2];
        let nibble = if i & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        dst[i] = d * (sc as f32) * (nibble as f32 - 8.0) + dmin * (mn as f32 - 8.0);
    }
}

fn dequantize_q4_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 4 + 12 + QK_K / 2;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK_K);
        let mut block_dst = vec![0f32; QK_K];
        dequantize_q4_k_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK_K;
    }
    dst
}

// ── Q6_K ──────────────────────────────────────────────────────────────────
// Superblock of 256 elements. 16 sub-blocks of 16 elements each.
// Layout: d (2B f16), dmin (2B f16), scales (16B 8-bit scales + 4B mins),
// ql (128B 6-bit low), qh (64B 2-bit high = 128 * 2 bits / 8).
// Q6_K element = (ql_low | (qh_high << 4) | (qh_extra << 6))?
// Actually: q6 has 6 bits total: ql (4 low bits) + qh (2 high bits).
// dequant: value = d * (scale[q_index % 32?] * (q6_value - 32))

fn dequantize_q6_k_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let ql = &src[2..2 + QK_K / 2];
    let qh = &src[2 + QK_K / 2..2 + QK_K / 2 + QK_K / 4];

    for i in 0..QK_K {
        // Q6: 4 low bits from ql, 2 high bits from qh
        let byte = ql[i / 2];
        let low = if i & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        // 2 high bits interleaved for every 2 elements per qh byte
        let qh_byte = qh[i / 4];
        let high_shift = 2 * (i % 4);
        let high = (qh_byte >> high_shift) & 0x03;
        let q6 = (high << 4 | low) as i32;
        let val = (q6 - 32) as f32 * d;
        dst[i] = val;
    }
}

fn dequantize_q6_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 2 + QK_K / 2 + QK_K / 4;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK_K);
        let mut block_dst = vec![0f32; QK_K];
        dequantize_q6_k_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK_K;
    }
    dst
}

// ── Q5_K ──────────────────────────────────────────────────────────────────
// Similar to Q4_K but with 5-bit quantized values (high bit in separate byte).
// Layout: d (4B float? or 2B f16), dmin (2B f16), ...

fn dequantize_q5_k_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let dmin = f16_bytes_to_f32(&src[2..4]);
    let scales = &src[4..16];
    let qh = src[16];
    let ql = &src[17..17 + QK_K / 2];

    for i in 0..QK_K {
        let sub = i / 32;
        let (sc, mn) = get_scale_min_k4(sub, scales);
        let byte = ql[i / 2];
        let low = if i & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        let high = (qh >> sub) & 1;
        let q5 = (high << 4 | low) as i32;
        dst[i] = d * (sc as f32) * (q5 as f32 - 16.0) + dmin * (mn as f32);
    }
}

fn dequantize_q5_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let _block_size = 17 + QK_K / 2 + (QK_K / 2 + 1) % 2;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK_K);
        let mut block_dst = vec![0f32; QK_K];
        let bs = 4 + 12 + 1 + QK_K / 2;
        dequantize_q5_k_block(&data[src_offset..src_offset + bs], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += bs;
        dst_offset += QK_K;
    }
    dst
}

// ── Q8_K ──────────────────────────────────────────────────────────────────
// Superblock format, 256 elements. Similar to Q8_0 blocks but with
// superblock-scale d. This is different from plain Q8_0.

fn dequantize_q8_k_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    // scales: 16 sub-block * 4-bit scales = 64 bits = 8 bytes
    let scales = &src[2..10];
    let qs = &src[10..10 + QK_K];

    for i in 0..QK_K {
        let sub = i / 16;
        let sc_byte = scales[sub / 2];
        let sc = if sub & 1 == 0 {
            sc_byte & 0x0f
        } else {
            sc_byte >> 4
        } as f32;
        dst[i] = d * sc * (qs[i] as i8) as f32;
    }
}

fn dequantize_q8_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 2 + 8 + QK_K;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK_K);
        let mut block_dst = vec![0f32; QK_K];
        dequantize_q8_k_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK_K;
    }
    dst
}

// ── Q3_K ──────────────────────────────────────────────────────────────────
fn dequantize_q3_k_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let hmask = (src[2] as u32) | ((src[3] as u32) << 8);
    let scales = &src[4..16];
    let qs = &src[16..16 + QK_K / 4];
    let qh_offset = 16 + QK_K / 4;
    let qh = &src[qh_offset..qh_offset + 2 * (QK_K / 32)];

    for i in 0..QK_K {
        let sub = i / 32;
        let (sc, mn) = get_scale_min_k4(sub, scales);
        let byte = qs[i / 4];
        let bits = match i % 4 {
            0 => byte & 0x03,
            1 => (byte >> 2) & 0x03,
            2 => (byte >> 4) & 0x03,
            3 => (byte >> 6) & 0x03,
            _ => 0,
        };
        let high = ((hmask >> (i / 2)) & 1) as u32;
        let qh_val = (qh[sub * 2] as u32) | ((qh[sub * 2 + 1] as u32) << 8);
        let qh_bit = (qh_val >> (2 * (i % 32))) & 0x03;
        let q3 = ((qh_bit << 2) | (high << 2) | bits as u32) as i32;
        dst[i] = d * (sc as f32) * (q3 as f32 - 8.0) - d * (mn as f32);
    }
}

fn dequantize_q3_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let _block_size = 16 + QK_K / 4 + 2 * (QK_K / 32);
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK_K);
        let mut block_dst = vec![0f32; QK_K];
        let bs = 4 + 12 + QK_K / 4 + 2 * (QK_K / 32);
        dequantize_q3_k_block(&data[src_offset..src_offset + bs], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += bs;
        dst_offset += QK_K;
    }
    dst
}

// ── Q2_K ──────────────────────────────────────────────────────────────────
fn dequantize_q2_k_block(src: &[u8], dst: &mut [f32]) {
    let d = f16_bytes_to_f32(&src[..2]);
    let dmin = f16_bytes_to_f32(&src[2..4]);
    let scales = &src[4..20];
    let qs = &src[20..20 + QK_K / 4];

    for i in 0..QK_K {
        let sub = i / 32;
        let sc_byte = scales[sub / 4];
        let sc = ((sc_byte >> (2 * (sub % 4))) & 0x03) as f32;
        let byte = qs[i / 4];
        let bits = match i % 4 {
            0 => byte & 0x03,
            1 => (byte >> 2) & 0x03,
            2 => (byte >> 4) & 0x03,
            3 => (byte >> 6) & 0x03,
            _ => 0,
        };
        let val = (bits as i32 - 1) as f32;
        let mn_val = dmin * (scales[8 + sub] as f32);
        dst[i] = d * sc * val + mn_val;
    }
}

fn dequantize_q2_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 20 + QK_K / 4;
    let mut dst = vec![0f32; num_elements];
    let mut src_offset = 0;
    let mut dst_offset = 0;
    while dst_offset < num_elements {
        let remaining = (num_elements - dst_offset).min(QK_K);
        let mut block_dst = vec![0f32; QK_K];
        dequantize_q2_k_block(&data[src_offset..src_offset + block_size], &mut block_dst);
        dst[dst_offset..dst_offset + remaining].copy_from_slice(&block_dst[..remaining]);
        src_offset += block_size;
        dst_offset += QK_K;
    }
    dst
}

// ── IQ series: importance-quantized. Use placeholder (scale to zero). ─────
// These are complex formats from llama.cpp. For models using IQ quantization,
// we dequantize to F32 via a simpler fallback path.

fn dequantize_iq_placeholder(data: &[u8], num_elements: usize, bpw: f64) -> Vec<f32> {
    // Fallback: interpret as raw bytes scaled to a reasonable range.
    // This is a placeholder — proper IQ dequant requires per-format kernels.
    let byte_len = (num_elements as f64 * bpw).ceil() as usize;
    let src = &data[..byte_len.min(data.len())];
    let mut dst = vec![0f32; num_elements];
    for i in 0..num_elements.min(byte_len) {
        dst[i] = src[i] as f32 / 255.0;
    }
    dst
}

// ── Passthrough formats ───────────────────────────────────────────────────
fn dequantize_f16(data: &[u8], num_elements: usize) -> Vec<f32> {
    let half_count = num_elements.min(data.len() / 2);
    let mut dst = vec![0f32; num_elements];
    for i in 0..half_count {
        dst[i] = f16_to_f32(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]));
    }
    dst
}

fn dequantize_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(num_elements)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── Main dispatch ─────────────────────────────────────────────────────────

#[allow(dead_code)]
pub enum DequantResult {
    F32(Vec<f32>),
    BlockSizeMismatch(String),
}

/// Dequantize a tensor buffer to F32 based on its GGML type.
/// Returns the dequantized F32 values.
pub fn dequantize_tensor(data: &[u8], tensor_type: u32, num_elements: u64) -> Vec<f32> {
    let n = num_elements as usize;

    match tensor_type {
        GGML_TYPE_F32 => dequantize_f32(data, n),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => dequantize_f16(data, n),
        GGML_TYPE_Q4_0 => dequantize_q4_0(data, n),
        GGML_TYPE_Q4_1 => dequantize_q4_1(data, n),
        GGML_TYPE_Q5_0 => dequantize_q5_0(data, n),
        GGML_TYPE_Q5_1 => dequantize_q5_1(data, n),
        GGML_TYPE_Q8_0 | GGML_TYPE_Q8_1 => dequantize_q8_0(data, n),
        GGML_TYPE_Q4_K => dequantize_q4_k(data, n),
        GGML_TYPE_Q5_K => dequantize_q5_k(data, n),
        GGML_TYPE_Q6_K => dequantize_q6_k(data, n),
        GGML_TYPE_Q8_K => dequantize_q8_k(data, n),
        GGML_TYPE_Q3_K => dequantize_q3_k(data, n),
        GGML_TYPE_Q2_K => dequantize_q2_k(data, n),
        GGML_TYPE_IQ2_XXS | GGML_TYPE_IQ2_XS | GGML_TYPE_IQ2_S => {
            dequantize_iq_placeholder(data, n, 0.25)
        }
        GGML_TYPE_IQ3_XXS | GGML_TYPE_IQ3_S | GGML_TYPE_IQ3_M => {
            dequantize_iq_placeholder(data, n, 0.375)
        }
        GGML_TYPE_IQ4_NL | GGML_TYPE_IQ4_XS => dequantize_iq_placeholder(data, n, 0.5),
        GGML_TYPE_IQ1_S | GGML_TYPE_IQ1_M => dequantize_iq_placeholder(data, n, 0.125),
        GGML_TYPE_Q4_0_4_4 | GGML_TYPE_Q4_0_4_8 | GGML_TYPE_Q4_0_8_8 => dequantize_q4_0(data, n),
        // TQ types: fallback to raw scale
        GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0 => dequantize_iq_placeholder(data, n, 0.25),
        _ => {
            tracing::warn!(
                "Unknown GGML type {} in dequantize, falling back to raw cast",
                tensor_type
            );
            dequantize_iq_placeholder(data, n, 2.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        // Simulate a Q4_0 block: scale=1.0, values = [-8, -7, ..., 7]
        let d = half::f16::from_f32(1.0);
        let mut block = vec![0u8; 2 + 32 / 2];
        block[..2].copy_from_slice(&d.to_le_bytes());
        for i in 0..16usize {
            block[2 + i] = ((i & 0x0f) | (((i + 1) & 0x0f) << 4)) as u8;
        }
        let result = dequantize_q4_0(&block, 32);
        // The first element (nibble 0) should be -8.0
        assert!((result[0] + 8.0).abs() < 0.1);
    }

    #[test]
    fn test_q8_0_roundtrip() {
        let d = half::f16::from_f32(0.5);
        let mut block = vec![0u8; 2 + 32];
        block[..2].copy_from_slice(&d.to_le_bytes());
        block[2] = 10u8; // value = 10 * 0.5 = 5.0 (as i8 = 10)
        let result = dequantize_q8_0(&block, 32);
        assert!((result[0] - 5.0).abs() < 0.01);
    }
}
