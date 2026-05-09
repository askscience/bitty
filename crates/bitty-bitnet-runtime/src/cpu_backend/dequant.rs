//! On-the-fly dequantization for GGML quantized formats during CPU inference.
//!
//! The block formats follow llama.cpp's reference implementation in
//! `ggml-quants.c`. The key-quant (`Q4_K`, `Q5_K`, `Q6_K`) layouts have some
//! non-obvious packing details; see the comments on each `*Block` struct.

use bitty_model::gguf::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0, GGML_TYPE_Q8_1,
};

pub const QK_K: usize = 256;
pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;

// ============================================================================
// Q4_K — 256-element superblocks, 4-bit quants.
// Layout: d(fp16, 2) | dmin(fp16, 2) | scales(12) | qs(128)   -> 144 bytes/block
// ============================================================================
pub struct Q4KBlock<'a> {
    pub d: f32,
    pub dmin: f32,
    pub scales: &'a [u8], // 12 bytes, 8 sub-blocks of packed (6-bit scale, 6-bit min)
    pub qs: &'a [u8],     // 128 bytes, 4-bit quants
}

impl<'a> Q4KBlock<'a> {
    pub const BLOCK_SIZE: usize = 4 + 12 + QK_K / 2; // 144

    pub fn new(data: &'a [u8]) -> Self {
        let d = f16_to_f32(&data[..2]);
        let dmin = f16_to_f32(&data[2..4]);
        Self {
            d,
            dmin,
            scales: &data[4..16],
            qs: &data[16..16 + QK_K / 2],
        }
    }

    /// Dequantize the full 256-element block into `out[..256]`.
    pub fn dequantize_into(&self, out: &mut [f32]) {
        // Reference (ggml-quants.c, dequantize_row_q4_K):
        //   for j in 0..QK_K step 64:
        //     (sc1, m1) = get_scale_min_k4(is+0, scales)
        //     (sc2, m2) = get_scale_min_k4(is+1, scales)
        //     d1 = d*sc1; m1 = dmin*m1; d2 = d*sc2; m2 = dmin*m2
        //     for l in 0..32: y[j+l]      = d1 * (q[l] & 0xF) - m1
        //     for l in 0..32: y[j+32+l]   = d2 * (q[l]  >> 4) - m2
        //     q += 32; is += 2
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut y_off = 0usize;
        while y_off < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, self.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, self.scales);
            let d1 = self.d * sc1 as f32;
            let md1 = self.dmin * m1 as f32;
            let d2 = self.d * sc2 as f32;
            let md2 = self.dmin * m2 as f32;
            for l in 0..32 {
                let b = self.qs[q_off + l];
                out[y_off + l] = d1 * (b & 0x0F) as f32 - md1;
                out[y_off + 32 + l] = d2 * (b >> 4) as f32 - md2;
            }
            y_off += 64;
            q_off += 32;
            is += 2;
        }
    }
}

// ============================================================================
// Q5_K — 256-element superblocks, 5-bit quants.
// Layout: d(2) | dmin(2) | scales(12) | qh(32) | qs(128)  -> 176 bytes/block
// ============================================================================
pub struct Q5KBlock<'a> {
    pub d: f32,
    pub dmin: f32,
    pub scales: &'a [u8], // 12
    pub qh: &'a [u8],     // 32, one bit per element (the high bit)
    pub qs: &'a [u8],     // 128, low-4 nibbles
}

impl<'a> Q5KBlock<'a> {
    pub const BLOCK_SIZE: usize = 4 + 12 + 32 + QK_K / 2; // 176

    pub fn new(data: &'a [u8]) -> Self {
        let d = f16_to_f32(&data[..2]);
        let dmin = f16_to_f32(&data[2..4]);
        Self {
            d,
            dmin,
            scales: &data[4..16],
            qh: &data[16..48],
            qs: &data[48..48 + QK_K / 2],
        }
    }

    pub fn dequantize_into(&self, out: &mut [f32]) {
        // Reference (ggml-quants.c, dequantize_row_q5_K):
        //   u1 = 1, u2 = 2; is = 0; ql = qs;
        //   for j in 0..QK_K step 64:
        //     (sc1, m1) = get_scale_min_k4(is+0, scales)
        //     (sc2, m2) = get_scale_min_k4(is+1, scales)
        //     d1 = d*sc1; md1 = dmin*m1; d2 = d*sc2; md2 = dmin*m2
        //     for l in 0..32:
        //       y[j+l]    = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - md1
        //       y[j+32+l] = d2 * ((ql[l] >>  4) + ((qh[l] & u2) != 0 ? 16 : 0)) - md2
        //     ql += 32; is += 2; u1 <<= 2; u2 <<= 2
        let mut is = 0usize;
        let mut ql_off = 0usize;
        let mut y_off = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        while y_off < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, self.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, self.scales);
            let d1 = self.d * sc1 as f32;
            let md1 = self.dmin * m1 as f32;
            let d2 = self.d * sc2 as f32;
            let md2 = self.dmin * m2 as f32;
            for l in 0..32 {
                let lo = self.qs[ql_off + l];
                let qhb = self.qh[l];
                let hi1 = if qhb & u1 != 0 { 16u8 } else { 0 };
                let hi2 = if qhb & u2 != 0 { 16u8 } else { 0 };
                let a = (lo & 0x0F) + hi1;
                let b = (lo >> 4) + hi2;
                out[y_off + l] = d1 * a as f32 - md1;
                out[y_off + 32 + l] = d2 * b as f32 - md2;
            }
            y_off += 64;
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ============================================================================
// Q6_K — 256-element superblocks, 6-bit quants.
// Layout: ql(128) | qh(64) | scales(16, int8) | d(fp16, 2)   -> 210 bytes/block
// NOTE: the fp16 `d` is at the END of the block, not the start.
// ============================================================================
pub struct Q6KBlock<'a> {
    pub d: f32,
    pub ql: &'a [u8],     // 128, low-4 bits
    pub qh: &'a [u8],     // 64,  high-2 bits (2 bits per element)
    pub scales: &'a [i8], // 16 signed 8-bit per-sub-block scales
}

impl<'a> Q6KBlock<'a> {
    pub const BLOCK_SIZE: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 128+64+16+2 = 210

    pub fn new(data: &'a [u8]) -> Self {
        let ql = &data[0..128];
        let qh = &data[128..192];
        let sc_bytes = &data[192..208];
        // Safety: &[i8] reinterpretation of &[u8] of same length
        let scales: &[i8] =
            unsafe { std::slice::from_raw_parts(sc_bytes.as_ptr() as *const i8, 16) };
        let d = f16_to_f32(&data[208..210]);
        Self { d, ql, qh, scales }
    }

    pub fn dequantize_into(&self, out: &mut [f32]) {
        // Reference (ggml-quants.c, dequantize_row_q6_K):
        //   for n in 0..QK_K step 128:  (outer = 0 or 1)
        //     for l in 0..32:
        //       is = l/16
        //       q1 = (i8)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
        //       q2 = (i8)((ql[l +32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
        //       q3 = (i8)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
        //       q4 = (i8)((ql[l +32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
        //       y[l+ 0] = d * sc[is+0] * q1
        //       y[l+32] = d * sc[is+2] * q2
        //       y[l+64] = d * sc[is+4] * q3
        //       y[l+96] = d * sc[is+6] * q4
        //     y  += 128; ql += 64; qh += 32; sc += 8
        let d = self.d;
        for outer in 0..(QK_K / 128) {
            let y_base = outer * 128;
            let ql_base = outer * 64;
            let qh_base = outer * 32;
            let sc_base = outer * 8;
            for l in 0..32 {
                let is = l / 16;
                let ql_lo = self.ql[ql_base + l];
                let ql_hi = self.ql[ql_base + l + 32];
                let qhb = self.qh[qh_base + l];

                let q1 = (((ql_lo & 0x0F) | (((qhb >> 0) & 0x3) << 4)) as i8).wrapping_sub(32);
                let q2 = (((ql_hi & 0x0F) | (((qhb >> 2) & 0x3) << 4)) as i8).wrapping_sub(32);
                let q3 = (((ql_lo >> 4) | (((qhb >> 4) & 0x3) << 4)) as i8).wrapping_sub(32);
                let q4 = (((ql_hi >> 4) | (((qhb >> 6) & 0x3) << 4)) as i8).wrapping_sub(32);

                out[y_base + l + 0] = d * self.scales[sc_base + is + 0] as f32 * q1 as f32;
                out[y_base + l + 32] = d * self.scales[sc_base + is + 2] as f32 * q2 as f32;
                out[y_base + l + 64] = d * self.scales[sc_base + is + 4] as f32 * q3 as f32;
                out[y_base + l + 96] = d * self.scales[sc_base + is + 6] as f32 * q4 as f32;
            }
        }
    }
}

// ============================================================================
// Q8_0 — 32-element blocks, 8-bit signed quants.
// Layout: d(fp16, 2) | qs(i8, 32)  -> 34 bytes/block
// ============================================================================
pub struct Q8_0Block<'a> {
    pub d: f32,
    pub qs: &'a [u8],
}

impl<'a> Q8_0Block<'a> {
    pub const BLOCK_SIZE: usize = 2 + QK8_0; // 34

    pub fn new(data: &'a [u8]) -> Self {
        let d = f16_to_f32(&data[..2]);
        Self {
            d,
            qs: &data[2..2 + QK8_0],
        }
    }

    pub fn dequantize_into(&self, out: &mut [f32]) {
        for i in 0..QK8_0 {
            out[i] = (self.qs[i] as i8) as f32 * self.d;
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        (self.qs[idx] as i8) as f32 * self.d
    }
}

// ============================================================================
// Q4_0 — 32-element blocks, 4-bit quants with an interleaved nibble layout.
// Layout: d(fp16, 2) | qs(16)   -> 18 bytes/block
// Elements 0..16 are the LOW nibbles of qs[0..16]; 16..32 are the HIGH nibbles.
// (Per llama.cpp dequantize_row_q4_0.)
// ============================================================================
pub struct Q4_0Block<'a> {
    pub d: f32,
    pub qs: &'a [u8], // 16 bytes
}

impl<'a> Q4_0Block<'a> {
    pub const BLOCK_SIZE: usize = 2 + QK4_0 / 2; // 18

    pub fn new(data: &'a [u8]) -> Self {
        let d = f16_to_f32(&data[..2]);
        Self {
            d,
            qs: &data[2..2 + QK4_0 / 2],
        }
    }

    pub fn dequantize_into(&self, out: &mut [f32]) {
        for j in 0..(QK4_0 / 2) {
            let b = self.qs[j];
            let x0 = ((b & 0x0F) as i32) - 8;
            let x1 = ((b >> 4) as i32) - 8;
            out[j] = x0 as f32 * self.d;
            out[j + QK4_0 / 2] = x1 as f32 * self.d;
        }
    }
}

// ============================================================================
// 6-bit scale/min extraction for Q4_K and Q5_K scales (12-byte field).
// Reference (ggml-common.h, get_scale_min_k4):
//   if (j < 4)
//     *d = q[j]   & 63;
//     *m = q[j+4] & 63;
//   else
//     *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
//     *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
// ============================================================================
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// ============================================================================
// Full-tensor dequantization helpers.
// ============================================================================

/// Dequantize a full tensor slice to F32.
pub fn dequantize_slice(data: &[u8], ggml_type: u32, num_elements: usize) -> Vec<f32> {
    match ggml_type {
        GGML_TYPE_F32 => {
            let n = num_elements.min(data.len() / 4);
            (0..n)
                .map(|i| {
                    f32::from_le_bytes([
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ])
                })
                .collect()
        }
        GGML_TYPE_F16 | GGML_TYPE_BF16 => {
            let n = num_elements.min(data.len() / 2);
            (0..n)
                .map(|i| f16_to_f32(&data[i * 2..i * 2 + 2]))
                .collect()
        }
        GGML_TYPE_Q4_K => dequant_q4k(data, num_elements),
        GGML_TYPE_Q5_K => dequant_q5k(data, num_elements),
        GGML_TYPE_Q6_K => dequant_q6k(data, num_elements),
        GGML_TYPE_Q8_0 | GGML_TYPE_Q8_1 => dequant_q8_0(data, num_elements),
        _ => {
            eprintln!(
                "dequantize_slice: unknown GGML type {}, using F16 fallback",
                ggml_type
            );
            let n = num_elements.min(data.len() / 2);
            (0..n)
                .map(|i| f16_to_f32(&data[i * 2..i * 2 + 2]))
                .collect()
        }
    }
}

fn dequant_q4k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut dst = vec![0f32; num_elements];
    let bs = Q4KBlock::BLOCK_SIZE;
    let blocks = num_elements.div_ceil(QK_K);
    let mut buf = [0f32; QK_K];
    for blk in 0..blocks {
        let start = blk * bs;
        if start + bs > data.len() {
            break;
        }
        let block = Q4KBlock::new(&data[start..start + bs]);
        block.dequantize_into(&mut buf);
        let base = blk * QK_K;
        let n = QK_K.min(num_elements - base);
        dst[base..base + n].copy_from_slice(&buf[..n]);
    }
    dst
}

fn dequant_q5k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut dst = vec![0f32; num_elements];
    let bs = Q5KBlock::BLOCK_SIZE;
    let blocks = num_elements.div_ceil(QK_K);
    let mut buf = [0f32; QK_K];
    for blk in 0..blocks {
        let start = blk * bs;
        if start + bs > data.len() {
            break;
        }
        let block = Q5KBlock::new(&data[start..start + bs]);
        block.dequantize_into(&mut buf);
        let base = blk * QK_K;
        let n = QK_K.min(num_elements - base);
        dst[base..base + n].copy_from_slice(&buf[..n]);
    }
    dst
}

fn dequant_q6k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut dst = vec![0f32; num_elements];
    let bs = Q6KBlock::BLOCK_SIZE;
    let blocks = num_elements.div_ceil(QK_K);
    let mut buf = [0f32; QK_K];
    for blk in 0..blocks {
        let start = blk * bs;
        if start + bs > data.len() {
            break;
        }
        let block = Q6KBlock::new(&data[start..start + bs]);
        block.dequantize_into(&mut buf);
        let base = blk * QK_K;
        let n = QK_K.min(num_elements - base);
        dst[base..base + n].copy_from_slice(&buf[..n]);
    }
    dst
}

fn dequant_q8_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut dst = vec![0f32; num_elements];
    let bs = Q8_0Block::BLOCK_SIZE;
    let blocks = num_elements.div_ceil(QK8_0);
    for blk in 0..blocks {
        let start = blk * bs;
        if start + bs > data.len() {
            break;
        }
        let block = Q8_0Block::new(&data[start..start + bs]);
        let base = blk * QK8_0;
        let n = QK8_0.min(num_elements - base);
        for i in 0..n {
            dst[base + i] = block.get(i);
        }
    }
    dst
}

#[inline]
pub fn f16_to_f32(bytes: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}
