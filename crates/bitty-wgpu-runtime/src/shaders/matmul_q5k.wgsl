// Q5_K fused dequant+matmul: weight in Q5_K format [out_dim, in_dim].
// Q5_K blocks: 256 elements in 176 bytes
//   (2 f16 d + 2 f16 dmin + 12 scales + 32 qh + 128 qs).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: MatMulConfig;

struct MatMulConfig {
    in_dim: u32,
    out_dim: u32,
}

const QK_K: u32 = 256u;
const BLOCK_SIZE_BYTES: u32 = 176u;
const BLOCK_SIZE_U32: u32 = 44u; // 176 / 4

fn f16_to_f32(h: u32) -> f32 {
    let sign = h & 0x8000u;
    let exp = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    if (exp == 0u) {
        return select(
            f32(mant) * 5.960464477539063e-08,
            -f32(mant) * 5.960464477539063e-08,
            sign != 0u,
        );
    }
    let m = f32(mant) / 1024.0 + 1.0;
    let e = f32(exp) - 15.0;
    return select(m * exp2(e), -m * exp2(e), sign != 0u);
}

fn get_scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    var sc: u32;
    var m: u32;
    if (j < 4u) {
        let shift = j * 8u;
        sc = (s0 >> shift) & 0x3Fu;
        m = (s1 >> shift) & 0x3Fu;
    } else {
        let j4 = j - 4u;
        let shift = j4 * 8u;
        let high = (s2 >> shift) & 0x0Fu;
        let low = (s1 >> (j * 6u - 24u)) & 0x30u;
        sc = high | low;
        let m_high = (s2 >> (shift + 4u)) & 0x0Fu;
        let m_low = (s0 >> (j * 6u - 24u)) & 0x30u;
        m = m_high | m_low;
    }
    return vec2(sc & 0x3Fu, m & 0x3Fu);
}

fn dequant_q5k_block(block_base: u32, buf: ptr<function, array<f32, 256>>) {
    let base_u32 = block_base / 4u;
    let d_u16 = weight[base_u32] & 0xFFFFu;
    let d = f16_to_f32(d_u16);
    let dmin_u16 = (weight[base_u32] >> 16u) & 0xFFFFu;
    let dmin = f16_to_f32(dmin_u16);

    // scales at base_u32 + 1 (12 bytes = 3 u32s)
    let s0 = weight[base_u32 + 1u];
    let s1 = weight[base_u32 + 2u];
    let s2 = weight[base_u32 + 3u];

    // qh at base_u32 + 4 (32 bytes = 8 u32s), one bit per element (high bit)
    // qs at base_u32 + 12 (128 bytes = 32 u32s), low-4 nibbles

    var u1: u32 = 1u;
    var u2: u32 = 2u;

    for (var j = 0u; j < QK_K; j += 64u) {
        let is = (j / 64u) * 2u;
        let (sc1, m1) = get_scale_min_k4(is, s0, s1, s2);
        let (sc2, m2) = get_scale_min_k4(is + 1u, s0, s1, s2);
        let d1 = d * f32(sc1);
        let md1 = dmin * f32(m1);
        let d2 = d * f32(sc2);
        let md2 = dmin * f32(m2);

        let qs_offset = is / 2u * 32u; // 32 bytes per qs chunk
        let qh_offset = j / 8u;        // 4 qh bytes per 64-element chunk

        for (var l = 0u; l < 32u; l++) {
            // qs: packed in u32 array starting at base_u32 + 12
            let qs_byte = qs_offset + l;
            let qs_u32 = base_u32 + 12u + qs_byte / 4u;
            let qs_shift = (qs_byte & 3u) * 8u;
            let qs = (weight[qs_u32] >> qs_shift) & 0xFFu;

            // qh: one byte holds high bits for 8 elements
            let qh_byte = qh_offset + l / 8u;
            let qh_u32 = base_u32 + 4u + qh_byte / 4u;
            let qh_shift = (qh_byte & 3u) * 8u;
            let qh = (weight[qh_u32] >> qh_shift) & 0xFFu;
            let qh_bit = qh >> (l & 7u);

            let lo = (qs & 0x0Fu) | (((qh_bit & u1) << 4u));
            let hi = (qs >> 4u)    | (((qh_bit & u2) << 4u));

            buf[j + l] = d1 * f32(lo) - md1;
            buf[j + 32u + l] = d2 * f32(hi) - md2;
        }
        u1 = u1 << 2u;
        u2 = u2 << 2u;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= config.out_dim) {
        return;
    }
    let in_dim = config.in_dim;
    let blocks_per_row = (in_dim + QK_K - 1u) / QK_K;
    var sum: f32 = 0.0;
    var buf: array<f32, 256>;

    for (var bi = 0u; bi < blocks_per_row; bi++) {
        let blk = j * blocks_per_row + bi;
        let block_byte_off = blk * BLOCK_SIZE_BYTES;
        dequant_q5k_block(block_byte_off, &buf);
        let x_start = bi * QK_K;
        let n = min(QK_K, in_dim - x_start);
        for (var k = 0u; k < n; k++) {
            sum += input[x_start + k] * buf[k];
        }
    }
    output[j] = sum;
}
