// Q4_K fused dequant+matmul: weight in Q4_K format [out_dim, in_dim].
// Q4_K blocks: 256 elements in 144 bytes (2 f16 d + 2 f16 dmin + 12 scale + 128 qs).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>; // uint4 for 4-bit quants
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: MatMulConfig;

struct MatMulConfig {
    in_dim: u32,
    out_dim: u32,
}

const QK_K: u32 = 256u;
const BLOCK_SIZE_BYTES: u32 = 144u;
const BLOCK_SIZE_U32: u32 = 36u; // 144 / 4

// f16 unpack helper
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

// Dequantize a Q4_K block into buf[256]
fn dequant_q4k_block(block_base: u32, buf: ptr<function, array<f32, 256>>) {
    let d_u16 = weight[block_base / 4u] & 0xFFFFu;
    let d = f16_to_f32(d_u16);
    let dmin_u16 = (weight[block_base / 4u] >> 16u) & 0xFFFFu;
    let dmin = f16_to_f32(dmin_u16);

    // scales at block_base + 4u (12 bytes = 3 u32s)
    let s0 = weight[block_base / 4u + 1u];
    let s1 = weight[block_base / 4u + 2u];
    let s2 = weight[block_base / 4u + 3u];

    // qs at block_base + 16u (128 bytes = 32 u32s, each holding 2 x 4-bit values)
    let qs_base = block_base / 4u + 4u;

    for (var j = 0u; j < QK_K; j += 64u) {
        let is = (j / 64u) * 2u;
        let (sc1, m1) = get_scale_min_k4(is, s0, s1, s2);
        let (sc2, m2) = get_scale_min_k4(is + 1u, s0, s1, s2);
        let d1 = d * f32(sc1);
        let md1 = dmin * f32(m1);
        let d2 = d * f32(sc2);
        let md2 = dmin * f32(m2);

        let q_off = is / 2u * 32u; // 32 bytes = 8 u32s for low/high nibbles
        for (var l = 0u; l < 32u; l++) {
            let qw = weight[qs_base + q_off / 4u + l / 2u];
            let shift = (l & 1u) * 16u;
            let b = (qw >> shift) & 0xFFu;
            buf[j + l] = d1 * f32(b & 0x0Fu) - md1;
            buf[j + 32u + l] = d2 * f32(b >> 4u) - md2;
        }
    }
}

fn get_scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    // Scales packed: s0[0:8]=scales[0], s0[8:16]=scales[1], etc.
    // mins at offset 4
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
        dequant_q4k_block(block_byte_off, &buf);
        let x_start = bi * QK_K;
        let n = min(QK_K, in_dim - x_start);
        for (var k = 0u; k < n; k++) {
            sum += input[x_start + k] * buf[k];
        }
    }
    output[j] = sum;
}
