// Q6_K fused dequant+matmul: weight in Q6_K format [out_dim, in_dim].
// Q6_K blocks: 256 elements in 210 bytes (ql:128, qh:64, scales:16 i8, d:2 f16).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: MatMulConfig;

struct MatMulConfig {
    in_dim: u32,
    out_dim: u32,
}

const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

fn f16_to_f32(h: u32) -> f32 {
    let sign = h & 0x8000u;
    let exp = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    if (exp == 0u) {
        return select(f32(mant) * 5.960464477539063e-08, -f32(mant) * 5.960464477539063e-08, sign != 0u);
    }
    let m = f32(mant) / 1024.0 + 1.0;
    let e = f32(exp) - 15.0;
    return select(m * exp2(e), -m * exp2(e), sign != 0u);
}

fn dequant_q6k_block(blk: u32, buf: ptr<function, array<f32, 256>>) {
    // ql: 128 bytes starting at blk*BLOCK_BYTES => 32 u32s
    // qh: 64 bytes at blk*BLOCK_BYTES + 128 => 16 u32s
    // scales: 16 i8 at blk*BLOCK_BYTES + 192 => 4 u32s
    // d: f16 at blk*BLOCK_BYTES + 208 => use lower 16 bits of that u32
    let u32_base = blk * (BLOCK_BYTES / 4u);
    var s: array<u32, 32>;
    for (var i = 0u; i < 32u; i++) {
        s[i] = weight[u32_base + i];
    }
    let qh0 = s[32u];
    let qh1 = s[33u];
    let qh2 = s[34u];
    let qh3 = s[35u];
    var sc: array<u32, 4>;
    sc[0] = s[48u]; sc[1] = s[49u]; sc[2] = s[50u]; sc[3] = s[51u];
    let d_u16 = weight[u32_base + 52u] & 0xFFFFu;
    let d = f16_to_f32(d_u16);
    let d_tbl: array<f32, 8> = array(
        d * f32(i32(sc[0u] & 0xFFu)) - 32.0, d * f32(i32((sc[0u] >> 8u) & 0xFFu)) - 32.0,
        d * f32(i32((sc[0u] >> 16u) & 0xFFu)) - 32.0, d * f32(i32((sc[0u] >> 24u) & 0xFFu)) - 32.0,
        d * f32(i32(sc[1u] & 0xFFu)) - 32.0, d * f32(i32((sc[1u] >> 8u) & 0xFFu)) - 32.0,
        d * f32(i32((sc[1u] >> 16u) & 0xFFu)) - 32.0, d * f32(i32((sc[1u] >> 24u) & 0xFFu)) - 32.0,
    );

    for (var l = 0u; l < 32u; l++) {
        let ql0 = s[l / 2u] & 0xFFFFu;
        let ql1 = (s[l / 2u] >> 16u) & 0xFFFFu;
        let low_ql = (ql0 >> ((l & 1u) * 8u)) & 0xFFu;
        let high_ql = (ql1 >> ((l & 1u) * 8u)) & 0xFFu;
        let qh_byte = (qh0 >> ((l & 1u) * 8u)) & 0xFFu;
        let q1 = (low_ql & 0x0F) | (((qh_byte >> 0) & 3) << 4);
        let q2 = (high_ql & 0x0F) | (((qh_byte >> 2) & 3) << 4);
        let q3 = (low_ql >> 4) | (((qh_byte >> 4) & 3) << 4);
        let q4 = (high_ql >> 4) | (((qh_byte >> 6) & 3) << 4);
        let is = l / 16u;
        buf[l + 0u] = d_tbl[is + 0u] * f32(q1);
        buf[l + 32u] = d_tbl[is + 4u] * f32(q2);
        buf[l + 64u] = d_tbl[is + 8u] * f32(q3);
        buf[l + 96u] = d_tbl[is + 12u] * f32(q4);
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
        dequant_q6k_block(blk, &buf);
        let x_start = bi * QK_K;
        let n = min(QK_K, in_dim - x_start);
        for (var k = 0u; k < n; k++) {
            sum += input[x_start + k] * buf[k];
        }
    }
    output[j] = sum;
}
