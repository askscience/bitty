// Q4_0 fused dequant+matmul: weight in Q4_0 format [out_dim, in_dim].
// Q4_0 blocks: 32 elements in 18 bytes (2 f16 d + 16 nibble-packed quants).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: MatMulConfig;

struct MatMulConfig {
    in_dim: u32,
    out_dim: u32,
}

const QK4_0: u32 = 32u;
const BLOCK_SIZE_BYTES: u32 = 18u;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= config.out_dim) {
        return;
    }
    let in_dim = config.in_dim;
    let blocks_per_row = (in_dim + QK4_0 - 1u) / QK4_0;
    var sum: f32 = 0.0;

    for (var bi = 0u; bi < blocks_per_row; bi++) {
        let blk = j * blocks_per_row + bi;
        let off = blk * BLOCK_SIZE_BYTES;
        let u32_off = off / 4u;
        let d_u16 = weight[u32_off] & 0xFFFFu;
        let d = f16_to_f32(d_u16);

        let x_start = bi * QK4_0;
        let n = min(QK4_0, in_dim - x_start);

        // Nibble-packed: qs[0..15] = low nibbles, qs[0..15] high nibbles
        // Per byte: low nibble → element[i], high nibble → element[16 + i]
        let n_bytes = (n + 1u) / 2u;
        for (var k = 0u; k < n_bytes; k++) {
            let byte_off = off + 2u + k;
            let u32_idx = byte_off / 4u;
            let shift = (byte_off & 3u) * 8u;
            let b = (weight[u32_idx] >> shift) & 0xFFu;
            let lo = i32(b & 0x0Fu) - 8;
            let hi = i32(b >> 4u) - 8;
            let lo_idx = x_start + k;
            let hi_idx = x_start + QK4_0 / 2u + k;
            if (lo_idx < in_dim) {
                sum += input[lo_idx] * f32(lo) * d;
            }
            if (hi_idx < in_dim) {
                sum += input[hi_idx] * f32(hi) * d;
            }
        }
    }
    output[j] = sum;
}
