// Q8_0 fused dequant+matmul: weight in Q8_0 format [out_dim, in_dim].
// Q8_0 blocks: 32 elements in 34 bytes (2 f16 d + 32 i8 quants).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: MatMulConfig;

struct MatMulConfig {
    in_dim: u32,
    out_dim: u32,
}

const QK8_0: u32 = 32u;
const BLOCK_SIZE_BYTES: u32 = 34u;

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
    let blocks_per_row = (in_dim + QK8_0 - 1u) / QK8_0;
    var sum: f32 = 0.0;

    for (var bi = 0u; bi < blocks_per_row; bi++) {
        let blk = j * blocks_per_row + bi;
        let off = blk * BLOCK_SIZE_BYTES;
        let u32_off = off / 4u;
        let d_u16 = weight[u32_off] & 0xFFFFu;
        let d = f16_to_f32(d_u16);

        let x_start = bi * QK8_0;
        let n = min(QK8_0, in_dim - x_start);
        for (var k = 0u; k < n; k++) {
            let byte_off = off + 2u + k;
            let u32_idx = byte_off / 4u;
            let shift = (byte_off & 3u) * 8u;
            let q_val = i32((weight[u32_idx] >> shift) & 0xFFu);
            if (q_val > 127) { q_val = q_val - 256; }
            sum += input[x_start + k] * f32(q_val) * d;
        }
    }
    output[j] = sum;
}
