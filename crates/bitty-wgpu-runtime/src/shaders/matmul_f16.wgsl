// F16 matrix-vector multiply. Weight is stored as F16 [out_dim, in_dim].
// Each u32 contains two f16 values.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: MatMulConfig;

struct MatMulConfig {
    in_dim: u32,
    out_dim: u32,
}

fn f16_to_f32(h: u32) -> f32 {
    let sign = f32(h >> 15u & 1u) * -2.0 + 1.0;
    let exp = i32((h >> 10u) & 0x1Fu) - 15;
    let mant = f32(h & 0x3FFu) / 1024.0;
    if (exp == -15) { return sign * mant * 5.960464477539063e-08; }
    return sign * (mant + 1.0) * exp2(f32(exp));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= config.out_dim) { return; }
    var sum: f32 = 0.0;
    let in_dim = config.in_dim;
    let f16_per_u32 = 2u;
    for (var i = 0u; i < in_dim; i++) {
        let flat = j * in_dim + i;
        let u32_idx = flat / f16_per_u32;
        let shift = (flat & 1u) * 16u;
        let h16 = (weight[u32_idx] >> shift) & 0xFFFFu;
        sum += input[i] * f16_to_f32(h16);
    }
    output[j] = sum;
}
