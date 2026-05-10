// Dense F32 matrix-vector multiply: output[j] = sum_i(input[i] * weight[j * in_dim + i])
// Weight layout: [out_dim, in_dim] row-major.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: MatMulConfig;

struct MatMulConfig {
    in_dim: u32,
    out_dim: u32,
}

const WARP: u32 = 32u;

// Each work item computes one output element (j).
// Inner loop is tiled to reduce memory accesses.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= config.out_dim) {
        return;
    }
    var sum: f32 = 0.0;
    let in_dim = config.in_dim;
    // Simple dot product — can be blocked for better cache use later
    for (var i = 0u; i < in_dim; i += 1u) {
        sum += input[i] * weight[j * in_dim + i];
    }
    output[j] = sum;
}
