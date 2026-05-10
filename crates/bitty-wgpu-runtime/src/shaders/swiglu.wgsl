// SwiGLU: gate = silu(x @ gate_proj), up = x @ up_proj, result = gate * up.
// The matmuls are dispatched separately via matmul_q4k/matmul_f32.
// This shader only performs the element-wise gate * up multiplication.
//
// Input layout: [gate_output, up_output] concatenated
// output[i] = silu(gate[i]) * up[i]

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dim: u32;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dim) {
        return;
    }
    output[i] = silu(gate[i]) * up[i];
}
