// RMS Normalization: output[i] = (input[i] / rms(input)) * weight[i]
// Parallel workgroup reduction for mean-square computation.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: RmsConfig;

struct RmsConfig {
    dim: u32,
    eps: f32,
    _pad: u32,
}

var<workgroup> ws_ms: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    let dim = config.dim;
    let eps = config.eps;

    // Each thread accumulates its share
    var ms: f32 = 0.0;
    for (var i = lid; i < dim; i += 256u) {
        let v = input[i];
        ms += v * v;
    }
    ws_ms[lid] = ms;

    // Tree reduction over workgroup
    var stride: u32 = 128u;
    for (; stride > 0u; stride >>= 1u) {
        workgroupBarrier();
        if (lid < stride) {
            ws_ms[lid] += ws_ms[lid + stride];
        }
    }
    workgroupBarrier();

    let inv_rms = 1.0 / sqrt(ws_ms[0] / f32(dim) + eps);

    for (var i = lid; i < dim; i += 256u) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}
