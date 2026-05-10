// Logit softcapping (Gemma2/3): logit[i] = cap * tanh(logit[i] / cap)
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> cap: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&logits)) { return; }
    logits[i] = cap * tanh(logits[i] / cap);
}
