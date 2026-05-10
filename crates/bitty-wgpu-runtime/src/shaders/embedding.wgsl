// Embedding lookup: token_ids[N] -> hidden[N, dim]
// Each thread writes one element of the output.

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> embed_table: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: EmbedConfig;

struct EmbedConfig {
    dim: u32,
    scale: f32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = config.dim;
    let token_count = arrayLength(&token_ids);
    if (idx >= token_count * dim) {
        return;
    }
    let token_idx = idx / dim;
    let offset = idx % dim;
    let tid = token_ids[token_idx];
    output[idx] = embed_table[tid * dim + offset] * config.scale;
}
