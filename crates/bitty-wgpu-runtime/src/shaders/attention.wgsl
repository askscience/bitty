// Fused causal attention: Q·K^T, softmax, weighted V sum.
// GQA support: K/V have fewer heads than Q (num_heads % num_kv_heads == 0).
//
// Bindings:
//   [0] q        — [num_heads * head_dim] current query (read)
//   [1] k_cache  — [num_kv_heads * seq_len * head_dim] accumulated keys (read)
//   [2] v_cache  — [num_kv_heads * seq_len * head_dim] accumulated values (read)
//   [3] output   — [num_heads * head_dim] attention output (read_write)
//   [4] config   — uniform AttentionConfig

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> config: AttentionConfig;

struct AttentionConfig {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    scale: f32,        // 1.0 / sqrt(head_dim)
}

// Maximum sequence length supported in a single softmax pass.
// For longer sequences the work item loops in chunks.
const MAX_SEQ_PER_PASS: u32 = 2048u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= config.num_heads) { return; }

    let hd = config.head_dim;
    let nk = config.num_kv_heads;
    let seq = config.seq_len;
    let groups = config.num_heads / nk;
    let kv_h = h / groups;
    let scale = config.scale;

    // Q offset for this head
    let q_off = h * hd;

    // Find max score for numerical stability (softmax subtracts max)
    var max_score: f32 = -1e30;
    for (var j = 0u; j < seq; j++) {
        let k_off = j * nk * hd + kv_h * hd;
        var dot: f32 = 0.0;
        for (var d = 0u; d < hd; d++) {
            dot += q[q_off + d] * k_cache[k_off + d];
        }
        let s = dot * scale;
        if (s > max_score) { max_score = s; }
    }

    // Compute exp sum and weighted V output
    var exp_sum: f32 = 0.0;
    let o_off = h * hd;
    for (var d = 0u; d < hd; d++) {
        output[o_off + d] = 0.0;
    }

    for (var j = 0u; j < seq; j++) {
        let k_off = j * nk * hd + kv_h * hd;
        var dot: f32 = 0.0;
        for (var d = 0u; d < hd; d++) {
            dot += q[q_off + d] * k_cache[k_off + d];
        }
        let s = (dot * scale) - max_score;
        let w = exp(s);
        exp_sum += w;

        let v_off = j * nk * hd + kv_h * hd;
        for (var d = 0u; d < hd; d++) {
            output[o_off + d] += w * v_cache[v_off + d];
        }
    }

    // Normalize by exp sum
    if (exp_sum > 0.0) {
        for (var d = 0u; d < hd; d++) {
            output[o_off + d] = output[o_off + d] / exp_sum;
        }
    }
}
