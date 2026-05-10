// RoPE — applies rotary position embedding to concatenated [Q | K] buffer in-place.
// cos_sin stores [cos0, sin0, cos1, sin1, ...] for the current position (precomputed on CPU).

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;
@group(0) @binding(1) var<storage, read> cos_sin: array<f32>;
@group(0) @binding(2) var<uniform> config: RopeConfig;

struct RopeConfig {
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    style: u32,  // 0 = Neox (i, i+half), 1 = Interleaved (2i, 2i+1)
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let hd = config.head_dim;
    let rp = hd / 2u;
    let total_pairs_q = config.num_q_heads * rp;
    let total_pairs_k = config.num_kv_heads * rp;

    if (i >= total_pairs_q + total_pairs_k) { return; }

    let pair_idx = i % rp;
    let cs0 = cos_sin[pair_idx * 2u];
    let cs1 = cos_sin[pair_idx * 2u + 1u];

    if (i < total_pairs_q) {
        let head = i / rp;
        let off = head * hd;
        if (config.style == 0u) {
            let q0 = qk[off + pair_idx];
            let q1 = qk[off + pair_idx + rp];
            qk[off + pair_idx] = q0 * cs0 - q1 * cs1;
            qk[off + pair_idx + rp] = q0 * cs1 + q1 * cs0;
        } else {
            let q0 = qk[off + 2u * pair_idx];
            let q1 = qk[off + 2u * pair_idx + 1u];
            qk[off + 2u * pair_idx] = q0 * cs0 - q1 * cs1;
            qk[off + 2u * pair_idx + 1u] = q0 * cs1 + q1 * cs0;
        }
    } else {
        let ki = i - total_pairs_q;
        let kv_head = ki / rp;
        let k_base = config.num_q_heads * hd;
        let off = k_base + kv_head * hd;
        if (config.style == 0u) {
            let k0 = qk[off + pair_idx];
            let k1 = qk[off + pair_idx + rp];
            qk[off + pair_idx] = k0 * cs0 - k1 * cs1;
            qk[off + pair_idx + rp] = k0 * cs1 + k1 * cs0;
        } else {
            let k0 = qk[off + 2u * pair_idx];
            let k1 = qk[off + 2u * pair_idx + 1u];
            qk[off + 2u * pair_idx] = k0 * cs0 - k1 * cs1;
            qk[off + 2u * pair_idx + 1u] = k0 * cs1 + k1 * cs0;
        }
    }
}
