struct Params {
  N: u32,
  D: u32,
  V: u32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> embed_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.N * params.D;
  if (idx >= total) {
    return;
  }

  let token = idx / params.D;
  let dim = idx % params.D;
  let token_id = token_ids[token];

  if (token_id < params.V) {
    let flat = token_id * params.D + dim;
    let packed = embed_table[flat / 2u];
    let pair = unpack2x16float(packed);
    output[idx] = select(pair.x, pair.y, (flat & 1u) == 1u);
  } else {
    output[idx] = 0.0;
  }
}
