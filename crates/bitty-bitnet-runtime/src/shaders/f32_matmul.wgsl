struct Params {
  N: u32,
  V: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> embed: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let flat_id = wg_id.x + wg_id.y * 65535u;
  let n = flat_id / params.V;
  let v = flat_id % params.V;

  if (n >= params.N || v >= params.V) {
    return;
  }

  let tid = local_id.x;
  var acc: f32 = 0.0;
  let hidden_base = n * params.D;
  let embed_base = v * params.D;
  let D_half = params.D / 2u;

  for (var dh = tid; dh < D_half; dh += WG_SIZE) {
    let d = dh * 2u;
    let packed = embed[embed_base / 2u + dh];
    let pair = unpack2x16float(packed);
    acc += hidden[hidden_base + d] * pair.x;
    acc += hidden[hidden_base + d + 1u] * pair.y;
  }

  shared_sums[tid] = acc;
  workgroupBarrier();

  for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[n * params.V + v] = shared_sums[0];
  }
}
