# Shaders

## Slang → WGSL pipeline (planned)

All GPU compute shaders are authored in [Slang](https://shader-slang.org)
(`*.slang` files) and cross-compiled to WGSL via `slangc`. The generated WGSL
is committed to `generated/` so that end users do not need `slangc` installed.

To regenerate after editing Slang sources:
```sh
slangc <name>.slang -target wgsl -o generated/<name>.wgsl
```

Build script: `build.rs` verifies that the committed WGSL matches the Slang
source at compile time. If `slangc` is not on PATH, it skips verification.

## Currently implemented (hand-written WGSL)

| Shader | Status |
|---|---|
| `rmsnorm.wgsl` | ✅ Complete |
| `embedding.wgsl` | ✅ Complete |
| `matmul_f32.wgsl` | ✅ Complete |
| `matmul_q4k.wgsl` | ✅ Complete |
| `matmul_q8_0.wgsl` | ✅ Complete |
| `swiglu.wgsl` | ✅ Complete |

## Slang source files still needed (TODO)

| Slang source | Target WGSL | Priority |
|---|---|---|
| `matmul_q6_k.slang` | `generated/matmul_q6k.wgsl` | High |
| `rope_neox.slang` | `generated/rope_neox.wgsl` | High |
| `rope_interleaved.slang` | `generated/rope_interleaved.wgsl` | High |
| `attention_score.slang` | `generated/attention_score.wgsl` | High |
| `softmax.slang` | `generated/softmax.wgsl` | High |
| `attention_combine.slang` | `generated/attention_combine.wgsl` | High |
| `matmul_f16.slang` | `generated/matmul_f16.wgsl` | Medium |
| `dequant_q4_k.slang` | (used in matmul_q4k.slang) | Medium |
| `dequant_q6_k.slang` | (used in matmul_q6k.slang) | Medium |
| `dequant_q8_0.slang` | (used in matmul_q8_0.slang) | Medium |
| `add_bias.slang` | `generated/add_bias.wgsl` | Low |
| `softcap.slang` | `generated/softcap.wgsl` | Low |
| `matmul_i2_s.slang` | `generated/matmul_i2s.wgsl` | Low (BitNet, deferred) |
