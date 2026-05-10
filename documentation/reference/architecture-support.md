# Architecture Support

## Supported Architectures

Bitty supports 14 model architectures for metadata extraction and architecture detection. Full inference (CPU backend) is supported for a growing subset.

### Full Inference Support

| Architecture | GPU Backend | CPU Backend | Status |
|-------------|-------------|-------------|--------|
| BitNet b1.58 | ✅ (`bitty-bitnet-runtime`) | ✅ (`cpu_backend`) | Stable |
| Llama | ❌ | ✅ (attention + MLP) | Stable |
| Mistral | ❌ | ✅ (attention + MLP) | Stable |
| Qwen2 | ❌ | ✅ (attention + MLP) | Stable |
| Qwen2.5 | ❌ | ✅ (attention + MLP) | Stable |
| Gemma | ❌ | ✅ (attention + MLP) | Stable |
| Phi | ❌ | ✅ (attention + MLP) | Stable |

### Metadata Extraction Only

| Architecture | Detection | Layer Info | Full Inference |
|-------------|-----------|------------|----------------|
| Qwen3.5 | ✅ | ✅ | ❌ |
| Qwen3 | ✅ | ✅ | ❌ |
| Gemma 2 | ✅ | ✅ | ❌ |
| Falcon | ✅ | ✅ | ❌ |
| StableLM | ✅ | ✅ | ❌ |
| DeepSeek | ✅ | ✅ | ❌ |
| Mamba | ✅ | ✅ | ❌ (SSM layer not yet fully integrated) |
| OneBit | ✅ | ✅ | ❌ |

## Architecture Detection

Detection is based on the `general.architecture` GGUF metadata key:

| Metadata Value | Architecture | Family |
|---------------|-------------|--------|
| `bitnet-b1.58` | BitNetB158 | BitNet |
| `llama` | Llama | Llama |
| `mistral` | Mistral | Mistral |
| `phi` | Phi | Phi |
| `qwen2` | Qwen2 | Qwen |
| `qwen2.5` | Qwen2(.5) | Qwen |
| `gemma` | Gemma | Gemma |
| `gemma2` | Gemma2 | Gemma |
| `falcon` | Falcon | Falcon |
| `stablelm` | StableLM | StableLM |
| `deepseek` | DeepSeek | DeepSeek |
| `mamba` | Mamba | Mamba |
| `onebit` | OneBit | OneBit |

Fallback: `Unknown` — basic tensor shape analysis still works for shard planning.

## Layer Types

### Attention (supported in CPU backend)

```rust
pub fn attention_forward(
    hidden: &[f32],
    wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32],
    kv_cache: &mut KvCache,
    rope: &RopeCache,
    config: &ModelConfig,
) -> Vec<f32>
```

Supports:
- Multi-head attention (MHA)
- Grouped query attention (GQA)
- Multi-query attention (MQA)
- Rotary Position Embeddings (RoPE)
- Flash attention-like kernel (optimized)

### MLP (supported in CPU backend)

```rust
pub fn mlp_forward(
    hidden: &[f32],
    w1: &[f32], w2: &[f32], w3: &[f32],
    config: &ModelConfig,
) -> Vec<f32>
```

Supports:
- SwiGLU activation (SiLU × sigmoid gate)
- ReLU activation
- GELU activation
- Standard FFN (2-layer)
- Gated FFN (3-layer)

### SSM (in development)

Mamba state space model layer:

```rust
pub fn ssm_forward(
    hidden: &[f32],
    ssm_state: &mut RecurrentState,
    config: &ModelConfig,
) -> Vec<f32>
```

- Selective scan algorithm
- Recurrent state management
- Discretized SSM parameters

### Linear Attention (Qwen-style)

```rust
pub fn linear_attn_forward(
    hidden: &[f32],
    config: &ModelConfig,
) -> Vec<f32>
```

## Adding a New Architecture

1. **Add variant** to `ModelArchitecture` enum in `bitty-model/src/model_metadata.rs`
2. **Add key mappings** in the architecture detection function
3. **Add tensor name patterns** in `bitty-bitnet-runtime/src/cpu_backend/loader/names.rs`
4. **Add layer implementation** in `bitty-bitnet-runtime/src/cpu_backend/layers/` if needed
5. **Add dispatch** in `bitty-bitnet-runtime/src/cpu_backend/layers/mod.rs`
6. **Add test** with a GGUF metadata fixture
7. **Add registry entry** in `models/registry.toml`
