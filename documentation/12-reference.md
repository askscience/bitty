# GGUF Format Reference

GGUF (GGML Universal Format) is a binary format for storing quantized neural network weights. It is the primary model format used by Bitty.

## File Structure

```
┌─────────────────────────────────────┐
│ Header                              │
│  - Magic: "GGUF" (0x46554747)       │
│  - Version: 3                       │
│  - Tensor Count: u64                │
│  - Metadata KV Count: u64           │
├─────────────────────────────────────┤
│ Metadata Key-Value Pairs            │
│  (repeated MetadataKVCount times)   │
│  - Key: string                      │
│  - Value: typed value               │
├─────────────────────────────────────┤
│ Tensor Info Entries                 │
│  (repeated TensorCount times)       │
│  - Name: string                     │
│  - Dimensions: [u64; N]             │
│  - GGML Type: u32                   │
│  - Offset: u64 (file offset)        │
├─────────────────────────────────────┤
│ Padding to 32-byte alignment        │
├─────────────────────────────────────┤
│ Tensor Data                         │
│  (at offsets specified in info)     │
│  - Raw quantized weight bytes       │
└─────────────────────────────────────┘
```

## Metadata Keys

### General

| Key | Type | Description |
|-----|------|-------------|
| `general.architecture` | string | Model architecture name |
| `general.name` | string | Model name |
| `general.description` | string | Model description |
| `general.file_type` | int32 | GGML file type version |
| `general.quantization_version` | int32 | Quantization format version |

### Model-Specific

| Key | Type | Architecture |
|-----|------|-------------|
| `{arch}.context_length` | int32 | All |
| `{arch}.embedding_length` | int32 | All |
| `{arch}.block_count` | int32 | All |
| `{arch}.feed_forward_length` | int32 | Llama, Mistral, etc. |
| `{arch}.attention.head_count` | int32 | All |
| `{arch}.attention.head_count_kv` | int32 | GQA models |
| `{arch}.attention.layer_norm_rms_epsilon` | f32 | All |
| `{arch}.rope.dimension_count` | int32 | RoPE models |
| `{arch}.rope.freq_base` | f32 | RoPE models |

Example: `llama.context_length`, `qwen2.attention.head_count`

### Tokenizer

| Key | Type | Description |
|-----|------|-------------|
| `tokenizer.ggml.model` | string | Tokenizer type (BPE, Unigram, WordPiece) |
| `tokenizer.ggml.bos_token_id` | int32 | Beginning of sequence token |
| `tokenizer.ggml.eos_token_id` | int32 | End of sequence token |
| `tokenizer.ggml.padding_token_id` | int32 | Padding token |
| `tokenizer.ggml.tokens` | [string] | Vocabulary tokens |
| `tokenizer.ggml.scores` | [float] | Token scores |
| `tokenizer.ggml.merges` | [string] | BPE merges |
| `tokenizer.ggml.token_type` | [int32] | Token types |
| `tokenizer.chat_template` | string | HuggingFace chat template |

## GGML Quantization Types

| Constant | Value | Name | Bytes per Element |
|----------|-------|------|-------------------|
| `GGML_TYPE_F32` | 0 | 32-bit float | 4 |
| `GGML_TYPE_F16` | 1 | 16-bit float | 2 |
| `GGML_TYPE_Q4_0` | 2 | 4-bit block (32-block) | 0.5 |
| `GGML_TYPE_Q4_1` | 3 | 4-bit block (32-block, higher precision) | 0.5 |
| `GGML_TYPE_Q5_0` | 6 | 5-bit block | 0.625 |
| `GGML_TYPE_Q5_1` | 7 | 5-bit block (higher precision) | 0.625 |
| `GGML_TYPE_Q8_0` | 8 | 8-bit block | 1 |
| `GGML_TYPE_Q8_1` | 9 | 8-bit block (higher precision) | 1 |
| `GGML_TYPE_Q2_K` | 10 | 2-bit K-quant | 0.25 |
| `GGML_TYPE_Q3_K` | 11 | 3-bit K-quant | 0.375 |
| `GGML_TYPE_Q4_K` | 12 | 4-bit K-quant | 0.5 |
| `GGML_TYPE_Q5_K` | 13 | 5-bit K-quant | 0.625 |
| `GGML_TYPE_Q6_K` | 14 | 6-bit K-quant | 0.75 |
| `GGML_TYPE_Q8_K` | 15 | 8-bit K-quant | 1 |
| `GGML_TYPE_IQ2_XXS` | 29 | 2-bit importance-weighted | 0.25 |
| `GGML_TYPE_IQ3_XXS` | 31 | 3-bit importance-weighted | 0.375 |
| `GGML_TYPE_TQ2_0` | 37 | Ternary 2-bit | 0.25 |

## Tensor Naming Convention

```
blk.{layer_id}.{component}.{operation}.weight
```

Examples:
- `blk.0.attn_q.weight` — Layer 0 attention query weights
- `blk.5.attn_k.weight` — Layer 5 attention key weights
- `blk.12.ffn_gate.weight` — Layer 12 FFN gate weights
- `token_embd.weight` — Input embedding
- `output.weight` — LM head / output projection
- `blk.0.attn_norm.weight` — Layer 0 attention normalization

## Parsing in Bitty

The `bitty-model` crate's `gguf.rs` module handles:

1. **Magic & version validation**: Confirm GGUF format and version 3
2. **Metadata extraction**: Iterate KV pairs, parse into `HashMap<String, Value>`
3. **Tensor info parsing**: Build tensor name → (shape, type, offset) map
4. **Architecture classification**: Match `general.architecture` to `ModelArchitecture`
5. **Layer metadata**: Extract per-layer information using naming conventions
6. **Memory-mapped access**: Use `memmap2` for zero-copy tensor data access

---

# Model Registry

**File**: `models/registry.toml`

The built-in registry contains 20+ models across multiple architecture families. Models are downloaded on-demand when running `bitty pull <name>`.

## BitNet Models

| Name | Parameters | Status |
|------|-----------|--------|
| `bitnet-b1.58` | 1.58-bit | stable (default) |

## Llama Family

| Name | Parameters | Status |
|------|-----------|--------|
| `tinyllama:1.1b` | 1.1B | stable |
| `smollm2:1.7b` | 1.7B | stable |
| `llama3.2:1b` | 1B | stable |
| `llama3.2:3b` | 3B | stable |
| `llama3:8b` | 8B | stable |

## Qwen Family

| Name | Parameters | Status |
|------|-----------|--------|
| `qwen2.5:0.5b` | 0.5B | stable |
| `deepseek-r1:1.5b` | 1.5B | stable |
| `qwen3.5:2b` | 2B | experimental |
| `qwen3:4b` | 4B | experimental |
| `qwen3:8b` | 8B | experimental |
| `qwen3:32b` | 32B | experimental |

## Gemma Family

| Name | Parameters | Status |
|------|-----------|--------|
| `gemma3:4b` | 4B | experimental |
| `gemma3:12b` | 12B | experimental |
| `gemma3:27b` | 27B | experimental |

## Mistral Family

| Name | Parameters | Status |
|------|-----------|--------|
| `mistral:7b` | 7B | stable |
| `mistral-nemo:12b` | 12B | experimental |

## Phi Family

| Name | Parameters | Status |
|------|-----------|--------|
| `phi3.5:3.8b` | 3.8B | stable |

## Registry Format

Each model entry follows this structure:

```toml
[[models]]
name = "model-name"
tag = "version-tag"
display_name = "Human Readable Name"
backend = "bitnet|candle"
quantization = "bit1|Q4_K_M|Q4_0|..."
filename = "model_file.gguf"
source = "https://huggingface.co/org/model-gguf"
url = "https://huggingface.co/org/model-gguf/resolve/main/model_file.gguf"
temperature = 0.7
num_predict = 128
num_ctx = 2048
status = "stable|experimental|deprecated"
```

## Adding a Model

To add a new model to the registry:

1. Find or create a GGUF-quantized version of the model
2. Add an entry to `models/registry.toml`
3. Test with `bitty pull <name>` and `bitty run <name>`
4. Submit a PR

---

# Quantization Types

Bitty's `Quantization` enum provides a unified abstraction over GGML quantization types, used for both weight storage and per-node assignment.

## Quantization Levels

```rust
pub enum Quantization {
    F32,   // 32-bit float (4 bytes/weight)
    Fp16,  // 16-bit float (2 bytes/weight)
    Q8,    // 8-bit integer (1 byte/weight)
    Q6,    // 6-bit block (0.75 bytes/weight)
    Q5,    // 5-bit block (0.625 bytes/weight)
    Q4,    // 4-bit block (0.5 bytes/weight)
    Q3,    // 3-bit block (0.375 bytes/weight)
    Q2,    // 2-bit block (0.25 bytes/weight)
    Bit1,  // 1-bit ternary (-1, 0, +1) (0.125 bytes/weight)
}
```

## Bytes Per Weight

```rust
impl Quantization {
    pub fn bytes_per_weight(&self) -> f32 {
        match self {
            F32  => 4.0,
            Fp16 => 2.0,
            Q8   => 1.0,
            Q6   => 0.75,
            Q5   => 0.625,
            Q4   => 0.5,
            Q3   => 0.375,
            Q2   => 0.25,
            Bit1 => 0.125,  // 2 bits per weight (ternary)
        }
    }
}
```

## Mapping to GGML Types

| Quantization | Corresponding GGML Types |
|--------------|-------------------------|
| F32 | `GGML_TYPE_F32` |
| Fp16 | `GGML_TYPE_F16` |
| Q8 | `GGML_TYPE_Q8_0`, `GGML_TYPE_Q8_1`, `GGML_TYPE_Q8_K` |
| Q6 | `GGML_TYPE_Q6_K` |
| Q5 | `GGML_TYPE_Q5_0`, `GGML_TYPE_Q5_1`, `GGML_TYPE_Q5_K` |
| Q4 | `GGML_TYPE_Q4_0`, `GGML_TYPE_Q4_1`, `GGML_TYPE_Q4_K` |
| Q3 | `GGML_TYPE_Q3_K`, `GGML_TYPE_IQ3_XXS` |
| Q2 | `GGML_TYPE_Q2_K`, `GGML_TYPE_IQ2_XXS` |
| Bit1 | `GGML_TYPE_TQ2_0` (ternary) |

## Precision vs Memory Trade-off

| Level | MB per 1B params | Relative Quality |
|-------|-----------------|------------------|
| F32 | 4000 MB | 100% (reference) |
| Fp16 | 2000 MB | ~99.9% |
| Q8 | 1000 MB | ~99.5% |
| Q6 | 750 MB | ~99.0% |
| Q5 | 625 MB | ~98.5% |
| Q4 | 500 MB | ~97.5% |
| Q3 | 375 MB | ~95.0% |
| Q2 | 250 MB | ~90.0% |
| Bit1 | 125 MB | ~85.0% (BitNet) |

## Assignment Strategy

The Halda scheduler assigns quantization per-node based on tier:

| Node Tier | Default Quantization | Critical Layers (embed + head) |
|-----------|---------------------|-------------------------------|
| S (high-end GPU) | Q4 | Fp16 |
| A (mid-range GPU) | Q4 | Fp16 |
| B (low-end GPU) | Q3 | Fp16 |
| C (high-end CPU) | Q2 | Fp16 |
| D (low-end CPU) | Q2 | Fp16 |

Critical layers (embedding and LM head) always use Fp16 regardless of tier, as they are most sensitive to quantization error.

## Computation Backend

Different quantization types are implemented in the CPU backend's `matmul/` module:

| Quantization | Matmul Implementation |
|-------------|----------------------|
| F32 | `matmul/f32.rs` |
| Q8 | `matmul/q8_0.rs` |
| Q6 | `matmul/q6k.rs` |
| Q5 | `matmul/q5k.rs` |
| Q4 | `matmul/q4_0.rs`, `matmul/q4k.rs` |

Each implementation provides a block-quantized matrix-vector multiply tailored to the specific quantization format.

---

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
