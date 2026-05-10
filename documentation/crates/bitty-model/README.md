# bitty-model

**Location**: `crates/bitty-model/`

**Purpose**: GGUF file parsing, model metadata extraction, architecture classification, activation compression codecs, and weight shard primitives.

## Modules

| Module | Responsibility |
|--------|---------------|
| `gguf.rs` | Binary GGUF parser: magic bytes, version, metadata KV pairs, tensor info, 37 GGML quantization type constants |
| `model_metadata.rs` | Architecture classification (`ModelArchitecture` enum), metadata extraction, `ShardPlan` builder |
| `bitnet.rs` | Backward-compatible re-exports for legacy naming |
| `activation_codec.rs` | `ActivationCodec` trait + FP8, Sparse TopK, Delta implementations |
| `tensor.rs` | `LowBitTensor`, `TensorShape`, quantization-aware size calculations |
| `shard.rs` | `WeightShard`, `WeightShardManifest`, `MappedWeightShard` (mmap-backed with SHA-256) |

## GGUF Parsing

The GGUF binary format consists of:
1. **Header**: Magic (`GGUF`), version (3), tensor count, metadata KV count
2. **Metadata KV pairs**: String-keyed values (strings, arrays, integers, floats)
3. **Tensor info**: Name, dimensions, GGML type, offset in file
4. **Tensor data**: Raw weight bytes at file offsets

Key function: `layer_id_from_tensor_name()` — parses tensor names like `blk.23.attn_q.weight` to extract layer ID (23).

## Supported Architectures

```rust
enum ModelArchitecture {
    BitNetB158,   // BitNet b1.58 — primary target
    OneBit,       // OneBit LLM variant
    Llama,        // Meta LLaMA family
    Mistral,      // Mistral AI family
    Phi,          // Microsoft Phi family
    Qwen2,        // Alibaba Qwen2
    Qwen35,       // Alibaba Qwen2.5
    Gemma,        // Google Gemma
    Gemma2,       // Google Gemma 2
    Falcon,       // TII Falcon
    StableLM,     // Stability AI StableLM
    DeepSeek,     // DeepSeek
    Mamba,        // State Space Model
    Unknown,      // Fallback
}
```

Classification is done by matching the `general.architecture` metadata key, with fallback heuristics.

## Activation Codecs

| Codec | Compression | Description |
|-------|-------------|-------------|
| `Fp8Linear` | 2:1 | Converts f16 to u8 via `((sample / 256) + 128).clamp(0, 255)` |
| `SparseTopK(0.30)` | ~3.3:1 | Keeps top 30% of values by magnitude, stores (index, f16) pairs |
| `Delta` | None | Passthrough with compression flag |

## Weight Shards

- **`WeightShard`**: Tensor name → raw bytes mapping for a layer range
- **`WeightShardManifest`**: Descriptor with SHA-256 of each shard for verification
- **`MappedWeightShard`**: Memory-mapped (mmap) shard for zero-copy access during inference

## Quantization Constants

37 GGML type constants are defined, including:
- `GGML_TYPE_F32` (0): 32-bit float
- `GGML_TYPE_F16` (1): 16-bit float
- `GGML_TYPE_Q4_0` (2): 4-bit block quantization
- `GGML_TYPE_Q8_0` (8): 8-bit block quantization
- `GGML_TYPE_TQ2_0` (37): Ternary 2-bit
- And all intermediate types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K variants)

`bytes_per_element()` and `quantization_from_ggml_type()` provide bidirectional mapping.
