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
