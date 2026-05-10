# bitty-candle-runtime

**Location**: `crates/bitty-candle-runtime/`

**Purpose**: Candle-based model loading, tokenizer integration, KV cache management, and sampling. This crate provides the foundational model infrastructure used by the GPU inference runtime.

## Modules

| Module | Responsibility |
|--------|---------------|
| `device.rs` | `auto_device()` — Metal, CUDA, or CPU detection |
| `tokenizer.rs` | HuggingFace tokenizer wrapper for GGUF models |
| `model.rs` | `CandleModel` — GGUF weight loading into Candle tensors |
| `load.rs` | Low-level weight loading from GGUF byte buffers |
| `layers.rs` | `ModelConfig` and transformer layer definitions |
| `dequant.rs` | Dequantization helpers for quantized weights |
| `kv_cache.rs` | Key-value cache for autoregressive decoding |
| `sampling.rs` | Token sampling (argmax, temperature) |

## Auto Device

```rust
pub fn auto_device() -> (Device, DeviceKind)
```

Detection order:
1. **CUDA** — If `candle-core` compiled with CUDA support and NVIDIA GPU present
2. **Metal** — If running on macOS with Metal GPU (default on Apple Silicon)
3. **CPU** — Fallback

Controlled by crate features: `cuda`, `metal`, `cpu-mkl`, `cpu-accelerate`

## Tokenizer

Wraps the HuggingFace `tokenizers` crate with GGUF-specific loading:

```rust
pub struct Tokenizer {
    inner: hf_tokenizers::Tokenizer,
    eos_id: u32,
    bos_id: u32,
    pad_id: u32,
}
```

### Features:
- Load tokenizer from GGUF metadata (embedded tokenizer.json)
- `encode()` — text → token IDs
- `decode()` — token IDs → text
- `apply_chat_template()` — apply HuggingFace chat template
- Special token management (EOS, BOS, PAD, UNK)
- Supports all HuggingFace tokenizer types (BPE, Unigram, WordPiece, etc.)

## CandleModel

```rust
pub struct CandleModel {
    pub tensors: HashMap<String, Tensor>,
    pub config: ModelConfig,
    pub device: Device,
}
```

- Loads GGUF weight tensors as Candle `Tensor` objects
- `config` contains model architecture parameters (hidden_size, num_layers, num_heads, etc.)
- Weights remain on the device (GPU/CPU) for inference

## KV Cache

```rust
pub struct KvCache {
    pub k: Tensor,
    pub v: Tensor,
    pub position: usize,
}
```

- Pre-allocated for maximum context length
- Updated in-place during autoregressive decoding
- `append()`, `trim()`, `reset()` operations
- Supports GQA (Grouped Query Attention) cache shapes

## Sampling

- `sample_token(logits, temperature)` — Returns `(token_id, logprob)`
- `argmax` sampling when temperature = 0
- Temperature-scaled softmax for stochastic sampling
