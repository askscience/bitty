# bitty-bitnet-runtime

**Location**: `crates/bitty-bitnet-runtime/`

**Purpose**: BitNet b1.58 model inference runtime with both GPU (Candle/wgpu) and CPU backends. This is the core inference engine.

## GPU Backend

### BitNetRuntime

The primary runtime for GPU-accelerated inference:

```rust
pub struct BitNetRuntime {
    model: SplitBitNetModel,
    tokenizer: Tokenizer,
    device: Device,  // Candle Device (Cuda, Metal, Cpu)
}
```

### Key methods:
- `load()` — Load model from GGUF file, auto-detect device (Metal/CUDA/CPU)
- `embed_tokens()` — Convert token IDs → hidden states on GPU
- `forward_layers()` — Execute a range of transformer layers
- `final_logits()` — Compute logits from final hidden state
- `sample()` — Sample next token with temperature
- `generate()` — Full autoregressive generation (stream or collect all)

### SplitBitNetModel

Wraps `CandleModel` with GPU activation management:

| Method | Description |
|--------|-------------|
| `embed_tokens()` | Token IDs → GPU tensor |
| `upload_activation()` | Upload CPU tensor to GPU |
| `forward_layers()` | Run layers on GPU |
| `read_activation()` | Download GPU tensor to CPU |
| `read_logits()` | Download logits from GPU |

### BitNetShard

Describes a layer range assignment for distributed execution:

```rust
pub struct BitNetShard {
    pub range: Range<usize>,
    pub owns_embedding: bool,
    pub owns_lm_head: bool,
}
```

### KV Cache

`BitNetKvCache` manages the key-value cache for autoregressive decoding. In distributed mode, it is a placeholder — each worker manages its own slice.

## CPU Backend

### CpuModel

Full CPU inference pipeline located in `cpu_backend/`:

```
cpu_backend/
├── mod.rs          # CpuModel, forward, generate
├── types.rs        # CpuModelMetadata, CpuLayer, KvCache, RopeCache, LmHead
├── loader/
│   ├── mod.rs      # GGUF weight loader
│   ├── metadata.rs # Config extraction from GGUF metadata
│   └── names.rs    # Tensor name classification
├── layers/
│   ├── mod.rs      # Layer dispatch
│   ├── attention.rs# Attention layer forward
│   ├── mlp.rs      # MLP layer forward
│   ├── ssm.rs      # Mamba state space model
│   └── linear_attn.rs # Linear attention (Qwen-style)
├── matmul/
│   ├── mod.rs      # Dispatches to per-type matmul
│   ├── f32.rs      # F32 matmul
│   ├── q4_0.rs     # Q4_0 quantized matmul
│   ├── q4k.rs      # Q4_K quantized matmul
│   ├── q5k.rs      # Q5_K quantized matmul
│   ├── q6k.rs      # Q6_K quantized matmul
│   └── q8_0.rs     # Q8_0 quantized matmul
├── dequant.rs      # Block dequantization readers
└── ops.rs          # RMSNorm, SiLU, softplus, softmax, RoPE
```

### Layer Types
- **Attention**: Multi-head self-attention with RoPE
- **MLP**: Feed-forward network (SwiGLU, ReLU, etc.)
- **SSM**: Mamba state space model layer
- **Linear Attention**: Qwen-style linear attention

### Quantized Matmul
Each quantization type has an optimized matrix-vector multiply:
- `f32` — Reference float implementation
- `q4_0` — 4-bit block quantization (32-block)
- `q4_k` — 4-bit K-quant (256-block)
- `q5_k` — 5-bit K-quant
- `q6_k` — 6-bit K-quant
- `q8_0` — 8-bit block quantization

## Backend Selection

```rust
fn auto_device() -> Device {
    // 1. Try Metal (macOS)
    // 2. Try CUDA (NVIDIA)
    // 3. Fall back to CPU
}
```

The device selection is also influenced by `BITTY_BACKEND` environment variable and the model's quantization type.
