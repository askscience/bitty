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
