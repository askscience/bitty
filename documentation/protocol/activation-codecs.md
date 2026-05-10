# Activation Codecs

Activation codecs compress intermediate tensor data sent between workers in the ring, reducing network bandwidth at the cost of some precision loss.

## Interface

```rust
pub trait ActivationCodec: Send + Sync {
    fn compress(&self, tensor: &ActivationTensor) -> Result<ActivationTensor>;
    fn decompress(&self, tensor: &ActivationTensor) -> Result<ActivationTensor>;
    fn kind(&self) -> CompressionKind;
}
```

## Available Codecs

### FP8 Linear Compression

**Compression ratio**: ~2:1 (16-bit → 8-bit)

Converts each fp16 value to u8 using linear quantization:

```
compressed = clamp(sample / 256 + 128, 0, 255)
decompressed = (value - 128) * 256
```

- Simple and fast (no search/sort)
- Preserves dynamic range reasonably for LLM activations
- Introduces ~0.39% quantization error on average

### Sparse TopK (30%)

**Compression ratio**: ~3.3:1 on average

1. Compute magnitude of each element
2. Keep only the top 30% of elements by magnitude
3. Store as (index: u32, value: f16) pairs
4. Zero out all other elements

```
Input:  [0.5, -2.3, 0.1, 4.2, -0.8, 1.1, ...]  (1280 elements)
Keep top 30% (384 elements):
Output: [(3, 4.2), (1, -2.3), (5, 1.1), ...]     (384 index-value pairs)
```

- Higher compression than FP8
- Non-deterministic output (depends on sparsity pattern)
- May lose information in dense activations

### Delta

**Compression ratio**: 1:1 (passthrough)

No compression — data is sent as-is with only the compression flag set. Useful for:
- Debugging and testing
- When network bandwidth is not a concern
- Layers where precision is critical (embedding, LM head)

## Selection Strategy

The compression codec is chosen per-layer-range by the coordinator based on:
1. **Node tier**: Lower-tier nodes get higher compression (Delta → FP8 → TopK)
2. **Layer position**: Embedding and LM head use Delta (no compression)
3. **Network bandwidth**: Low-bandwidth links use TopK
4. **Experimental**: Configurable per-deployment

## Usage in Ring

```rust
// Worker sends compressed activation
let codec = ActivationCodec::new(CompressionKind::Fp8);
let compressed = codec.compress(&activation)?;

// Send over Iroh
frame.send(opcode, token, &compressed).await?;

// Next worker receives and decompresses
let codec = ActivationCodec::new(compressed.compression);
let decompressed = codec.decompress(&compressed)?;

// Execute layers with decompressed activation
let output = executor.execute_range(decompressed).await?;
```

## CRC32 Checksums

Every activation tensor includes a CRC32 checksum of the packed data. The receiving worker verifies the checksum before decompression to detect corruption in transit. Checksum failures increment the `dlm_checksum_failures_total` metric.
