# bitty-inference

**Location**: `crates/bitty-inference/`

**Purpose**: Inference executor traits, request lifecycle orchestration, sampling algorithms, and test doubles (fake/stub executors). Also contains the Tiny Language Model for integration testing.

## Modules

| Module | Responsibility |
|--------|---------------|
| `executor.rs` | `LayerExecutor` trait, `FakeLayerExecutor`, `StubLayerExecutor`, `LowBitReferenceExecutor` |
| `backend.rs` | `ModelBackendKind`, `BackendCapability`, `BackendDecision` metadata types |
| `lifecycle.rs` | `BatchPolicy`, `PrefixCacheKey`, `RequestLifecycle`, `InferencePhase`, `DecodePipeline` |
| `rust_bitnet.rs` | `BitNetBackendProbe`, `BitNetLayerExecutor` — bridges bitnet-runtime with executor trait |
| `sampling.rs` | `argmax()`, `sample_with_temperature()`, `xorshift_f32()` |
| `tiny_lm.rs` | Byte-level probabilistic model for testing (no external weights needed) |

## LayerExecutor Trait

The central abstraction for model execution:

```rust
#[async_trait]
pub trait LayerExecutor: Send + Sync {
    /// Execute a range of layers on an activation tensor
    async fn execute_range(
        &self,
        request_id: Uuid,
        layer_range: Range<usize>,
        activation: ActivationTensor,
        inference_phase: InferencePhase,
    ) -> Result<ActivationTensor, ExecutorError>;

    /// Compute final logits from the last hidden state
    async fn final_logits(
        &self,
        request_id: Uuid,
        activation: ActivationTensor,
    ) -> Result<BitNetLogits, ExecutorError>;

    /// Decode a token ID to text
    async fn decode_token_text(
        &self,
        token_id: u32,
    ) -> Result<String, ExecutorError>;
}
```

### Implementations

| Implementation | Purpose |
|----------------|---------|
| `FakeLayerExecutor` | In-memory no-op executor for testing; returns identity activations |
| `StubLayerExecutor` | Always returns errors — tests error handling paths |
| `LowBitReferenceExecutor` | Reference implementation for correctness checking |
| `BitNetLayerExecutor` | Wraps `BitNetRuntime` with KV cache + decode state management |

## Sampling

### argmax
Returns the index of the maximum logit value. Used for greedy decoding (temperature = 0).

### sample_with_temperature
1. Apply softmax to logits: `p_i = exp(logit_i / T) / sum_j exp(logit_j / T)`
2. Sample from the resulting distribution using xorshift64 PRNG
3. Returns the sampled token ID + logprobs

### xorshift_f32
Fast non-cryptographic PRNG producing f32 in [0, 1). Based on xorshift64, no external dependencies.

## Request Lifecycle

```
InferencePhase::Embed      → Run embedding layer
InferencePhase::Forward(i) → Run transformer layer i
InferencePhase::Final      → Run LM head → logits → sample
```

The `RequestLifecycle` enum tracks which phase a request is in, enabling the coordinator to manage multi-token generation across the ring.

## Tiny Language Model

`TinyLanguageModel` is an in-repo byte-level probabilistic model for testing without external weights:
- 96-character vocabulary (printable ASCII subset)
- Markov chain based on character bigram/trigram probabilities
- Deterministic output for a given seed
- ~2KB of state, compiles instantly
- Used in `bitty-sim` for end-to-end cluster tests

## Binaries

| Binary | Purpose |
|--------|---------|
| `bitty-tiny-lm` | Run the tiny LM as a standalone demo |
| `bitty-rust-bitnet` | Run BitNet GPU runtime on a GGUF model |
| `bitty-client` | gRPC client to coordinator for manual testing |
