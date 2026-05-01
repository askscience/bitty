# Bitty

Bitty is a Rust workspace for experimenting with distributed LLM inference over
heterogeneous peer networks. The initial implementation focuses on scheduler
correctness, deterministic simulation, and protocol boundaries before adding
production model kernels.

## Crates

- `bitty-protocol`: shared cluster messages and domain types.
- `bitty-coordinator`: worker registry, Halda scheduling, topology, routing, and snapshots.
- `bitty-worker`: profiling, shard lifecycle, ring execution, keepalive, and worker metrics.
- `bitty-model`: low-bit shard and activation codec primitives.
- `bitty-inference`: executor traits and request lifecycle orchestration.
- `bitty-sim`: deterministic in-process cluster simulation.
- `bitty-observability`: metrics and tracing helpers.

## Local Testing

Run the full verification suite:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Run a local Halda scheduling pass:

```bash
cargo run -p bitty-coordinator -- --nodes 8 --layers 16
```

Run a local worker profile and dummy keepalive pass:

```bash
cargo run -p bitty-worker -- --node-id local-worker-0 --keepalive
```

Run the deterministic in-process ring simulator:

```bash
cargo run -p bitty-sim -- --nodes 8 --layers 16 --tokens 4
```

Inject a simulated node failure:

```bash
cargo run -p bitty-sim -- --nodes 8 --layers 16 --tokens 4 --drop-node sim-0
```

Run the tiny local language model:

```bash
cargo run -p bitty-inference --bin bitty-tiny-lm -- --prompt "The coordinator" --chars 240 --seed 7
```

The tiny model is an in-repo byte-level probabilistic model trained from a small
distributed-inference corpus. It is useful for testing prompt-conditioned text
generation without downloading external weights. It is not yet a transformer,
BitNet, or distributed shard executor.

## BitNet b1.58 Testing

The smallest official practical BitNet target is Microsoft's
`BitNet-b1.58-2B-4T` GGUF model. Bitty delegates real BitNet execution to the
official `bitnet.cpp` runtime because BitNet requires specialized kernels.

Install the runtime and model:

```bash
scripts/setup_bitnet.sh
```

Run inference:

```bash
cargo run -p bitty-inference --bin bitty-bitnet -- \
  --prompt "Explain BitNet in one paragraph" \
  --n-predict 80
```

The default model path is:

```text
external/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

You can override paths if you already have the runtime or model:

```bash
cargo run -p bitty-inference --bin bitty-bitnet -- \
  --runtime-dir /path/to/BitNet \
  --model /path/to/ggml-model-i2_s.gguf \
  --prompt "What is 1-bit inference?"
```
