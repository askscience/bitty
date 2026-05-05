# Performance notes (evidence-driven)

## SIMD / CPU intrinsics

Bitty's own Rust sources do not use `std::arch` SIMD. Real BitNet inference in this repo is delegated to **`oxbitnet` + `wgpu`** (`crates/bitty-bitnet-runtime`). Before adding AVX/NEON here, profile the path you care about:

1. GPU vs CPU-bound: if the GPU queue is saturated, intrinsics in Bitty will not help.
2. If a flamegraph points inside **`oxbitnet`**, the fix belongs in that dependency or its configuration—not random intrinsics in Bitty.

## Benchmarks

Run all benchmarks:

```bash
cargo bench --workspace
```

Run a specific benchmark group:

```bash
# Model data & metadata
cargo bench -p bitty-model --bench gguf_parsing
cargo bench -p bitty-model --bench metadata_extraction
cargo bench -p bitty-model --bench gguf_helpers
cargo bench -p bitty-model --bench tensor_ops
cargo bench -p bitty-model --bench activation_codec
cargo bench -p bitty-model --bench shard_planning

# Protocol & wire format
cargo bench -p bitty-protocol --bench logits_wire
cargo bench -p bitty-protocol --bench activation_wire

# Inference & execution
cargo bench -p bitty-inference --bench ring_execution

# Scheduling & topology
cargo bench -p bitty-coordinator --bench scheduling

# Cluster simulation
cargo bench -p bitty-sim --bench cluster_simulation

# Worker profiling
cargo bench -p bitty-worker --bench profiling
```

### Benchmark categories

| Crate | Benchmarks | What's measured |
|-------|-----------|-----------------|
| `bitty-model` | 6 files | GGUF parsing (2→80 layers), metadata extraction, arch classification, tensor ops, activation codecs (FP8/sparse/raw), shard planning |
| `bitty-protocol` | 2 files | Logits wire encoding, activation tensor encode/decode (4KB–1MB), CRC32 checksum, logits codec |
| `bitty-inference` | 1 file | FakeLayerExecutor forward throughput, logits, dispatch overhead |
| `bitty-coordinator` | 1 file | Halda scheduler scaling (4→256 nodes × 30→120 layers), compute score ranking |
| `bitty-sim` | 1 file | Cluster build time, token streaming throughput (4/8/16 nodes) |
| `bitty-worker` | 1 file | HardwareProfiler wall-clock time, compute score, memory estimation |

### Regression detection

Runtime metrics already exist (Halda duration, worker counters, executor histograms). These Criterion micro-benchmarks complement those metrics for components that are pure data-transform (GGUF parsing, serialization, codec, scheduling). They do not replace runtime metrics for system-level behavior (network I/O, GPU kernels, disk access).
