# Benchmarking

## Running Benchmarks

Benchmarks use the Criterion.rs framework and are defined in each crate's `benches/` directory.

```bash
# Run all benchmarks
cargo bench

# Run benchmarks for a specific crate
cargo bench -p bitty-model

# Filter benchmarks by name
cargo bench -p bitty-model "GGUF"
```

## Available Benchmarks

| Crate | Benchmarks | What's Measured |
|-------|-----------|-----------------|
| `bitty-model` | 6 | GGUF parsing (2-80 layers), metadata extraction (14 architectures), quantization helpers, tensor ops, activation codecs (4KB-1MB), shard planning |
| `bitty-protocol` | 2 | Logits wire encoding, activation tensor encode/decode, CRC32 |
| `bitty-inference` | 1 | FakeLayerExecutor forward throughput |
| `bitty-coordinator` | 1 | Halda scheduling (4-256 nodes, 30-120 layers) |
| `bitty-sim` | 1 | Cluster build time, token streaming (4/8/16 nodes) |
| `bitty-worker` | 1 | Hardware profiling wall-clock |

## CI Benchmarks

Benchmarks can be triggered manually from GitHub Actions:

```yaml
# .github/workflows/benches.yml
on:
  workflow_dispatch:  # manual trigger only
```

## Interpreting Results

Criterion produces:
- Mean execution time with confidence intervals
- Throughput (elements/sec where applicable)
- Comparison against previous runs (regression detection)
- HTML reports in `target/criterion/`

### Example Output

```
GGUF parsing/8_layers
  time:   [2.3456 ms 2.4567 ms 2.5678 ms]
  thrpt:  [3.11 MiB/s 3.26 MiB/s 3.41 MiB/s]

Halda scheduling/128_nodes_80_layers
  time:   [145.67 µs 148.90 µs 152.34 µs]
```

## Adding a New Benchmark

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_function(c: &mut Criterion) {
    c.bench_function("my_function/input_size", |b| {
        b.iter(|| my_function(black_box(input)))
    });
}

criterion_group!(benches, bench_my_function);
criterion_main!(benches);
```

## Profiling

```bash
# Build with debug symbols
cargo build --release

# Use perf (Linux)
perf record target/release/bitty run ...
perf report

# Use Instruments (macOS)
xcrun xctrace record --template "Time Profiler" \
  --launch target/release/bitty run bitnet-b1.58 "test"
```
