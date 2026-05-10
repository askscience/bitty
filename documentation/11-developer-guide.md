# Developer Guide

## Prerequisites

- Rust 1.78+ (`rustup install 1.78`)
- `protoc` (Protocol Buffers compiler) — for `bitty-protocol/build.rs`
- Cargo toolchain with `clippy`, `rustfmt`, `llvm-tools`
- Optional: `cargo-audit`, `cargo-tarpaulin` (for coverage)

## Setup

```bash
# Clone
git clone https://github.com/askscience/bitty.git
cd bitty

# Build all crates
cargo build --release

# Run all tests
cargo test --workspace

# Run lints (must pass CI)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Project Structure

```
crates/
├── bitty-protocol/        # Foundation: types, protobuf, Iroh framing
├── bitty-model/           # GGUF parsing, metadata, codecs
├── bitty-inference/       # Executor traits, sampling, lifecycle
├── bitty-coordinator/     # Halda scheduler, gRPC server, registry
├── bitty-worker/          # Ring execution, profiling, shard store
├── bitty-bitnet-runtime/  # BitNet GPU + CPU inference
├── bitty-candle-runtime/  # Candle model loading, tokenizer
├── bitty-sim/             # In-process cluster simulation
├── bitty-cli/             # User-facing CLI binary
└── bitty-observability/   # Metrics & tracing
```

## Development Workflow

### Adding a new crate

1. Create `crates/<name>/Cargo.toml`
2. Add to workspace in root `Cargo.toml`
3. Create `src/lib.rs` with the public API
4. Add dependencies (use workspace dependencies where possible)

### Modifying protobuf definitions

1. Edit `proto/bitty/v1/cluster.proto`
2. Rebuild: `cargo build -p bitty-protocol`
3. Update domain types in `bitty-protocol/src/` if needed
4. Regenerate RPC implementations in coordinator/worker

### Adding a new model architecture

1. Add architecture variant to `ModelArchitecture` enum in `bitty-model`
2. Add GGUF key mappings in `model_metadata.rs`
3. Add tensor name patterns in CPU backend `names.rs`
4. Add layer support in `cpu_backend/layers/` if needed
5. Add to `layers/mod.rs` dispatch
6. Add test fixture to `model_metadata.rs` tests

## Code Style

- Follow Rust 2021 edition idioms
- Use `cargo fmt` (no configuration deviations)
- Clippy must pass with `-D warnings`
- All public items must have doc comments
- Use `tracing` for logging, not `println!`
- Error types should implement `std::error::Error`
- Prefer `thiserror` for error enums

## Key Patterns

### Async trait pattern
```rust
#[async_trait]
pub trait LayerExecutor: Send + Sync {
    async fn execute_range(...) -> Result<...>;
}
```

### Error handling
```rust
#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("worker {0} not found")]
    WorkerNotFound(NodeId),
    #[error("network error: {0}")]
    Network(#[from] tonic::Status),
}
```

### Configuration
```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub default_temperature: f64,
    pub default_num_ctx: usize,
}
```

---

# Testing

## Running Tests

```bash
# All tests
cargo test --workspace

# Single crate
cargo test -p bitty-coordinator

# Single test
cargo test -p bitty-coordinator test_halda_coverage

# With output
cargo test -- --nocapture
```

## Test Categories

### Unit Tests

Embedded in each crate alongside the code. Examples:

- **bitty-protocol**: Checksum detection, proto round-trips, endpoint validation, frame encoding, Iroh URI parsing, auth mode comparison
- **bitty-model**: i2_s decode, layer ID parsing, architecture classification, quantization derivation, codec round-trips, shard verification
- **bitty-coordinator**: Halda coverage (proptest with 1-20 nodes, 1-80 layers), critical layer preservation, weak CPU filtering
- **bitty-worker**: RAM override profiling, token auth, SHA-256 verification
- **bitty-sim**: Cluster ring execution, chaos drop, token streaming
- **bitty-cli**: Settings get/set, registry parsing, server models, modelfile parsing, secret redaction

### Integration Tests (ignored by default)

Some tests require a real GGUF model file:

```rust
#[ignore]
#[test]
fn split_local_logits_match_full_local_logits_for_temperature_zero() {
    // Requires BITTY_GGUF_MODEL env var
}
```

Run with:
```bash
BITTY_GGUF_MODEL=/path/to/model.gguf cargo test -- --ignored
```

### Deterministic Simulation Tests

The `bitty-sim` crate provides fully deterministic cluster simulation:

```rust
#[test]
fn test_ring_execution() {
    let profiles = demo_profiles(4);
    let layers = demo_layers(32);
    let mut cluster = SimulatedCluster::new(&profiles, &layers);
    let report = cluster.run_tokens(10);
    assert!(report.checksum_ok);
}
```

### Property-Based Tests

Halda scheduler uses `proptest` for property-based testing:

```rust
proptest! {
    #[test]
    fn test_halda_coverage(
        num_nodes in 1..20usize,
        num_layers in 1..80usize,
    ) {
        let profiles = demo_profiles(num_nodes);
        let layers = demo_layers(num_layers);
        let config = SchedulerConfig::default();
        let assignments = Halda::assign(&profiles, &layers, &config);
        prop_assert!(assignments.is_ok());
    }
}
```

## Writing Tests

### Test Conventions

- Tests live in the same file as the code, in a `#[cfg(test)] mod tests` block
- Integration tests go in `tests/` directory at crate root
- Use `#[ignore]` for tests requiring external model weights
- Use `proptest` for property-based tests of algorithms

### Adding a Mock Executor

```rust
struct MyMockExecutor;

#[async_trait]
impl LayerExecutor for MyMockExecutor {
    async fn execute_range(&self, ...) -> Result<ActivationTensor, ExecutorError> {
        Ok(activation)  // identity passthrough
    }
    // ...
}
```

### Test Fixtures

Reusable test fixtures are available:

- `bitty_sim::demo_profiles(n)` — creates n hardware profiles
- `bitty_sim::demo_layers(n)` — creates n layer metadata entries
- `FakeLayerExecutor` — identity passthrough executor
- `StubLayerExecutor` — always-errors executor

## Code Coverage

```bash
# Install coverage tool
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --workspace --out Html
```

---

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

---

# CI/CD

## CI Pipeline

Defined in `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rust-lang/setup-rust-toolchain@v1
      - run: cargo fmt --all -- --check
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: cargo test --workspace
      - run: cargo audit
```

### Stages

1. **Format check**: `cargo fmt --all -- --check`
   - Ensures consistent code formatting
   - Must pass before merging

2. **Clippy**: `cargo clippy --workspace --all-targets -- -D warnings`
   - All targets include tests, benchmarks, and binaries
   - Warnings are treated as errors (`-D warnings`)

3. **Tests**: `cargo test --workspace`
   - Runs all unit tests across all crates
   - Does NOT include ignored tests (those requiring model weights)

4. **Audit**: `cargo audit`
   - Checks for known security vulnerabilities in dependencies
   - Fails if any advisory is found

## Benchmark CI

Defined in `.github/workflows/benches.yml`:

```yaml
name: Benchmarks
on:
  workflow_dispatch:  # manual trigger only
```

- Manual trigger only (not on every push)
- Runs all 12 Criterion benchmarks
- Results are commented on the triggering PR

## Local Pre-merge Checklist

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo audit
cargo bench  # optional, for performance-sensitive changes
```

## Release Process

1. Bump version in all `Cargo.toml` files
2. Update `CHANGELOG.md` (if maintained)
3. Run full CI locally
4. Create a GitHub release
5. Tag with version number (e.g., `v0.1.0`)

## Adding Dependencies

```bash
# Add to workspace Cargo.toml
[workspace.dependencies]
my-crate = "0.1"

# Use in crate Cargo.toml
[dependencies]
my-crate.workspace = true
```

Run `cargo audit` to verify no vulnerabilities are introduced.

---

# Crate Dependencies

## Dependency Graph

```
bitty-protocol  (standalone — no internal deps)
       │
       ▼
bitty-model     (depends on: bitty-protocol)
       │
       ▼
bitty-inference (depends on: bitty-protocol, bitty-model)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
bitty-coordinator   bitty-worker   bitty-bitnet-runtime
(deps: protocol,     (deps: protocol,  (deps: protocol,
 model, inference,    model, inference,  model, inference,
 observability)       observability)     candle-runtime)
       │                                      │
       └──────────┬───────────────────────────┘
                  ▼
          bitty-candle-runtime
       (depends on: bitty-model)
                  │
         ┌────────┴────────┐
         ▼                 ▼
   bitty-sim          bitty-cli
   (deps: protocol,   (deps: protocol,
    model, inference,   model, inference,
    coordinator,        coordinator,
    worker)             server libs)
         │
         ▼
   bitty-observability  (standalone)
```

## External Dependencies by Crate

### bitty-protocol
- `tonic` 0.12, `prost` 0.13 — gRPC framework
- `iroh` 0.98.2 — P2P networking
- `serde`, `serde_json` — serialization
- `uuid` — request IDs
- `crc32fast` — checksums
- `tokio` — async runtime
- `tracing` — logging

### bitty-model
- `memmap2` 0.9 — memory-mapped GGUF parsing
- `sha2` — SHA-256 for shard verification
- `rayon` — parallel decoding
- `crc32fast` — checksums

### bitty-bitnet-runtime
- `candle-core` 0.10.2 — GPU tensor ops
- `wgpu` 26 — GPU abstraction for profiling
- `tokenizers` 0.23 — HF tokenizer
- `half` — f16 type

### bitty-candle-runtime
- `candle-core` 0.10.2
- `tokenizers` 0.23
- `half` — f16 type

### bitty-coordinator
- `governor` 0.6 — rate limiting
- `bincode` — snapshot serialization
- `proptest` — property-based testing (dev)

### bitty-worker
- `sysinfo` 0.36 — system introspection
- `nvml-wrapper` 0.12 — NVIDIA GPU detection
- `wgpu` 26 — GPU adapter detection

### bitty-cli
- `clap` 4 — CLI argument parsing
- `reqwest` — HTTP downloads
- `toml` — config file parsing
- `tower-http` — HTTP server infrastructure

### bitty-observability
- `metrics` 0.24 — metrics framework
- `metrics-exporter-prometheus` 0.16
- `tracing-subscriber` — log routing

## Internal Dependency Rules

- `bitty-protocol` must NOT depend on any other bitty crate
- `bitty-model` must NOT depend on `bitty-inference` or higher crates
- `bitty-cli` may depend on any crate
- `bitty-observability` is standalone
- Cyclic dependencies are forbidden
