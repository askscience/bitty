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
