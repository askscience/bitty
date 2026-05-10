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
