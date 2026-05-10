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
