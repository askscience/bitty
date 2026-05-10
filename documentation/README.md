# Bitty Documentation

**Bitty** is an experimental distributed inference engine for running large language models across heterogeneous peer-to-peer networks. It uses GGUF-format models, Iroh-based encrypted P2P transport, and a tiered-ring topology for splitting model execution across multiple machines.

## Documentation Structure

```
documentation/
├── architecture/          # System architecture & design
│   ├── overview.md        # High-level architecture
│   ├── data-flow.md       # Request lifecycle & data flow
│   └── security.md        # Auth, encryption, threat model
├── crates/                # Per-crate deep dives
│   ├── bitty-protocol/
│   ├── bitty-model/
│   ├── bitty-inference/
│   ├── bitty-coordinator/
│   ├── bitty-worker/
│   ├── bitty-bitnet-runtime/
│   ├── bitty-candle-runtime/
│   ├── bitty-sim/
│   ├── bitty-cli/
│   └── bitty-observability/
├── user-guide/            # End-user documentation
│   ├── installation.md
│   ├── cli-reference.md
│   ├── configuration.md
│   ├── model-management.md
│   ├── cluster-setup.md
│   └── http-api.md
├── developer-guide/       # Contributor documentation
│   ├── getting-started.md
│   ├── crate-dependencies.md
│   ├── testing.md
│   ├── benchmarking.md
│   └── ci-cd.md
├── protocol/              # Wire protocol & networking
│   ├── protobuf.md
│   ├── iroh-transport.md
│   └── activation-codecs.md
└── reference/             # Reference material
    ├── model-registry.md
    ├── gguf-format.md
    ├── quantization-types.md
    └── architecture-support.md
```

## Quick Links

- [Architecture Overview](architecture/overview.md)
- [CLI Reference](user-guide/cli-reference.md)
- [Contributing Guide](developer-guide/getting-started.md)
- [Protocol Definition](protocol/protobuf.md)
