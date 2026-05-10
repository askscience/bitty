# Bitty Documentation

**Bitty** is a distributed inference engine for running large language models across heterogeneous peer-to-peer networks. It uses GGUF-format models, Iroh-based encrypted P2P transport, and a tiered-ring topology for splitting model execution across multiple machines.

## Documentation

| # | File | What |
|---|---|---|
| 01 | [Installation](01-installation.md) | Prerequisites, quick install, build from source, feature flags, uninstall |
| 02 | [Quickstart](02-quickstart.md) | First run, GPU auto-detect, basic commands |
| 03 | [CLI Reference](03-cli-reference.md) | All `bitty` commands and flags |
| 04 | [Model Management](04-model-management.md) | Registry, pulling, listing, removing models |
| 05 | [GPU Acceleration](05-gpu-acceleration.md) | Auto-detect, Metal/CUDA/Vulkan/DX12 backends, performance, troubleshooting |
| 06 | [Cluster Setup](06-cluster-setup.md) | Distributed inference, node configuration |
| 07 | [Configuration](07-configuration.md) | config.toml, settings, environment variables |
| 08 | [HTTP API](08-http-api.md) | REST API, `/api/generate`, OpenAI-compatible endpoints |
| 09 | [Architecture](09-architecture.md) | System design, data flow, security model |
| 10 | [Protocol](10-protocol.md) | Protobuf definitions, Iroh transport, activation codecs |
| 11 | [Developer Guide](11-developer-guide.md) | Getting started, testing, benchmarking, CI/CD, crate deps |
| 12 | [Reference](12-reference.md) | GGUF format, quantization types, model registry, architecture support |
| 13 | [Crates Reference](13-crates-reference.md) | Per-crate overview of all 11 workspace members |

## Quick Links

- [Installation](01-installation.md)
- [GPU Acceleration](05-gpu-acceleration.md)
- [CLI Reference](03-cli-reference.md)
- [Architecture](09-architecture.md)
