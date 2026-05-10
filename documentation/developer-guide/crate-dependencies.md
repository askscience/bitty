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
