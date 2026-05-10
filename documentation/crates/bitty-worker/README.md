# bitty-worker

**Location**: `crates/bitty-worker/`

**Purpose**: Worker node runtime — executes assigned layer ranges in the inference ring, profiles local hardware, stores weight shards, and reports metrics.

## Modules

| Module | Responsibility |
|--------|---------------|
| `ring.rs` | `RingWorker<E: LayerExecutor>` — ring execution with checksum verification |
| `network.rs` | `NetworkWorker<E>` — gRPC `WorkerService` implementation |
| `profiler.rs` | `HardwareProfiler` — system introspection |
| `shard.rs` | `ShardStore` — weight shard storage |
| `keepalive.rs` | `touch_weights()` — keep model resident in memory |
| `metrics.rs` | Prometheus metrics registration |

## Ring Execution

The `RingWorker<E>` wraps any `LayerExecutor` implementation and handles the ring protocol:

```
Receive activation (from coordinator or previous worker)
    │
    ▼
Verify CRC32 checksum
    │
    ▼
Decompress activation
    │
    ▼
Execute assigned layers via LayerExecutor::execute_range()
    │
    ▼
Compress output activation
    │
    ▼
Forward to next node in ring
```

### Key methods:
- `execute_activation()` — processes a single activation through the assigned range
- `compute_logits()` — runs LM head, returns logits to coordinator
- `sample_token()` — samples next token from logits

## Hardware Profiler

The `HardwareProfiler` introspects the local system to build a `HardwareProfile`:

| Metric | Source | Override |
|--------|--------|----------|
| CPU count | `sysinfo` | `BITTY_*` env vars |
| CPU TFLOPS | Estimated from frequency + cores | `BITTY_CPU_TFLOPS` |
| RAM | `sysinfo` | `BITTY_RAM_MB` |
| GPU detection | NVML (NVIDIA) + wgpu | `BITTY_GPU_NAME`, `BITTY_GPU_TFLOPS` |
| GPU VRAM | NVML / wgpu | `BITTY_VRAM_MB` |
| Network RTT | Estimated | `BITTY_NETWORK_RTT_MS` |
| Uplink speed | Estimated | `BITTY_UPLINK_MBPS` |
| Tier classification | Derived from above | `BITTY_NODE_ROLE` |

### Tier Classification Logic

```
if gpu_tflops >= 20 && vram_mb >= 16000 → Tier S
if gpu_tflops >= 10 && vram_mb >= 8000  → Tier A
if gpu_present                          → Tier B
if cpu_tflops >= 1 && ram_mb >= 16000  → Tier C
else                                    → Tier D
```

## Shard Store

- `ShardStore` manages weight shards in memory and via memory-mapped files
- Weights are verified by SHA-256 before use
- `load_shard()` — loads a `WeightShardManifest`, fetches and verifies weight data
- `cleanup()` — releases all shard resources

## Keepalive

`touch_weights()` periodically runs a tiny activation through the model to keep weights resident in GPU/CPU memory, preventing OS page-out or GPU memory reclamation.

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `dlm_layer_latency_us` | Histogram | Per-layer execution latency |
| `dlm_activation_bytes_total` | Counter | Total activation bytes transferred |
| `dlm_checksum_failures_total` | Counter | CRC32 mismatch count |
| `dlm_tokens_generated_total` | Counter | Tokens produced |

## CLI

```
bitty-worker --node-id ID --listen ADDR --coordinator HOST:PORT --model PATH --token TOKEN
```

| Flag | Description |
|------|-------------|
| `--node-id` | Unique human-readable node name |
| `--listen` | gRPC listen address |
| `--coordinator` | Coordinator endpoint to register with |
| `--model` | Path to GGUF model file |
| `--token` | Shared auth token |
