# Architecture Overview

Bitty is a **distributed inference engine** that splits neural network model execution across a peer-to-peer network of heterogeneous nodes. It is designed to allow running large models on modest hardware by pooling resources across machines.

## High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / HTTP API                        │
│                     (bitty-cli / serve)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       Coordinator                            │
│                   (bitty-coordinator)                        │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  Registry   │  │   Halda    │  │  RequestRouter       │   │
│  │  (node mgt) │  │  Scheduler │  │  (batching/routing)  │   │
│  └────────────┘  └────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Worker 1 │◄───►│ Worker 2 │◄───►│ Worker N │  (Ring Topology)
   │ (layers  │     │ (layers  │     │ (layers  │
   │  0-7)    │     │  8-15)   │     │  16-31)  │
   └──────────┘     └──────────┘     └──────────┘
        │                │                │
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Shard 0  │     │ Shard 1  │     │ Shard 2  │  (Weight Shards)
   └──────────┘     └──────────┘     └──────────┘
```

## Core Principles

1. **Heterogeneous support**: Nodes with varying CPU, GPU, RAM, and network capabilities participate proportionally.
2. **Layer-level parallelism**: Model layers are the unit of distribution. Each node executes a contiguous range of layers.
3. **Ring execution**: Activations flow through workers in a ring topology. Each worker receives the previous worker's output, runs its layers, and forwards to the next.
4. **P2P transport**: Encrypted Iroh QUIC connections with NAT traversal. No central network infrastructure required.
5. **Graceful degradation**: The Halda scheduler accounts for node capacity, network latency, and memory budget.

## Crate Dependency Graph

```
bitty-protocol  (foundation types, protobuf, Iroh framing)
       │
       ▼
bitty-model     (GGUF parsing, metadata, quantization, codecs)
       │
       ▼
bitty-inference (executor traits, lifecycle, sampling)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
bitty-coordinator   bitty-worker   bitty-bitnet-runtime
(registry, Halda,     (ring exec,     (GPU inference via
 gRPC server)          profiling)       Candle/wgpu)
       │                                      │
       └──────────┬───────────────────────────┘
                  ▼
          bitty-candle-runtime
       (model loading, tokenizer, KVCache)
                  │
         ┌────────┴────────┐
         ▼                 ▼
   bitty-sim          bitty-cli
   (simulation)       (user CLI)
         │
         ▼
   bitty-observability (metrics, tracing)
```

## Key Concepts

### Node Tiers
Nodes are classified into tiers based on hardware capability:

| Tier | Criteria | Typical Quantization |
|------|----------|---------------------|
| S | GPU with >16GB VRAM | FP16 / Q4 |
| A | GPU with 8-16GB VRAM | Q4 |
| B | GPU <8GB VRAM | Q3 |
| C | CPU with >16GB RAM | Q2 |
| D | CPU with limited RAM | Q2 (minimal layers) |

### Ring Topology
Workers form a directed ring where each worker knows its successor. The coordinator determines:
1. Which layer ranges each worker executes
2. The order of workers in the ring
3. The quantization precision per worker (based on tier)

### Execution Modes
- **Local mode**: Runs entirely on a single machine via `BitNetRuntime` (GPU) or `CpuModel` (CPU)
- **Distributed mode**: Splits layers across multiple machines via the coordinator
- **Simulated mode**: In-process deterministic simulation for testing (`bitty-sim`)
