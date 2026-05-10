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

---

# Data Flow

## Request Lifecycle

### Distributed Mode

```
User CLI                  Coordinator               Worker Ring
    │                          │                        │
    │  1. GenerateRequest      │                        │
    ├─────────────────────────►│                        │
    │                          │  2. Halda.assign()     │
    │                          │     (compute topology) │
    │                          │                        │
    │                          │  3. ForwardActivation   │
    │                          │  (embedding + layers)  │
    │                          ├───────────────────────►│
    │                          │                        │
    │                          │  ┌─── 4. Execute layer │
    │                          │  │    range, forward   │
    │                          │  │    activation ──────►│
    │                          │  │         (ring hop)   │
    │                          │  │  ◄─── activation ───┤
    │                          │  │       (ring back)   │
    │                          │  └── repeat until done │
    │                          │                        │
    │                          │  5. FinalLogits         │
    │                          │◄───────────────────────│
    │                          │                        │
    │                          │  6. SampleToken         │
    │                          ├───────────────────────►│
    │                          │                        │
    │  ◄─── TokenOutput ───────┤                        │
    │  (streaming)             │                        │
    │                          │                        │
    └──────────────────────────┘                        ┘
```

### Detailed Steps

#### 1. Generate Request
- User sends prompt string + generation parameters (temperature, max tokens, context size)
- CLI tokenizes the prompt using the model's tokenizer (HuggingFace tokenizers)
- Sends tokenized prompt to the coordinator via gRPC or Iroh

#### 2. Scheduling (Halda)
- Coordinator maintains a `Registry` of connected workers with their `HardwareProfile`
- On each request (or periodically), the Halda scheduler computes:
  - Layer assignments: which worker gets which layer range
  - Quantization per worker: based on tier (S/A → Q4, B → Q3, C/D → Q2)
  - Ring order: workers sorted by compute score for optimal pipeline
  - Memory budgets: respects per-node VRAM/RAM limits

#### 3. Forward Activation
- The embedding layer runs on the first assigned worker (the one holding layer 0)
- Initial activation tensor is sent with shape `[batch, seq_len, hidden_size]`
- Activation can be compressed using the configured codec (FP8, TopK, Delta)

#### 4. Ring Execution
- Each worker:
  a. Receives activation from previous worker (or coordinator)
  b. Verifies CRC32 checksum
  c. Runs `LayerExecutor::execute_range()` on its assigned layers
  d. Compresses output activation
  e. Forwards to next worker in ring
- The ring is traversed once per token

#### 5. Final Logits
- The last worker (holding LM head) computes logits over the vocabulary
- Returns `BitNetLogits` (f32 vector) to the coordinator

#### 6. Sampling
- Coordinator or final worker samples the next token:
  - `argmax()` if temperature == 0 (greedy)
  - `sample_with_temperature()`: softmax + xorshift64 PRNG
- Token is streamed back to user
- Process repeats from step 4 until:
  - Max tokens reached
  - EOS token generated
  - User stops generation

### Local Mode

```
User CLI → BitNetRuntime / CpuModel
              │
     ┌────────┴────────┐
     │  Embed tokens    │
     │  Forward layers  │
     │  Sample          │
     │  Decode          │
     └─────────────────┘
              │
         Token stream
```

In local mode, the entire model runs on a single machine using either:
- **GPU backend** (Candle/wgpu): `BitNetRuntime` with `SplitBitNetModel`
- **CPU backend**: `CpuModel` with optimized quantized matmul kernels

## Activation Data Flow

```
Input prompt (tokens)
    │
    ▼
┌─────────────────────┐
│  Embedding Layer    │  token IDs → hidden states
└─────────┬───────────┘
          │ activation tensor
          ▼
┌─────────────────────┐
│  Worker (Layers     │  self-attention + FFN / MLP / SSM
│   i..j)             │
│  ┌───────────────┐  │
│  │ LayerExecutor  │  │
│  │ .execute_range()│ │
│  └───────────────┘  │
└─────────┬───────────┘
          │ activation tensor (compressed)
          ▼
    (ring hop via Iroh)
          │
          ▼
┌─────────────────────┐
│  ...more workers...  │  repeat for assigned ranges
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Final Worker       │  LM head → logits
│  (LM Head)          │
└─────────┬───────────┘
          │ logits f32[]
          ▼
┌─────────────────────┐
│  Sampler            │  argmax / softmax + temperature
└─────────┬───────────┘
          │ token_id
          ▼
┌─────────────────────┐
│  Token Decoder      │  token_id → text
└─────────┬───────────┘
          │ "hello"
          ▼
       output
```

## Compression Pipeline

```
Before ring send:
  f16 activation [N, H]
       │
       ▼
  ActivationCodec
  ┌─────────────────┐
  │ FP8 Linear:     │  (sample/256 + 128).clamp(0,255)
  │ Sparse TopK 30% │  keep top 30% by magnitude
  │ Delta (passthru)│  no compression
  └─────────────────┘
       │
       ▼
  packed bytes + CRC32 checksum

After ring receive:
  packed bytes + CRC32 checksum
       │
       ▼
  verify CRC32
  decompress → f16 activation [N, H]
```

---

# Security Architecture

## Authentication Modes

Bitty supports two authentication modes for cluster communication:

### InsecureLocal

```
AuthMode::InsecureLocal
```

- No authentication required
- Intended for local development and testing
- All nodes must be on `127.0.0.1` or `localhost`
- Validated in `endpoint.rs` — rejects non-local addresses

### PreSharedToken

```
AuthMode::PreSharedToken
```

- All nodes share a pre-arranged token
- Token is passed via CLI argument (`--token`) or invite URL
- Comparison uses **constant-time** verification to prevent timing attacks
- Implemented via `security.rs`: `compare_constant_time(a, b)` using XOR-based comparison

## Iroh Transport Security

- All P2P communication uses **Iroh QUIC connections** with built-in encryption (TLS 1.3)
- Node identities are established via Ed25519 keypairs stored in `~/.bitty/iroh-secret.key`
- ALPN protocol negotiation: `bitty/scheduler/0` and `bitty/worker/0`
- NAT traversal via Iroh relay servers (configurable, defaults to public relays)

## Input Validation

The `validation.rs` module provides bounds checking for all wire inputs:

- **Activations**: maximum dimension size (configurable, default 65536)
- **Prompts**: maximum length (configurable, default 65536 tokens)
- **Logits**: maximum vocabulary size (configurable, default 262144)
- **Paths**: maximum length, no path traversal (no `..` segments)

## Rate Limiting

The coordinator uses the `governor` crate for rate limiting:
- **Default**: 100 requests per second per worker
- Configurable via `NetworkConfig`
- Enforced in `network.rs` on the gRPC handler

## Threat Model

| Threat | Mitigation |
|--------|-----------|
| Unauthorized node joining cluster | Pre-shared token with constant-time verification |
| Eavesdropping on P2P traffic | Iroh QUIC TLS 1.3 encryption |
| Malicious activation data | CRC32 checksum verification per tensor |
| Path traversal in model paths | Path validation (no `..`, length limits) |
| Denial of service (DDoS) | Governor rate limiting (100 req/s default) |
| Token theft from logs | Secret redaction in `secrets.rs` |
| Man-in-the-middle on join | Worker-initiated connection to coordinator |

## Secret Management

- Authentication tokens are **never logged** — the `secrets.rs` module redacts token-like strings from log output
- The Iroh secret key is stored at `~/.bitty/iroh-secret.key` with restricted permissions
- Cluster invite URLs are one-shot: they include the token and endpoint, but are intended for ephemeral use
