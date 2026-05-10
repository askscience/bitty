# bitty-protocol

**Location**: `crates/bitty-protocol/`

**Purpose**: Foundation crate providing shared domain types, protobuf message definitions, Iroh transport framing, CRC32 checksums, endpoint validation, and security primitives. Every other crate depends on this.

## Key Types

### Identity & Topology

| Type | Description |
|------|-------------|
| `NodeId(String)` | Unique node identifier (human-readable label) |
| `NodeTier` | Hardware capability tier: `S`, `A`, `B`, `C`, `D` |
| `ModelStage` | Stage of model assigned: `LayerRange`, `EmbeddingAndLayers`, `FinalLayersAndLmHead` |
| `LayerAssignment` | Full assignment: node, layer range, quantization, weight, ring next-node |
| `TopologyUpdate` | Topology epoch + list of assignments |
| `RingTopology` | Ring ordering of nodes |

### Hardware & Performance

| Type | Description |
|------|-------------|
| `HardwareProfile` | CPU/GPU TFLOPS, memory, network bandwidth, tier. Includes `effective_compute_score()` for ranking. |
| `Quantization` | 9 levels: `F32`, `Fp16`, `Q8`, `Q6`, `Q5`, `Q4`, `Q3`, `Q2`, `Bit1` |
| `CompressionKind` | Activation compression: `None`, `Fp8`, `TopK`, `Delta` |
| `ActivationDType` | Tensor dtype: `Fp16`, `Fp8`, `I8` |
| `LayerMetadata` | Per-layer metrics: weight bytes, activation bytes, FLOPs |

### Wire Messages

| Type | Description |
|------|-------------|
| `ActivationTensor` | Tensor payload with shape, dtype, CRC32, compression |
| `BitNetLogits` | Logits vector with CRC32 |
| `TokenOutput` | Generated token: text, logprobs, latency |
| `GenerateRequest` | Generation request with prompt, params, metadata |
| `Heartbeat` | Node heartbeat with throughput metrics |
| `ShardManifestMessage` | Weight shard descriptor with SHA-256 hash |

### Transport

| Type | Description |
|------|-------------|
| `IrohFrame` | Binary frame: 4-byte length, 1-byte opcode, 2-byte token len, token, payload |
| Opcodes | `Register=1`, `Heartbeat=2`, `Generate=3`, `ForwardActivation=4`, etc. |
| ALPN | `bitty/scheduler/0`, `bitty/worker/0` |

## Modules

| Module | Responsibility |
|--------|---------------|
| `pb` | Generated protobuf code via `tonic::include_proto!` |
| `cli.rs` | CLI argument parsing helpers (port, address, token) |
| `endpoint.rs` | Endpoint normalization, validation, URL parsing |
| `iroh_transport.rs` | Iroh binary frame encode/decode, URI parsing |
| `logits_codec.rs` | f32 ↔ LE byte conversion for wire transmission |
| `registration.rs` | Worker registration protocol version check |
| `security.rs` | `AuthMode` enum, constant-time token comparison |
| `validation.rs` | Bounds checking for activations, prompts, logits, paths |

## Protobuf Services

Defined in `proto/bitty/v1/cluster.proto`:

### CoordinatorService
```protobuf
rpc RegisterWorker(RegisterRequest) → RegisterResponse
rpc Heartbeat(Heartbeat) → HeartbeatAck
rpc Generate(GenerateRequest) → stream TokenOutput
rpc StreamTokens(stream ClientMessage) → stream TokenOutput
rpc ClusterStatus(Empty) → ClusterStatusResponse
rpc ListModels(Empty) → ListModelsResponse
```

### WorkerService
```protobuf
rpc ForwardActivation(ActivationTensor) → ActivationTensor
rpc FinalLogits(ActivationTensor) → BitNetLogits
rpc SampleToken(SampleRequest) → TokenOutput
rpc ApplyTopology(TopologyUpdate) → Empty
rpc LoadShard(ShardManifestMessage) → Empty
rpc Cleanup(Empty) → Empty
```

## Build Process

`build.rs` compiles the protobuf definitions at build time using `tonic-build`. Generated code lives in the `pb` module and is re-exported through the crate's public API.
