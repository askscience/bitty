# Protocol Buffers

**File**: `proto/bitty/v1/cluster.proto`

## Services

### CoordinatorService

Handled by the coordinator node for worker registration, heartbeats, and client inference requests.

```protobuf
service CoordinatorService {
    // Worker registration with hardware profile
    rpc RegisterWorker(RegisterRequest) returns (RegisterResponse);

    // Periodic health check from workers
    rpc Heartbeat(Heartbeat) returns (HeartbeatAck);

    // Stream text generation from a prompt
    rpc Generate(GenerateRequest) returns (stream TokenOutput);

    // Bidirectional streaming (for future use)
    rpc StreamTokens(stream ClientMessage) returns (stream TokenOutput);

    // Get cluster status
    rpc ClusterStatus(Empty) returns (ClusterStatusResponse);

    // List models available in the cluster
    rpc ListModels(Empty) returns (ListModelsResponse);
}
```

### WorkerService

Handled by worker nodes for activation forwarding and layer execution.

```protobuf
service WorkerService {
    // Forward activation tensor to the next worker
    rpc ForwardActivation(ActivationTensor) returns (ActivationTensor);

    // Compute final logits from last hidden state
    rpc FinalLogits(ActivationTensor) returns (BitNetLogits);

    // Sample next token from logits
    rpc SampleToken(SampleRequest) returns (TokenOutput);

    // Apply new topology assignment
    rpc ApplyTopology(TopologyUpdate) returns (Empty);

    // Load a weight shard for assigned layers
    rpc LoadShard(ShardManifestMessage) returns (Empty);

    // Cleanup and release resources
    rpc Cleanup(Empty) returns (Empty);
}
```

## Message Types

### HardwareProfile

```protobuf
message HardwareProfile {
    string node_id = 1;
    string tier = 2;         // S, A, B, C, D
    float cpu_tflops = 3;
    float gpu_tflops = 4;
    uint64 ram_mb = 5;
    uint64 vram_mb = 6;
    float network_rtt_ms = 7;
    float uplink_mbps = 8;
    string gpu_name = 9;
    uint32 cpu_cores = 10;
    optional float disk_mbps = 11;
}
```

### GenerateRequest

```protobuf
message GenerateRequest {
    string model = 1;
    repeated uint32 prompt_tokens = 2;
    float temperature = 3;
    uint32 max_tokens = 4;
    uint32 num_ctx = 5;
    string request_id = 6;
    map<string, string> metadata = 7;
}
```

### ActivationTensor

```protobuf
message ActivationTensor {
    bytes data = 1;              // packed tensor bytes
    repeated uint64 shape = 2;   // tensor dimensions
    string dtype = 3;            // fp16, fp8, i8
    uint32 checksum = 4;         // CRC32 of data
    CompressionKind compression = 5;
    string request_id = 6;
    uint32 token_count = 7;      // for KV cache management
}
```

### BitNetLogits

```protobuf
message BitNetLogits {
    bytes data = 1;              // f32 logits, little-endian
    uint32 checksum = 2;         // CRC32 of data
    uint32 vocab_size = 3;
    string request_id = 4;
}
```

### TokenOutput

```protobuf
message TokenOutput {
    uint32 token_id = 1;
    string text = 2;
    float logprob = 3;
    bool done = 4;
    uint64 latency_us = 5;
    optional string finish_reason = 6;  // "stop", "length"
}
```

### TopologyUpdate

```protobuf
message TopologyUpdate {
    uint64 epoch = 1;
    repeated LayerAssignment assignments = 2;
    uint32 total_layers = 3;
}
```

### LayerAssignment

```protobuf
message LayerAssignment {
    string node_id = 1;
    uint32 start_layer = 2;
    uint32 end_layer = 3;       // exclusive
    string quantization = 4;    // F32, Fp16, Q8, Q6, Q5, Q4, Q3, Q2, Bit1
    string next_node_id = 5;    // next node in ring
    bool owns_embedding = 6;
    bool owns_lm_head = 7;
}
```

### Heartbeat

```protobuf
message Heartbeat {
    string node_id = 1;
    uint64 tokens_generated = 2;
    float avg_latency_us = 3;
    uint64 memory_used_mb = 4;
    uint64 uptime_seconds = 5;
}
```

### ShardManifestMessage

```protobuf
message ShardManifestMessage {
    string node_id = 1;
    uint32 start_layer = 2;
    uint32 end_layer = 3;
    string shard_hash = 4;       // SHA-256 of shard data
    uint64 shard_size = 5;
    string model_name = 6;
}
```

## Transport

The protobuf messages are serialized and transmitted over:

1. **gRPC (tonic)**: Standard HTTP/2 gRPC for coordinator-worker communication. Uses `proto`-generated service stubs.

2. **Iroh binary frames** (for P2P): Protobuf messages are wrapped in `IrohFrame`:
   ```
   [4-byte payload length][1-byte opcode][2-byte token length][token][protobuf payload]
   ```

## Opcode Mapping

| Opcode | RPC | Direction |
|--------|-----|-----------|
| 1 | Register | Worker → Coordinator |
| 2 | Heartbeat | Worker → Coordinator |
| 3 | Generate | Coordinator → Worker |
| 4 | ForwardActivation | Worker → Worker |
| 5 | FinalLogits | Worker → Coordinator |
| 6 | SampleToken | Coordinator → Worker |
| 7 | ApplyTopology | Coordinator → Worker |
| 8 | LoadShard | Coordinator → Worker |
| 9 | Cleanup | Coordinator → Worker |
| 10 | StreamTokens | Bidirectional |

---

# Iroh Transport

Bitty uses [Iroh](https://iroh.computer) for encrypted peer-to-peer transport between cluster nodes. Iroh provides QUIC-based connections with built-in NAT traversal via relay servers.

## ALPN Protocol IDs

```
Coordinator: bitty/scheduler/0
Worker:      bitty/worker/0
```

## Binary Frame Protocol

Protobuf messages are wrapped in a custom binary frame format on top of Iroh bidirectional streams.

### Frame Format

```
┌──────────────────────────────────────────────────┐
│ 4 bytes: Payload Length (big-endian u32)         │
├──────────────────────────────────────────────────┤
│ 1 byte:  Opcode                                  │
├──────────────────────────────────────────────────┤
│ 2 bytes: Token Length (big-endian u16)           │
├──────────────────────────────────────────────────┤
│ Token Length bytes: Auth Token (UTF-8)          │
├──────────────────────────────────────────────────┤
│ Payload Length bytes: Protobuf Payload           │
└──────────────────────────────────────────────────┘
```

### Opcodes

| Opcode | RPC | Description |
|--------|-----|-------------|
| 1 | Register | Worker registers with coordinator |
| 2 | Heartbeat | Worker sends health metrics |
| 3 | Generate | Start text generation |
| 4 | ForwardActivation | Forward tensor to next worker |
| 5 | FinalLogits | Return logits to coordinator |
| 6 | SampleToken | Sample next token |
| 7 | ApplyTopology | Apply new layer assignments |
| 8 | LoadShard | Load weight shard |
| 9 | Cleanup | Free resources |

## URI Format

Cluster invite URIs use the following format:

```
iroh://<endpoint_id>?token=<auth_token>&relay=<relay_url>&addr=<socket_addr>
```

Example:
```
iroh://abc123def456?token=my-secret-token&relay=https://relay.iroh.network&addr=192.168.1.100:50051
```

## Connection Lifecycle

```
Client (worker)                    Server (coordinator)
     │                                    │
     │  1. Resolve relay URL              │
     │  2. Establish QUIC connection      │
     ├───────────────────────────────────►│
     │  3. ALPN negotiation               │
     │     (bitty/scheduler/0)            │
     │                                    │
     │  4. Open bi-directional stream     │
     ├───────────────────────────────────►│
     │  5. Send Register frame            │
     │     (opcode=1, token, protobuf)    │
     │                                    │
     │  6. Receive RegisterResponse       │
     │◄───────────────────────────────────┤
     │                                    │
     │  7. Periodic Heartbeat frames      │
     ├───────────────────────────────────►│
     │                                    │
     │  8. Receive ForwardActivation      │
     │◄───────────────────────────────────┤
     │  9. Execute, send response         │
     ├───────────────────────────────────►│
     │                                    │
```

## Relay Configuration

- **Default**: `public` — uses Iroh's public relay servers
- **Custom**: Set `iroh_relays` in config to a relay URL
- **LAN only**: Workers on the same local network connect directly via QUIC, bypassing relays

## NAT Traversal

1. Iroh uses STUN-like mechanisms for NAT type detection
2. Relays are used as fallback when direct connections fail
3. Connections are encrypted end-to-end regardless of relay usage

## Identity

- Each node generates an Ed25519 keypair stored at `~/.bitty/iroh-secret.key`
- The public key serves as the node's Iroh endpoint ID
- Keys are persisted across restarts for stable identity

---

# Activation Codecs

Activation codecs compress intermediate tensor data sent between workers in the ring, reducing network bandwidth at the cost of some precision loss.

## Interface

```rust
pub trait ActivationCodec: Send + Sync {
    fn compress(&self, tensor: &ActivationTensor) -> Result<ActivationTensor>;
    fn decompress(&self, tensor: &ActivationTensor) -> Result<ActivationTensor>;
    fn kind(&self) -> CompressionKind;
}
```

## Available Codecs

### FP8 Linear Compression

**Compression ratio**: ~2:1 (16-bit → 8-bit)

Converts each fp16 value to u8 using linear quantization:

```
compressed = clamp(sample / 256 + 128, 0, 255)
decompressed = (value - 128) * 256
```

- Simple and fast (no search/sort)
- Preserves dynamic range reasonably for LLM activations
- Introduces ~0.39% quantization error on average

### Sparse TopK (30%)

**Compression ratio**: ~3.3:1 on average

1. Compute magnitude of each element
2. Keep only the top 30% of elements by magnitude
3. Store as (index: u32, value: f16) pairs
4. Zero out all other elements

```
Input:  [0.5, -2.3, 0.1, 4.2, -0.8, 1.1, ...]  (1280 elements)
Keep top 30% (384 elements):
Output: [(3, 4.2), (1, -2.3), (5, 1.1), ...]     (384 index-value pairs)
```

- Higher compression than FP8
- Non-deterministic output (depends on sparsity pattern)
- May lose information in dense activations

### Delta

**Compression ratio**: 1:1 (passthrough)

No compression — data is sent as-is with only the compression flag set. Useful for:
- Debugging and testing
- When network bandwidth is not a concern
- Layers where precision is critical (embedding, LM head)

## Selection Strategy

The compression codec is chosen per-layer-range by the coordinator based on:
1. **Node tier**: Lower-tier nodes get higher compression (Delta → FP8 → TopK)
2. **Layer position**: Embedding and LM head use Delta (no compression)
3. **Network bandwidth**: Low-bandwidth links use TopK
4. **Experimental**: Configurable per-deployment

## Usage in Ring

```rust
// Worker sends compressed activation
let codec = ActivationCodec::new(CompressionKind::Fp8);
let compressed = codec.compress(&activation)?;

// Send over Iroh
frame.send(opcode, token, &compressed).await?;

// Next worker receives and decompresses
let codec = ActivationCodec::new(compressed.compression);
let decompressed = codec.decompress(&compressed)?;

// Execute layers with decompressed activation
let output = executor.execute_range(decompressed).await?;
```

## CRC32 Checksums

Every activation tensor includes a CRC32 checksum of the packed data. The receiving worker verifies the checksum before decompression to detect corruption in transit. Checksum failures increment the `dlm_checksum_failures_total` metric.
