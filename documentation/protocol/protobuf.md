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
