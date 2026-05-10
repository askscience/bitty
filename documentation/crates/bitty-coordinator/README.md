# bitty-coordinator

**Location**: `crates/bitty-coordinator/`

**Purpose**: Central orchestration node — maintains the worker registry, runs the Halda scheduler for layer assignment, implements the gRPC coordinator service, and manages request routing with batching.

## Modules

| Module | Responsibility |
|--------|---------------|
| `scheduler/mod.rs` | Re-exports Halda |
| `scheduler/halda.rs` | **Halda scheduler** — core assignment algorithm |
| `registry.rs` | Worker registry with health tracking |
| `topology.rs` | Ring topology management |
| `network.rs` | gRPC `CoordinatorService` implementation |
| `router.rs` | `RequestRouter` with batching |
| `snapshot.rs` | Coordinator state serialization (bincode) |
| `security.rs` | Rate limiter re-export |
| `kv_index.rs` | `KvIndex` — prefix cache key → node ownership with TTL |
| `worker_client.rs` | `WorkerRpcClient` enum (TCP/gRPC or Iroh) |

## Halda Scheduler

### Algorithm

1. **Score computation**: Rank nodes by `effective_compute_score()`:
   ```
   score = (gpu_tflops OR cpu_tflops * 0.2) * sqrt(memory_bandwidth)
           / (rtt_penalty * uplink_penalty)
   ```

2. **Filtering**: Remove nodes with score < 15% of the strongest node's score (except GPU nodes, which are always kept)

3. **Layer distribution**: Distribute layers proportionally to each node's share of total compute score:
   ```
   node_layers = round(total_layers * node_score / total_score)
   ```

4. **Quantization assignment**:
   | Tier | Quantization |
   |------|-------------|
   | S / A | Q4 |
   | B | Q3 |
   | C / D | Q2 |
   | Critical layers (embed + lm_head) | Fp16 |

5. **Ring linking**: After assigning layer ranges, link nodes into a ring via `next_node_id`

6. **Validation**: Ensure all layers are covered with no double assignments

### Configuration

```rust
pub struct SchedulerConfig {
    pub memory_reserve_fraction: f32,     // default 0.15 (15%)
    pub critical_quantization: Quantization,  // default Fp16
    pub weak_node_threshold: f32,         // default 0.15 (15% of strongest)
}
```

## Registry

- Maintains `HashMap<NodeId, RegisteredNode>`
- Each `RegisteredNode` has:
  - `HardwareProfile` (static, from registration)
  - `Heartbeat` (dynamic, updated periodically)
  - `NodeHealth` status: `Healthy`, `Degraded`, `MissingHeartbeat`
- Heartbeat timeout: configurable (default 30s)
- Missing nodes are evicted after timeout

## Network Coordinator

Implements the `CoordinatorService` gRPC service:

| RPC | Description |
|-----|-------------|
| `RegisterWorker` | Accepts worker registration, validates protocol version |
| `Heartbeat` | Updates worker health, returns pending topology changes |
| `Generate` | **Core RPC** — streaming text generation |
| `ClusterStatus` | Returns current cluster state |
| `ListModels` | Lists models available in the cluster |

### Generate Flow
1. Receives `GenerateRequest` with prompt tokens
2. Runs Halda to get/refresh topology
3. Sends `ApplyTopology` to workers
4. Sends embedding activation to first worker
5. Receives logits from last worker
6. Samples token (or delegates sampling)
7. Streams `TokenOutput` back to caller
8. Repeats for autoregressive generation

## Request Router

- Buffers incoming requests for up to 50ms (max)
- Batch size: up to 16 requests
- Prefix caching via `KvIndex` with TTL
- Routes batched requests through the worker ring efficiently

## CLI

```
bitty-coordinator --nodes N --layers N --listen ADDR --model PATH --token TOKEN
```

| Flag | Description |
|------|-------------|
| `--nodes` | Expected number of worker nodes |
| `--layers` | Total model layers |
| `--listen` | gRPC listen address (e.g., `0.0.0.0:50051`) |
| `--model` | Path to GGUF model file |
| `--token` | Shared auth token |
