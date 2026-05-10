# Cluster Setup

## Architecture

A Bitty cluster consists of:
- **1 Coordinator**: Runs the Halda scheduler and manages the ring
- **N Workers**: Each holds a shard of the model and executes assigned layers
- **Clients**: Connect to the coordinator via gRPC or HTTP API

## Quick Start: 2-Node Cluster

### Node 1 (Coordinator + Worker)

```bash
# Start the coordinator
bitty node --leader --listen 0.0.0.0:50051

# This generates an invite URL like:
# iroh://<id>?token=<secret>&relay=...
```

### Node 2 (Worker)

```bash
# Join the cluster using the invite URL from Node 1
bitty join iroh://<id>?token=<secret>&relay=<relay>&addr=<addr>
```

### Run Inference

```bash
# From any node or external client
bitty run bitnet-b1.58 "Hello, cluster!"

# Or via the HTTP API
curl http://<coordinator>:11435/api/generate \
  -d '{"model":"bitnet-b1.58","prompt":"Hello, cluster!"}'
```

## Multi-Node Setup

### Step 1: Start the Coordinator

```bash
bitty node --leader \
  --listen 0.0.0.0:50051 \
  --model ~/.bitty/models/llama3.2:1b/model.gguf
```

The coordinator:
- Listens for worker registrations on port 50051 (gRPC)
- Runs the Halda scheduler
- Manages the ring topology
- Exposes the HTTP API on port 11435

### Step 2: Start Workers

On each worker machine:

```bash
bitty node --join <invite-url>
```

Or manually:

```bash
bitty-worker \
  --node-id "worker-1" \
  --listen 0.0.0.0:50052 \
  --coordinator <coordinator-ip>:50051 \
  --model ~/.bitty/models/llama3.2:1b/model.gguf \
  --token <shared-token>
```

### Step 3: Verify Cluster

```bash
bitty cluster status
bitty cluster nodes
```

## Hardware Requirements

### Coordinator
- Moderate CPU, minimal RAM
- Good network connectivity (low latency to all workers)
- No GPU required

### Workers
- Each worker needs enough RAM/VRAM for its assigned layer shard
- Workers can be heterogeneous (different GPU/CPU/network capabilities)
- The Halda scheduler automatically distributes layers proportionally

## Network Topology

```
                    ┌──────────────┐
                    │  Coordinator  │
                    │  (port 50051) │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ Worker 1│◄─────►│ Worker 2│◄─────►│ Worker N│
   │(GPU, 24G)│      │(CPU, 32G)│      │(GPU, 8G) │
   └─────────┘       └─────────┘       └─────────┘
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │Layers 0-7│      │Layers 8-15│     │Layers 16-23│
   │ Q4       │      │ Q2       │      │ Q3        │
   └─────────┘       └─────────┘       └─────────┘
```

## Environment Variables for Cluster

```bash
# Override hardware detection (useful for testing)
export BITTY_NETWORK_RTT_MS=10
export BITTY_UPLINK_MBPS=1000
export BITTY_NODE_ROLE=A

# Override endpoint
export BITTY_WORKER_ENDPOINT=192.168.1.100:50052
```

## Troubleshooting

### Workers not registering
- Check firewall rules (gRPC port 50051)
- Verify shared token matches
- Check coordinator is reachable

### Slow inference
- Check network latency between nodes
- Consider changing quantization tier
- Verify each worker has enough memory

### Worker dropped
- Check worker logs: `bitty logs`
- Workers are evicted after 30s without heartbeat
- Coordinator will re-schedule when nodes reconnect

### Security
- Use a strong random token for production
- Tokens are compared in constant time
- All P2P traffic is encrypted via Iroh
- Set `cluster_mode = "private"` in config
