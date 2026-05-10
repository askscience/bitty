# Configuration

## Config File

**Location**: `~/.bitty/config.toml` (or `$BITTY_DATA_DIR/config.toml`)

### All Settings

```toml
# Data directory (default: ~/.bitty)
data_dir = "/home/user/.bitty"

# Model cache directory (default: ~/.bitty/models)
models_dir = "/home/user/.bitty/models"

# Default model identifier (default: "bitnet-b1.58")
default_model = "bitnet-b1.58"

# HTTP API listen address (default: "127.0.0.1:11435")
api_host = "127.0.0.1:11435"

# Auto-download missing models when running (default: true)
auto_pull = true

# Auto-start background runtime (default: true)
auto_start_node = true

# Default sampling temperature (default: 0.7)
default_temperature = 0.7

# Default max tokens to generate (default: 128)
default_num_predict = 128

# Default context window size (default: 2048)
default_num_ctx = 2048

# Iroh relay configuration (default: "public")
iroh_relays = "public"

# Cluster visibility mode (default: "private")
cluster_mode = "private"

# Active cluster target (default: "")
active_cluster = "my-cluster"
```

## Environment Variables

### Hardware Profiling Overrides

These override automatic hardware detection in the profiler:

| Variable | Description |
|----------|-------------|
| `BITTY_RAM_MB` | System RAM in MB |
| `BITTY_VRAM_MB` | GPU VRAM in MB |
| `BITTY_GPU_NAME` | GPU device name string |
| `BITTY_GPU_TFLOPS` | GPU TFLOPS (FP16) |
| `BITTY_BACKEND` | Inference backend override |
| `BITTY_MAX_LAYERS` | Max layers this node can run |
| `BITTY_NODE_ROLE` | Node role override |
| `BITTY_NETWORK_RTT_MS` | Network round-trip time in ms |
| `BITTY_UPLINK_MBPS` | Network uplink speed in Mbps |
| `BITTY_DISK_MBPS` | Disk I/O speed in MB/s |

### Runtime Configuration

| Variable | Description |
|----------|-------------|
| `BITTY_WORKER_ENDPOINT` | Worker gRPC endpoint |
| `BITTY_MODEL_PATH` | Override model file path |
| `BITTY_DATA_DIR` | Override data directory |
| `BITTY_DISABLE_MODEL_LAYERS` | Disable layer distribution (local mode) |

### Logging

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Tracing log level (e.g., `info`, `debug`, `bitty=debug`) |
| `BITTY_LOG_DIR` | Log directory (default: `~/.bitty/logs`) |

## Example: CPU-Only Setup

```toml
default_num_ctx = 1024
default_num_predict = 64
```

```bash
export BITTY_BACKEND=cpu
export BITTY_RAM_MB=32000
```

## Example: Multi-GPU Cluster Node

```toml
default_temperature = 0.8
auto_start_node = true
```

```bash
export BITTY_GPU_TFLOPS=20.5
export BITTY_VRAM_MB=24000
export BITTY_GPU_NAME="NVIDIA RTX 4090"
export BITTY_UPLINK_MBPS=1000
export BITTY_NETWORK_RTT_MS=5
```
