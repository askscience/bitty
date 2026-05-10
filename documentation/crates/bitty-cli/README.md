# bitty-cli

**Location**: `crates/bitty-cli/`

**Purpose**: User-facing `bitty` binary providing an Ollama-like CLI for model management, cluster orchestration, and HTTP API serving.

## CLI Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `bitty run <model> [prompt]` | Run a model locally or via cluster |
| `bitty pull <model>` | Download a model from the registry |
| `bitty ls` / `bitty list` | List installed models |
| `bitty show <model>` | Show model details |
| `bitty ps` | Show loaded/running models |
| `bitty stop <model>` | Stop a model or background runtime |
| `bitty start` | Start background runtime |
| `bitty serve` | Start HTTP API server |
| `bitty chat` | Interactive chat session |

### Cluster Commands

| Command | Description |
|---------|-------------|
| `bitty node` | Start a distributed node (leader or join) |
| `bitty cluster` | Cluster management |
| `bitty invite` / `bitty share` | Print cluster invite URL |
| `bitty join` / `bitty connect` | Join a cluster via invite |
| `bitty use <cluster>` | Switch active cluster |
| `bitty clusters` | List saved cluster aliases |
| `bitty status` | Cluster health summary |
| `bitty models` | Browse cluster models |

### Model Management

| Command | Description |
|---------|-------------|
| `bitty create` | Create model profile from Modelfile |
| `bitty rm <model>` | Remove a model |
| `bitty cp <src> <dst>` | Copy/alias a model profile |
| `bitty generate <model> <prompt>` | Generate text via cluster |
| `bitty settings` | Get/set configuration |

### Maintenance

| Command | Description |
|---------|-------------|
| `bitty logs` | View/clear logs |
| `bitty clean` | Remove models/state (keep config) |
| `bitty reset` | Remove everything (fresh start) |
| `bitty setup` | Interactive first-time setup |
| `bitty help` | Show help |
| `bitty version` | Show version |

## Modules

| Module | Responsibility |
|--------|---------------|
| `main.rs` | CLI entry point, 38 subcommand parsing |
| `settings.rs` | `BittySettings` — config.toml management |
| `model_store.rs` | `ModelSpec`, registry parsing, pull/download |
| `server.rs` | HTTP API server (Ollama + OpenAI compatible) |
| `modelfile.rs` | Ollama-style Modelfile parser |
| `cluster_store.rs` | `ClusterStore` — clusters.toml alias management |
| `logger.rs` | Logging with rotation |
| `secrets.rs` | Secret redaction for logging |
| `ui.rs` | Terminal UI helpers (ANSI, spinners, prompts) |

## Configuration

**File**: `~/.bitty/config.toml`

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | `~/.bitty` | Data directory |
| `models_dir` | `~/.bitty/models` | Model cache |
| `default_model` | `bitnet-b1.58` | Default model |
| `api_host` | `127.0.0.1:11435` | HTTP API listen address |
| `auto_pull` | `true` | Auto-download missing models |
| `auto_start_node` | `true` | Auto-start background runtime |
| `default_temperature` | `0.7` | Default temperature |
| `default_num_predict` | `128` | Default max tokens |
| `default_num_ctx` | `2048` | Default context window |
| `iroh_relays` | `public` | Iroh relay config |
| `cluster_mode` | `private` | Cluster visibility |

## HTTP API

### Ollama-Compatible Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/generate` | Generate text |
| POST | `/api/chat` | Chat completion |
| GET | `/api/tags` | List models |
| POST | `/api/show` | Show model details |
| POST | `/api/pull` | Pull a model |

### OpenAI-Compatible Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List models |
| POST | `/v1/chat/completions` | Chat completion |
| POST | `/v1/completions` | Text completion |

## Modelfile Format

Bitty supports Ollama-style Modelfiles for creating custom model profiles:

```dockerfile
FROM bitnet-b1.58
PARAMETER temperature 0.8
PARAMETER num_ctx 4096
SYSTEM "You are a helpful assistant."
TEMPLATE "{{ .Prompt }}"
MESSAGE user "Hello"
LICENSE MIT
```
