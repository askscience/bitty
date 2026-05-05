# Bitty

Bitty is an experimental Rust project for testing distributed inference over
heterogeneous peer networks with GGUF models. It supports BitNet-style 1-bit
models and is expanding toward all GGUF model architectures (Llama, Mistral,
Phi, Qwen, Gemma, and more).

This project is **in development**. It is useful for testing, research, and
iterating on distributed inference runtime ideas. It is not yet a polished
production inference server, and APIs or commands may still change.

Install from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/askscience/bitty/main/scripts/install_bitty.sh | bash
```

The normal user experience is intentionally similar to Ollama:

```bash
bitty setup
bitty share home
bitty run bitnet-b1.58 "Hello"
bitty serve
```

Simple commands start and reuse the local peer runtime in the background.
`bitty share`, `bitty connect`, and `bitty use` add short local names for
clusters, so everyday commands do not need repeated `iroh://...` invites.

## What Bitty Does

Bitty provides:

- An Ollama-like CLI for pulling, running, listing, showing, and managing models.
- A local model cache under `~/.bitty/models`.
- A settings file under `~/.bitty/config.toml`.
- Persistent logs under `~/.bitty/logs/bitty.log` with simple rotation.
- A Rust BitNet runtime for Microsoft's `BitNet-b1.58-2B-4T` GGUF model.
- Architecture detection for all GGUF models (Llama, Mistral, Phi, Qwen2, Gemma, Falcon, StableLM, DeepSeek, and more).
- Automatic quantization detection and classification from GGUF tensor types.
- An Iroh-based peer transport for encrypted peer-to-peer node communication.
- Experimental scheduler/worker internals for distributed layer execution.
- A simple HTTP API on `127.0.0.1:11435` for Ollama-style and OpenAI-compatible clients.
- Comprehensive Criterion benchmarks for GGUF parsing, metadata extraction, serialization, scheduling, and cluster simulation.

## Current Status

What works today:

- Install Bitty from source with one script.
- Pull the known BitNet GGUF model into the local Bitty model cache.
- Run any local GGUF file by path — Bitty auto-detects architecture, quantization, layer count, and metadata.
- Start and stop a background Bitty runtime with simple commands.
- Run prompts with `bitty run MODEL`; if a cluster is active, Bitty uses it by default.
- Start an API server with `bitty serve`.
- Manage local settings, model profiles, and aliases.
- Inspect local logs and cluster/node health from the `bitty` CLI.
- Start advanced Bitty nodes that communicate over Iroh.
- Run simulations, unit tests, and documented distributed control-plane smoke tests.

Important limitations:

- Bitty is still a development/test app.
- The main end-to-end inference runtime currently supports BitNet b1.58 GGUF.
- Metadata extraction and architecture detection work for all GGUF models; full inference for non-BitNet architectures is in progress.
- The distributed runtime is experimental and should be tested before relying on it.
- The HTTP API is a compatibility layer, not a full Ollama replacement yet.

## Install

Bitty has one installer:

```bash
scripts/install_bitty.sh
```

The installer can:

- Install Rust with `rustup` if needed.
- Install native build dependencies on common Linux distributions and macOS.
- Build the `bitty` binary.
- Link `bitty` into `~/.local/bin` by default.
- Optionally run tests after building.

If `~/.local/bin` is not in your shell path, add it:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

You can put that line in your shell profile, for example `~/.zshrc` or
`~/.bashrc`.

Install and run tests:

```bash
scripts/install_bitty.sh --run-tests
```

Build a debug binary instead of release:

```bash
scripts/install_bitty.sh --debug
```

Skip automatic system dependency installation:

```bash
scripts/install_bitty.sh --no-system-deps
```

Override install paths:

```bash
scripts/install_bitty.sh --install-dir "$HOME/src/bitty" --bin-dir "$HOME/.local/bin"
```

## Quick Start

Prepare Bitty and pull the default BitNet model if it is missing:

```bash
bitty setup
```

Create a cluster invite on the first computer:

```bash
bitty share home
```

On another computer, paste the invite printed by `bitty share`:

```bash
bitty connect 'PASTE_INVITE_HERE' --name home
```

Run a prompt. If a Bitty runtime is active or this machine has joined a
cluster, `bitty run` uses that cluster automatically:

```bash
bitty run bitnet-b1.58 "Explain 1-bit inference in simple words"
```

Bitty keeps the local scheduler/worker runtime in the background. Use `bitty ps`
to check it and `bitty stop` to stop it.

Run any local GGUF file directly — Bitty auto-detects the architecture:

```bash
bitty run /path/to/any-model.gguf "Hello"
```

Start interactive chat:

```bash
bitty run bitnet-b1.58
```

or:

```bash
bitty chat bitnet-b1.58
```

List installed models:

```bash
bitty ls
```

Show model details:

```bash
bitty show bitnet-b1.58
```

Start the local API server:

```bash
bitty serve
```

The server listens on `127.0.0.1:11435` by default, so it does not conflict
with Ollama's default `127.0.0.1:11434` port.

## Common Commands

`bitty pull MODEL`

Downloads a known model into the local Bitty model cache.

```bash
bitty pull bitnet-b1.58
```

`bitty run MODEL [PROMPT]`

Runs a model. If an active cluster is saved, Bitty sends the request to that
cluster. Use `--local` to force local-only generation, or `--node TARGET` to use
a specific cluster for one command. If no prompt is provided, Bitty opens an
interactive chat loop.

```bash
bitty run bitnet-b1.58 "Hello"
bitty run bitnet-b1.58
bitty run bitnet-b1.58 "Hello" --local
```

`bitty chat MODEL`

Friendly alias for interactive model use.

```bash
bitty chat bitnet-b1.58
```

`bitty ls` or `bitty list`

Lists models installed under `~/.bitty/models`.

```bash
bitty ls
```

`bitty show MODEL`

Shows model path, backend, quantization, layer count, source, and default
generation parameters.

```bash
bitty show bitnet-b1.58
```

`bitty ps`

Shows models Bitty has marked as loaded/running locally.

```bash
bitty ps
```

`bitty stop MODEL`

Stops tracking a loaded/running local model session.

```bash
bitty stop bitnet-b1.58
```

`bitty serve`

Starts the local HTTP API server.

```bash
bitty serve
bitty serve --host 127.0.0.1:11435
```

`bitty rm MODEL`

Removes a local model cache/profile.

```bash
bitty rm bitnet-b1.58
```

`bitty cp SOURCE DEST`

Creates a local alias/profile copy.

```bash
bitty cp bitnet-b1.58 my-bitnet
bitty run my-bitnet "Hello"
```

`bitty create NAME -f Modelfile`

Creates a local model profile from a Bitty/Ollama-style `Modelfile`.

```bash
bitty create my-bitnet -f Modelfile
```

`bitty logs`

Shows recent Bitty log lines from `~/.bitty/logs/bitty.log`.

```bash
bitty logs
bitty logs --lines 200
bitty logs --path
bitty logs --clear
```

Logs rotate automatically when `bitty.log` grows past roughly 1 MB. Bitty keeps
up to three rotated files: `bitty.log.1`, `bitty.log.2`, and `bitty.log.3`.

`bitty clean` and `bitty reset`

Use these when you want a clean slate on this machine. Both stop the background
Bitty runtime if it is running, print what they will remove, then ask you to
type **`yes`** exactly (anything else aborts with no changes).

`bitty clean [--data-dir DIR]`

Removes downloaded models, `clusters.toml`, `cluster-token`, `iroh-secret.key`,
`logs/`, `state/`, and `runtime/` under the data directory. Your **`config.toml`**
is kept (API host, defaults, and similar), but **`active_cluster`** is cleared so
Bitty no longer points at a saved cluster.

`bitty reset [--data-dir DIR]`

Deletes the **entire** Bitty data directory (including `config.toml`) and
recreates a fresh default `config.toml`, similar to a first install.

By default the data directory is `~/.bitty` (or `BITTY_DATA_DIR`). For safety,
these commands only run when that directory is named **`.bitty`**. To reset a
custom-named data directory, set **`BITTY_ALLOW_ANY_DATA_DIR_RESET=1`**.

```bash
bitty clean
bitty clean --data-dir /path/to/.bitty
bitty reset
```

`bitty cluster`

Checks and inspects the distributed Bitty cluster.

```bash
bitty share home
bitty connect 'iroh://INVITE_FROM_ANOTHER_NODE' --name home
bitty use home
bitty use 'iroh://INVITE_FROM_ANOTHER_NODE' --name home
bitty clusters
bitty start
bitty stop
bitty cluster status
bitty cluster nodes
bitty cluster check
```

`bitty share` starts the background runtime if needed, prints the local Iroh
invite string, and saves it under a local name. `bitty connect` accepts either a
full invite or a saved name, saves it, and starts a background worker node.
`bitty use` switches the active cluster without starting a node, and `bitty
clusters` lists saved local names.

`cluster status` prints leader, topology, worker count, model readiness, and
layer assignments. `cluster nodes` focuses on node/layer placement. `cluster
check` exits with an error when the cluster is not ready. Pass `--node TARGET`
only when you want to inspect a cluster that is not saved as the active cluster.

Cluster names are local aliases stored under `~/.bitty/clusters.toml`. If you
try to save a different invite under an existing name, Bitty reports a conflict
instead of silently replacing it. Use `--replace` only when you intentionally
want that name to point to a different cluster.

## Generation Options

Common generation and runtime flags:

```bash
bitty run bitnet-b1.58 "Hello" --temperature 0.2 --num-predict 64
```

Supported user-facing flags include:

- `--temperature`
- `--num-predict`
- `--num-ctx`
- `--seed`
- `--top-k`
- `--top-p`
- `--system`
- `--template`
- `--data-dir`
- `--no-auto-pull`
- `--no-daemon`
- `--local`
- `--node`
- `--join`

Some advanced flags are accepted for compatibility and future use even when the
current local runtime does not use all of them yet.

## Models

Bitty ships with a small built-in registry at `models/registry.toml`.

The first built-in model is:

```text
bitnet-b1.58
```

It points to Microsoft's `BitNet-b1.58-2B-4T` GGUF file:

```text
ggml-model-i2_s.gguf
```

Downloaded models are stored under:

```text
~/.bitty/models/<name>/<tag>/
```

For the default model, the local file is normally:

```text
~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf
```

### Running Any GGUF Model

Bitty auto-detects model architecture, quantization, layer count, and metadata
from any GGUF file. Supported architectures include:

- **Llama** (Llama 2, Llama 3, Llama 3.1)
- **Mistral** (Mistral 7B, Mixtral)
- **Phi** (Phi-3, Phi-3.5)
- **Qwen2** (Qwen2, Qwen2.5)
- **Gemma** (Gemma 1, Gemma 2)
- **Falcon**, **StableLM**, **DeepSeek**, **Mamba**
- **BitNet** (b1.58, OneBit)

Run with an explicit local GGUF path:

```bash
bitty run /path/to/model.gguf "Hello"
bitty run /path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf "Write a haiku"
```

Bitty reads `general.architecture` from the GGUF header, extracts all metadata
(hidden size, attention heads, vocab size, context length, ROPE dimensions), and
classifies the quantization level from tensor types (F32 through IQ2_XXS).

### Built-in Registry

The registry supports standard fields:

```toml
[[model]]
name = "my-model"
backend = "bitnet-i2s"
quantization = "q4_k"
filename = "model-q4_k.gguf"
layers = 32
url = "https://huggingface.co/org/model-gguf/resolve/main/model-q4_k.gguf"
source = "org/model-gguf"
```

Add entries to `models/registry.toml` to make models pullable by name.

## Settings

Bitty stores settings in:

```text
~/.bitty/config.toml
```

Print the settings file path:

```bash
bitty settings path
```

Show all settings:

```bash
bitty settings get
```

Show one setting:

```bash
bitty settings get default_model
```

Set a value:

```bash
bitty settings set default_model bitnet-b1.58
bitty settings set default_temperature 0.2
bitty settings set api_host 127.0.0.1:11435
bitty settings get active_cluster
```

`active_cluster` is usually managed automatically by `bitty share`, `bitty
connect`, `bitty node`, `bitty invite`, `bitty join`, and `bitty use`. Set it
manually only when you want this machine to use a specific cluster target by
default.

Current settings include:

- `data_dir`
- `models_dir`
- `default_model`
- `api_host`
- `auto_pull`
- `auto_start_node`
- `default_temperature`
- `default_num_predict`
- `default_num_ctx`
- `iroh_relays`
- `cluster_mode`

## Modelfile

Bitty supports a practical subset of Ollama-style `Modelfile` instructions:

- `FROM`
- `PARAMETER`
- `SYSTEM`
- `TEMPLATE`
- `MESSAGE`
- `LICENSE`

Example:

```text
FROM ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf
PARAMETER temperature 0.2
PARAMETER num_predict 128
SYSTEM """You are a concise assistant."""
```

Create a local profile:

```bash
bitty create concise-bitnet -f Modelfile
bitty run concise-bitnet "What is BitNet?"
```

Adapters such as LoRA are not applied yet. Unsupported instructions are kept as
metadata or ignored with development-friendly behavior.

## HTTP API

Start the API server:

```bash
bitty serve
```

Default address:

```text
http://127.0.0.1:11435
```

Ollama-style endpoints:

- `POST /api/generate`
- `POST /api/chat`
- `GET /api/tags`
- `POST /api/show`
- `POST /api/pull`

OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

List models:

```bash
curl http://127.0.0.1:11435/v1/models
```

Generate text:

```bash
curl http://127.0.0.1:11435/api/generate \
  -H 'content-type: application/json' \
  -d '{"model":"bitnet-b1.58","prompt":"Hello"}'
```

OpenAI-compatible chat:

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"bitnet-b1.58","messages":[{"role":"user","content":"Hello"}]}'
```

## Distributed Mode

Bitty is designed around shared resources. The normal distributed flow uses a
background runtime, so people do not need to keep a terminal open.

First computer:

```bash
bitty setup
bitty share home
```

Second computer:

```bash
bitty setup
bitty connect 'PASTE_INVITE_FROM_BITTY_SHARE' --name home
```

Then use Bitty normally:

```bash
bitty run bitnet-b1.58 "Hello"
bitty status
```

`bitty share` and `bitty connect` save the cluster name locally. Later, use
`bitty use home` to switch back to it, `bitty clusters` to list saved names,
`bitty ps` to check the background runtime, and `bitty stop` to stop it.

### Advanced Manual Nodes

The advanced manual flow is still available when you want direct control over
the node process, ports, or flags.

The first node becomes the scheduler leader and also starts a local worker
runtime. Iroh is enabled by default and stores a stable peer key under
`~/.bitty`.

Start the first node:

```bash
bitty node --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf --layers 30
```

The node prints an `iroh://...` join value containing the peer identity, cluster
token, and Iroh addressing information. Bitty also saves this as the active
cluster in `~/.bitty/config.toml`.

You can also print the current local Iroh invite and save a local cluster name:

```bash
bitty invite --name home
```

Keep the first manual `bitty node` process running. `bitty invite` is only for
printing the invite when you are using manual nodes.

Join another machine:

```bash
bitty join 'iroh://INVITE_FROM_BITTY_INVITE' \
  --name home \
  --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf
```

After joining, the invite is saved on that machine too. Later, you can switch
back to it with `bitty use home` or inspect saved names with `bitty clusters`.
Use the cluster normally:

```bash
bitty run bitnet-b1.58 "Hello"
bitty chat bitnet-b1.58
```

Check cluster state:

```bash
bitty status
bitty cluster status
bitty cluster nodes
bitty cluster check
```

You can still override the saved target for one command:

```bash
bitty run bitnet-b1.58 "Hello" --node 'iroh://INVITE_FROM_BITTY_CLUSTER_INVITE'
bitty status --node 'iroh://INVITE_FROM_BITTY_CLUSTER_INVITE'
```

Iroh first attempts direct peer-to-peer QUIC and can use relays for NAT
traversal or encrypted fallback. This lets machines communicate across networks
without manually entering worker IP addresses. Use `--no-iroh` only for
fully-local TCP-only testing.

### Security Notes

Network-facing coordinator and worker RPCs should use a shared cluster token.
Iroh invites include the token in the invite URL, so treat invite strings like
secrets and avoid pasting them into public logs or issue trackers. The CLI
redacts token arguments from Bitty logs, but shell history and terminal scrollback
can still contain pasted invites.

The standalone TCP binaries are local-first by default: unauthenticated requests
are accepted only from loopback/internal calls. When binding to `0.0.0.0` or
joining over TCP from another machine, pass the same token to both sides:

```bash
cargo run -p bitty-coordinator -- \
  --listen 0.0.0.0:50051 \
  --model /models/ggml-model-i2_s.gguf \
  --token "$BITTY_CLUSTER_TOKEN"

cargo run -p bitty-worker -- \
  --node-id worker-a \
  --listen 0.0.0.0:50061 \
  --public-endpoint worker-a.example:50061 \
  --coordinator coordinator.example:50051 \
  --model /models/ggml-model-i2_s.gguf \
  --token "$BITTY_CLUSTER_TOKEN"
```

## Benchmarks

Bitty has comprehensive Criterion micro-benchmarks covering every hot path in the
codebase. Run all benchmarks:

```bash
cargo bench --workspace
```

Run a specific benchmark group:

```bash
# GGUF parsing — byte-level parsing at 2/30/80 layers, 100–1000 tensors, variable metadata
cargo bench -p bitty-model --bench gguf_parsing

# Metadata extraction — architecture classification (14 archs), ModelMetadata::from_gguf at scale
cargo bench -p bitty-model --bench metadata_extraction

# Low-level helpers — layer_id_from_tensor_name, ggml_type_name, bytes_per_element, i2_s decode
cargo bench -p bitty-model --bench gguf_helpers

# Tensor operations — packed_len_for all 9 quantizations, validate_len
cargo bench -p bitty-model --bench tensor_ops

# Activation codecs — FP8, sparse topk, raw encode/decode at 4KB–1MB, roundtrip
cargo bench -p bitty-model --bench activation_codec

# Shard planning — shard_plan for 1/4/8 nodes × 4/30/80 layers, layer_metadata scaling
cargo bench -p bitty-model --bench shard_planning

# Protocol wire format — logits encoding, CRC32 checksum (4KB–1MB), activation proto roundtrip
cargo bench -p bitty-protocol --bench logits_wire
cargo bench -p bitty-protocol --bench activation_wire

# Inference executor — FakeLayerExecutor forward throughput, logits, dispatch overhead
cargo bench -p bitty-inference --bench ring_execution

# Scheduler — Halda::assign scaling with 4–256 nodes × 30–120 layers, compute score ranking
cargo bench -p bitty-coordinator --bench scheduling

# Cluster simulation — SimulatedCluster::build (4/8/16 nodes), token streaming throughput
cargo bench -p bitty-sim --bench cluster_simulation

# Worker profiling — HardwareProfiler wall-clock time, compute score, memory estimation
cargo bench -p bitty-worker --bench profiling
```

All benchmarks use `criterion = "0.5"` with `harness = false`. Results are
written to `target/criterion/` with HTML reports. CI runs all 12 benchmarks via
`.github/workflows/benches.yml` (manual trigger).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo audit
```

Run a local Halda scheduling pass:

```bash
cargo run -p bitty-coordinator -- --nodes 8 --layers 16
```

Run a local worker profile and dummy keepalive pass:

```bash
cargo run -p bitty-worker -- --node-id local-worker-0 --keepalive
```

Run the deterministic in-process ring simulator:

```bash
cargo run -p bitty-sim -- --nodes 8 --layers 16 --tokens 4
```

Inject a simulated node failure:

```bash
cargo run -p bitty-sim -- --nodes 8 --layers 16 --tokens 4 --drop-node sim-0
```

Run the tiny local language model:

```bash
cargo run -p bitty-inference --bin bitty-tiny-lm -- --prompt "The coordinator" --chars 240 --seed 7
```

The tiny model is an in-repo byte-level probabilistic model. It is useful for
testing prompt-conditioned text generation without downloading external weights.
It is not a transformer, BitNet model, or distributed shard executor.

## Manual BitNet Tests

Run the Rust BitNet runtime directly:

```bash
cargo run -p bitty-inference --bin bitty-rust-bitnet -- \
  --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf \
  --prompt "Explain BitNet in one paragraph" \
  --max-tokens 80
```

Run the ignored split-vs-full correctness gate with a downloaded GGUF:

```bash
BITTY_GGUF_MODEL=~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf \
  cargo test -p bitty-bitnet-runtime \
  split_local_logits_match_full_local_logits_for_temperature_zero -- --ignored
```

Same-machine Iroh smoke test:

```bash
bitty node --data-dir /tmp/bitty-a \
  --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf \
  --layers 30

bitty node --data-dir /tmp/bitty-b \
  --join 'iroh://INVITE_PRINTED_BY_FIRST_NODE' \
  --node-id worker-b \
  --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf

bitty run bitnet-b1.58 "Hello"
bitty cluster check
```

## Workspace Layout

- `bitty-cli`: user-facing `bitty` binary.
- `bitty-protocol`: shared protobuf messages, domain types, CRC32 checksums, and Iroh framing.
- `bitty-coordinator`: worker registry, Halda scheduling, topology, routing, and snapshots.
- `bitty-worker`: profiling, shard lifecycle, ring execution, keepalive, and worker metrics.
- `bitty-bitnet-runtime`: Rust BitNet split runtime for loading, layer ranges, KV cache, logits, and sampling.
- `bitty-model`: GGUF parsing, architecture detection, model metadata extraction, low-bit shard primitives, and activation codecs (FP8, sparse topk, delta).
- `bitty-inference`: executor traits and request lifecycle orchestration.
- `bitty-sim`: deterministic in-process cluster simulation.
- `bitty-observability`: metrics and tracing helpers.

## Advanced Debug Binaries

The role-specific binaries still exist for debugging lower-level pieces:

```bash
cargo run -p bitty-coordinator -- --listen 0.0.0.0:50051 --model /path/to/model.gguf --layers 30
cargo run -p bitty-worker -- --node-id worker-a --listen 0.0.0.0:50061 --coordinator 127.0.0.1:50051 --model /path/to/model.gguf
cargo run -p bitty-inference --bin bitty-client -- --coordinator 127.0.0.1:50051 --prompt "Hello"
```

These commands are for development and debugging. Most users should use
`bitty pull`, `bitty run`, and `bitty serve`.

## License

Bitty is licensed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE) for details.
