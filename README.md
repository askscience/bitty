# Bitty

Bitty is an experimental Rust project for testing distributed inference with
BitNet-style 1-bit models over heterogeneous peer networks.

This project is **in development**. It is useful for testing, research, and
iterating on distributed BitNet runtime ideas. It is not yet a polished
production inference server, and APIs or commands may still change.

The normal user experience is intentionally similar to Ollama:

```bash
bitty pull bitnet-b1.58
bitty run bitnet-b1.58
bitty serve
```

Advanced users can still run low-level distributed nodes, inspect scheduling,
and test Iroh peer-to-peer transport directly.

## What Bitty Does

Bitty provides:

- An Ollama-like CLI for pulling, running, listing, showing, and managing models.
- A local model cache under `~/.bitty/models`.
- A settings file under `~/.bitty/config.toml`.
- Persistent logs under `~/.bitty/logs/bitty.log` with simple rotation.
- A Rust BitNet runtime path for Microsoft's `BitNet-b1.58-2B-4T` GGUF model.
- An Iroh-based peer transport for encrypted peer-to-peer node communication.
- Experimental scheduler/worker internals for distributed layer execution.
- A simple HTTP API on `127.0.0.1:11435` for Ollama-style and OpenAI-compatible clients.

## Current Status

What works today:

- Install Bitty from source with one script.
- Pull the known BitNet GGUF model into the local Bitty model cache.
- Run local prompts with `bitty run MODEL`.
- Start an API server with `bitty serve`.
- Manage local settings, model profiles, and aliases.
- Inspect local logs and cluster/node health from the `bitty` CLI.
- Start advanced Bitty nodes that communicate over Iroh.
- Run simulations, unit tests, and distributed control-plane smoke tests.

Important limitations:

- Bitty is still a development/test app.
- The main supported real model path is currently BitNet b1.58 GGUF.
- Normal GGUF models are not broadly supported yet.
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

Install from GitHub:

```bash
git clone git@github.com:askscience/bitty.git
cd bitty
scripts/install_bitty.sh
```

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

Pull the default BitNet model:

```bash
bitty pull bitnet-b1.58
```

Run a prompt:

```bash
bitty run bitnet-b1.58 "Explain 1-bit inference in simple words"
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

Runs a model. If no prompt is provided, Bitty opens an interactive chat loop.

```bash
bitty run bitnet-b1.58 "Hello"
bitty run bitnet-b1.58
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

`bitty cluster`

Checks and inspects the distributed Bitty cluster.

```bash
bitty cluster status --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
bitty cluster nodes --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
bitty cluster check --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
bitty cluster invite
```

`cluster status` prints leader, topology, worker count, model readiness, and
layer assignments. `cluster nodes` focuses on node/layer placement. `cluster
check` exits with an error when the cluster is not ready. `cluster invite`
prints the local Iroh invite string for sharing with another Bitty node.

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

You can also run with an explicit local GGUF path:

```bash
bitty run /path/to/ggml-model-i2_s.gguf "Hello"
```

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
```

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

The easy commands hide distributed internals. For experiments, Bitty can also run
explicit nodes.

The first node becomes the scheduler leader and also starts a local worker
runtime. Iroh is enabled by default and stores a stable peer key under
`~/.bitty`.

Start the first node:

```bash
bitty node --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf --layers 30
```

The node prints an `iroh://...` join value containing the peer identity and
cluster token.

You can also print the current local Iroh invite:

```bash
bitty cluster invite
```

Join another machine:

```bash
bitty node \
  --join 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' \
  --node-id worker-a \
  --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf
```

Send a request to the leader:

```bash
bitty generate \
  --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' \
  --prompt "Hello" \
  --max-tokens 32 \
  --temperature 0
```

Check cluster state:

```bash
bitty status --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
bitty cluster status --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
bitty cluster nodes --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
bitty cluster check --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
```

Iroh first attempts direct peer-to-peer QUIC and can use relays for NAT
traversal or encrypted fallback. This lets machines communicate across networks
without manually entering worker IP addresses. Use `--no-iroh` only for
fully-local TCP-only testing.

## Development

Run the full verification suite:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
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
  --join 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' \
  --node-id worker-b \
  --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf

bitty generate \
  --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' \
  --prompt "Hello" \
  --max-tokens 8 \
  --temperature 0
```

## Workspace Layout

- `bitty-cli`: user-facing `bitty` binary.
- `bitty-protocol`: shared protobuf messages, domain types, and Iroh framing.
- `bitty-coordinator`: worker registry, Halda scheduling, topology, routing, and snapshots.
- `bitty-worker`: profiling, shard lifecycle, ring execution, keepalive, and worker metrics.
- `bitty-bitnet-runtime`: Rust BitNet split runtime for loading, layer ranges, KV cache, logits, and sampling.
- `bitty-model`: low-bit shard and activation codec primitives.
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
