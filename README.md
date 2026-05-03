# Bitty

Bitty is a Rust workspace for experimenting with distributed BitNet inference
over heterogeneous peer networks. The user-facing command is `bitty`: every
machine runs a node, Bitty creates an Iroh peer identity by default, and the
internal scheduler/worker split is hidden behind leader and join behavior.

## Crates

- `bitty-protocol`: shared cluster messages and domain types.
- `bitty-coordinator`: worker registry, Halda scheduling, topology, routing, and snapshots.
- `bitty-worker`: profiling, shard lifecycle, ring execution, keepalive, and worker metrics.
- `bitty-bitnet-runtime`: Rust BitNet split runtime for model loading, embedding, layer ranges, KV cache, logits, and sampling.
- `bitty-model`: low-bit shard and activation codec primitives.
- `bitty-inference`: executor traits and request lifecycle orchestration.
- `bitty-sim`: deterministic in-process cluster simulation.
- `bitty-observability`: metrics and tracing helpers.

## Install Bitty

Bitty has one installer: [`scripts/install_bitty.sh`](scripts/install_bitty.sh).
Run it on every PC/server. It can install Rust with `rustup`, build release
binaries, link the `bitty` command into `~/.local/bin`, and print the exact node
command for the role you choose.

The installer also checks native build dependencies. On common Linux
distributions it can install the required compiler tools, `pkg-config`, OpenSSL
development headers, and `protoc` automatically with the system package manager.
Use `--no-system-deps` if you prefer to install OS packages yourself.

Quick start:

```bash
git clone git@github.com:askscience/bitty.git
cd bitty
scripts/install_bitty.sh
bitty pull bitnet-b1.58
bitty run bitnet-b1.58
```

Common commands:

```bash
bitty pull bitnet-b1.58
bitty run bitnet-b1.58 "Explain 1-bit inference"
bitty chat bitnet-b1.58
bitty ls
bitty show bitnet-b1.58
bitty serve
```

Additional machines can still join a distributed Bitty cluster with the Iroh
invite printed by the first `bitty node`, but normal single-machine usage starts
with model names instead of raw GGUF paths.

Advanced distributed node:

```bash
bitty node --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf
```

The installer links binaries into `~/.local/bin` by default. You can override
paths with `INSTALL_DIR`, `BIN_DIR`, `REPO_URL`, or command-line flags. The
model file must exist locally on every node that will execute BitNet layers.
`bitty pull` stores models under `~/.bitty/models` by default.

## Local Testing

Install Bitty from GitHub on another machine for development:

```bash
git clone git@github.com:askscience/bitty.git
cd bitty
scripts/install_bitty.sh --install-dir "$PWD" --debug --run-tests
```

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

The tiny model is an in-repo byte-level probabilistic model trained from a small
distributed-inference corpus. It is useful for testing prompt-conditioned text
generation without downloading external weights. It is not yet a transformer,
BitNet, or distributed shard executor.

## BitNet b1.58 Testing

The smallest official practical BitNet target is Microsoft's
`BitNet-b1.58-2B-4T` GGUF model. The distributed path uses the Rust
`bitty-bitnet-runtime` split runtime and does not call `bitnet.cpp`.

Run the legacy local `bitnet.cpp` wrapper only if you have installed
`bitnet.cpp` separately:

```bash
cargo run -p bitty-inference --bin bitty-bitnet -- \
  --prompt "Explain BitNet in one paragraph" \
  --n-predict 80
```

Run the Rust split runtime locally:

```bash
cargo run -p bitty-inference --bin bitty-rust-bitnet -- \
  --model /path/to/ggml-model-i2_s.gguf \
  --prompt "Explain BitNet in one paragraph" \
  --max-tokens 80
```

The same runtime is used by `BitNetLayerExecutor` inside workers for distributed
layer-range execution.

The default model path is:

```text
external/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

You can override paths if you already have the runtime or model:

```bash
cargo run -p bitty-inference --bin bitty-bitnet -- \
  --runtime-dir /path/to/BitNet \
  --model /path/to/ggml-model-i2_s.gguf \
  --prompt "What is 1-bit inference?"
```

## Distributed BitNet Smoke

Start the first node. It becomes the scheduler leader and also starts a local
worker runtime. Iroh is enabled by default, stores a stable node key under
`~/.bitty`, and prints a secure join value containing the cluster token:

```bash
cargo run -p bitty-cli --bin bitty -- node \
  --model /path/to/ggml-model-i2_s.gguf \
  --layers 30
```

Run more nodes with the printed `iroh://...` join value. Scheduler RPCs,
heartbeats, worker activation forwarding, logits, shard load, cleanup, status,
and generation all run over Iroh's encrypted QUIC streams:

```bash
cargo run -p bitty-cli --bin bitty -- node \
  --join 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' \
  --node-id worker-a \
  --model /path/to/ggml-model-i2_s.gguf
```

Send a generation request to the leader:

```bash
cargo run -p bitty-cli --bin bitty -- generate \
  --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' \
  --prompt "Hello" \
  --max-tokens 32 \
  --temperature 0
```

Check cluster state:

```bash
cargo run -p bitty-cli --bin bitty -- status --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
```

Interactive chat:

```bash
cargo run -p bitty-cli --bin bitty -- chat --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'
```

Correctness gates to run before changing performance code:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Manual gates for a downloaded Microsoft GGUF:

```bash
# full local Rust gate
cargo run -p bitty-inference --bin bitty-rust-bitnet -- \
  --model /path/to/ggml-model-i2_s.gguf \
  --prompt "Hello" \
  --max-tokens 8 \
  --temperature 0

# split local Rust gate
BITTY_GGUF_MODEL=/path/to/ggml-model-i2_s.gguf \
  cargo test -p bitty-bitnet-runtime \
  split_local_logits_match_full_local_logits_for_temperature_zero -- --ignored

# same-machine Iroh gate
cargo run -p bitty-cli --bin bitty -- node --data-dir /tmp/bitty-a --model /path/to/ggml-model-i2_s.gguf --layers 30
cargo run -p bitty-cli --bin bitty -- node --data-dir /tmp/bitty-b --join 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' --node-id worker-b --model /path/to/ggml-model-i2_s.gguf
cargo run -p bitty-cli --bin bitty -- generate --node 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' --prompt "Hello" --max-tokens 8 --temperature 0
```

## Multi-PC Usage

The normal runtime is `bitty node`. The first node is the leader; joined nodes
contribute worker runtime capacity. Users do not need to start separate
coordinator and worker binaries.

Iroh is enabled by default for node identity, leader lookup, scheduler RPCs, and
worker activation traffic. Iroh first attempts direct peer-to-peer QUIC and uses
relays for NAT traversal or encrypted fallback, so nodes can communicate across
the internet without manual worker IPs. Public Iroh relays are useful for
development and testing; production deployments should configure dedicated relays.
Use `--no-iroh` only when you want a fully local/offline TCP-only run.

What works today:

- Install and run the project on two PCs independently.
- Run PC 1 as a Bitty leader node and PC 2 as a joined Bitty node.
- Run the in-process simulator with many virtual nodes.
- Run the real `BitNet-b1.58-2B-4T` model locally through the Rust split runtime.
- Exercise the scheduler, topology, worker profile, activation codec, and
  simulated ring tests.

Two-PC distributed smoke test:

On PC 1:

```bash
cargo run -p bitty-cli --bin bitty -- node \
  --model /path/to/ggml-model-i2_s.gguf \
  --layers 30
```

On PC 2, use the `iroh://...` join value printed by PC 1:

```bash
cargo run -p bitty-cli --bin bitty -- node \
  --join 'iroh://PC1_IROH_NODE_ID?token=CLUSTER_TOKEN' \
  --node-id pc2 \
  --model /path/to/ggml-model-i2_s.gguf
```

The two-machine gate is deterministic when `--temperature 0` is used.

## Advanced Debug Binaries

The role-specific binaries still exist for debugging lower-level pieces:

```bash
cargo run -p bitty-coordinator -- --listen 0.0.0.0:50051 --model /path/to/model.gguf --layers 30
cargo run -p bitty-worker -- --node-id worker-a --listen 0.0.0.0:50061 --coordinator 127.0.0.1:50051 --model /path/to/model.gguf
cargo run -p bitty-inference --bin bitty-client -- --coordinator 127.0.0.1:50051 --prompt "Hello"
```

## License

Bitty is licensed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE) for details.
