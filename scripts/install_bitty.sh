#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:askscience/bitty.git}"
INSTALL_DIR="${INSTALL_DIR:-"$HOME/bitty"}"
BIN_DIR="${BIN_DIR:-"$HOME/.local/bin"}"
ROLE="${ROLE:-node}"
MODEL_PATH="${MODEL_PATH:-}"
JOIN="${JOIN:-${COORDINATOR:-}}"
NODE_ID="${NODE_ID:-"$(hostname)-node"}"
LISTEN="${LISTEN:-0.0.0.0:50051}"
WORKER_LISTEN="${WORKER_LISTEN:-}"
PUBLIC_ENDPOINT="${PUBLIC_ENDPOINT:-}"
LAYERS="${LAYERS:-30}"
RUN_TESTS="${RUN_TESTS:-0}"
BUILD_PROFILE="${BUILD_PROFILE:-release}"
INSTALL_RUST="${INSTALL_RUST:-1}"

usage() {
  cat <<'EOF'
Install Bitty and print the command to run it.

Usage:
  scripts/install_bitty.sh [options]

Common:
  scripts/install_bitty.sh --role node --model /models/ggml-model-i2_s.gguf
  scripts/install_bitty.sh --role join --join 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN' --model /models/ggml-model-i2_s.gguf
  scripts/install_bitty.sh --role client --join 'iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN'

Options:
  --role node|join|client   Which command to print after install. Default: node
  --install-dir PATH        Checkout/build directory. Default: $HOME/bitty
  --bin-dir PATH            Symlink destination. Default: $HOME/.local/bin
  --repo URL                Git repository. Default: git@github.com:askscience/bitty.git
  --model PATH              Local GGUF model path
  --join HOST:PORT          Existing Bitty node to join or call; accepts iroh:// invites
  --node-id ID              Node id. Default: $(hostname)-node
  --listen HOST:PORT        Main listen address. Default: 0.0.0.0:50051
  --worker-listen HOST:PORT TCP fallback worker listen address
  --public-endpoint ADDR    TCP fallback worker address reachable by other nodes
  --layers N                Model layer count. Default: 30
  --run-tests               Run cargo test --workspace after build
  --debug                   Build debug binaries instead of release
  --no-rustup               Do not auto-install Rust if cargo is missing
  -h, --help                Show this help

Environment variables with the same uppercase names are also supported:
  REPO_URL, INSTALL_DIR, BIN_DIR, ROLE, MODEL_PATH, JOIN, NODE_ID, LISTEN,
  WORKER_LISTEN, PUBLIC_ENDPOINT, LAYERS, RUN_TESTS, BUILD_PROFILE, INSTALL_RUST.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --join|--coordinator) JOIN="$2"; shift 2 ;;
    --node-id) NODE_ID="$2"; shift 2 ;;
    --listen) LISTEN="$2"; shift 2 ;;
    --worker-listen) WORKER_LISTEN="$2"; shift 2 ;;
    --public-endpoint) PUBLIC_ENDPOINT="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --run-tests) RUN_TESTS=1; shift ;;
    --debug) BUILD_PROFILE=debug; shift ;;
    --no-rustup) INSTALL_RUST=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ROLE" in
  node|leader) ROLE="node" ;;
  join|worker) ROLE="join" ;;
  client) ;;
  *) echo "invalid --role: $ROLE" >&2; usage >&2; exit 2 ;;
esac

if ! command -v git >/dev/null 2>&1; then
  echo "git is required. Install git, then rerun this script." >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  if [ "$INSTALL_RUST" != "1" ]; then
    echo "cargo is missing and --no-rustup was used." >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install Rust automatically." >&2
    exit 1
  fi
  echo "Installing Rust with rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

if [ -d "$INSTALL_DIR/.git" ]; then
  echo "Updating Bitty checkout at $INSTALL_DIR"
  git -C "$INSTALL_DIR" pull --ff-only
elif [ -e "$INSTALL_DIR" ]; then
  echo "install dir exists but is not a git checkout: $INSTALL_DIR" >&2
  exit 1
else
  echo "Cloning $REPO_URL into $INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

if [ "$BUILD_PROFILE" = "release" ]; then
  cargo build --release --workspace --bins
  TARGET_DIR="$INSTALL_DIR/target/release"
else
  cargo build --workspace --bins
  TARGET_DIR="$INSTALL_DIR/target/debug"
fi

if [ "$RUN_TESTS" = "1" ]; then
  cargo test --workspace
fi

mkdir -p "$BIN_DIR"
for bin in bitty bitty-coordinator bitty-worker bitty-client bitty-rust-bitnet bitty-sim; do
  if [ -x "$TARGET_DIR/$bin" ]; then
    ln -sf "$TARGET_DIR/$bin" "$BIN_DIR/$bin"
  fi
done

cat <<EOF

Bitty installed.

Install dir:
  $INSTALL_DIR

Binaries linked in:
  $BIN_DIR

Add this to your shell profile if needed:
  export PATH="$BIN_DIR:\$PATH"

EOF

case "$ROLE" in
  node)
    if [ -z "$MODEL_PATH" ]; then
      MODEL_PATH="/path/to/ggml-model-i2_s.gguf"
    fi
    if [ -z "$WORKER_LISTEN" ]; then
      WORKER_LISTEN="0.0.0.0:50061"
    fi
    cat <<EOF
Start the first Bitty node:
  $BIN_DIR/bitty node --model "$MODEL_PATH" --layers "$LAYERS"

The node stores a stable Iroh identity in ~/.bitty and prints an iroh:// join invite.
EOF
    ;;
  join)
    if [ -z "$MODEL_PATH" ]; then
      MODEL_PATH="/path/to/ggml-model-i2_s.gguf"
    fi
    if [ -z "$JOIN" ]; then
      JOIN="iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN"
    fi
    cat <<EOF
Join an existing Bitty node:
  $BIN_DIR/bitty node --join "$JOIN" --node-id "$NODE_ID" --model "$MODEL_PATH"
EOF
    ;;
  client)
    if [ -z "$JOIN" ]; then
      JOIN="iroh://LEADER_IROH_NODE_ID?token=CLUSTER_TOKEN"
    fi
    cat <<EOF
Send a request:
  $BIN_DIR/bitty generate --node "$JOIN" --prompt "Hello" --max-tokens 32 --temperature 0
EOF
    ;;
esac
