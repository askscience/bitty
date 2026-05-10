#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/askscience/bitty.git}"
BRANCH="${BRANCH:-gpu-vulkan}"
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
CLUSTER_TOKEN="${CLUSTER_TOKEN:-}"
RUN_TESTS="${RUN_TESTS:-0}"
BUILD_PROFILE="${BUILD_PROFILE:-release}"
INSTALL_RUST="${INSTALL_RUST:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"

usage() {
  cat <<'EOF'
Install Bitty and print the command to run it.

Usage:
  scripts/install_bitty.sh [options]

Common:
  scripts/install_bitty.sh
  bitty pull bitnet-b1.58
  bitty node --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf
  bitty run bitnet-b1.58
  scripts/install_bitty.sh --role node --model bitnet-b1.58
  scripts/install_bitty.sh --role join --join 'iroh://INVITE_FROM_BITTY_CLUSTER_INVITE' --model /models/ggml-model-i2_s.gguf
  scripts/install_bitty.sh --role client --join 'iroh://INVITE_FROM_BITTY_CLUSTER_INVITE'

Options:
  --role node|join|client   Which command to print after install. Default: node
  --install-dir PATH        Checkout/build directory. Default: $HOME/bitty
  --bin-dir PATH            Symlink destination. Default: $HOME/.local/bin
  --branch BRANCH            Git branch. Default: testing
  --repo URL                Git repository. Default: git@github.com:askscience/bitty.git
  --model PATH              Local GGUF model path
  --join HOST:PORT          Existing Bitty node to join or call; accepts iroh:// invites
  --node-id ID              Node id. Default: $(hostname)-node
  --listen HOST:PORT        Main listen address. Default: 0.0.0.0:50051
  --worker-listen HOST:PORT TCP fallback worker listen address
  --public-endpoint ADDR    TCP fallback worker address reachable by other nodes
  --layers N                Model layer count. Default: 30
  --cluster-token TOKEN     Shared token for TCP coordinator/worker RPCs
  --run-tests               Run cargo test --workspace after build
  --debug                   Build debug binaries instead of release
  --no-system-deps          Do not auto-install native build dependencies
  --no-rustup               Do not auto-install Rust if cargo is missing
  -h, --help                Show this help

Environment variables with the same uppercase names are also supported:
  REPO_URL, BRANCH, INSTALL_DIR, BIN_DIR, ROLE, MODEL_PATH, JOIN, NODE_ID, LISTEN,
  WORKER_LISTEN, PUBLIC_ENDPOINT, LAYERS, CLUSTER_TOKEN, RUN_TESTS, BUILD_PROFILE,
  INSTALL_RUST, INSTALL_SYSTEM_DEPS.

Performance controls for heterogeneous clusters:
  BITTY_NODE_ROLE=coordinator|worker|client
  BITTY_DISABLE_MODEL_LAYERS=1
  BITTY_MAX_LAYERS=N
  BITTY_RAM_MB, BITTY_VRAM_MB, BITTY_GPU_NAME, BITTY_GPU_TFLOPS
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif has_cmd sudo; then
    sudo "$@"
  else
    return 1
  fi
}

missing_cmds() {
  missing=""
  for cmd in "$@"; do
    if ! has_cmd "$cmd"; then
      missing="$missing $cmd"
    fi
  done
  echo "$missing"
}

configure_macos_openssl() {
  if [ "$(uname -s)" = "Darwin" ] && has_cmd brew; then
    if [ -d "$(brew --prefix openssl@3 2>/dev/null)" ]; then
      export OPENSSL_DIR="$(brew --prefix openssl@3)"
      export PKG_CONFIG_PATH="$OPENSSL_DIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    fi
  fi
}

system_deps_ready() {
  configure_macos_openssl
  missing="$(missing_cmds git curl pkg-config protoc)"
  if [ -n "$missing" ]; then
    return 1
  fi
  pkg-config --exists openssl >/dev/null 2>&1
}

print_manual_deps() {
  cat >&2 <<'EOF'
Bitty needs native build dependencies before Cargo can compile everything:
  git, curl, C/C++ build tools, pkg-config/pkgconf, OpenSSL development headers,
  and protoc/protobuf compiler.

Manual install examples:
  Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y build-essential pkg-config libssl-dev protobuf-compiler git curl ca-certificates clang
  Fedora/RHEL:   sudo dnf install -y gcc gcc-c++ make pkgconf-pkg-config openssl-devel protobuf-compiler git curl ca-certificates clang
  Arch Linux:    sudo pacman -Sy --needed base-devel pkgconf openssl protobuf git curl ca-certificates clang
  Alpine:        sudo apk add build-base pkgconf openssl-dev protobuf-dev protobuf git curl ca-certificates clang
  macOS:         brew install pkgconf openssl protobuf git curl

Rerun scripts/install_bitty.sh after installing them.
EOF
}

install_system_deps() {
  configure_macos_openssl
  if system_deps_ready; then
    return 0
  fi

  if [ "$INSTALL_SYSTEM_DEPS" != "1" ]; then
    return 0
  fi

  case "$(uname -s)" in
    Linux)
      if has_cmd apt-get; then
        echo "Installing Bitty native build dependencies with apt..."
        run_privileged apt-get update
        run_privileged apt-get install -y build-essential pkg-config libssl-dev protobuf-compiler git curl ca-certificates clang
      elif has_cmd dnf; then
        echo "Installing Bitty native build dependencies with dnf..."
        run_privileged dnf install -y gcc gcc-c++ make pkgconf-pkg-config openssl-devel protobuf-compiler git curl ca-certificates clang
      elif has_cmd yum; then
        echo "Installing Bitty native build dependencies with yum..."
        run_privileged yum install -y gcc gcc-c++ make pkgconfig openssl-devel protobuf-compiler git curl ca-certificates clang
      elif has_cmd pacman; then
        echo "Installing Bitty native build dependencies with pacman..."
        run_privileged pacman -Sy --needed base-devel pkgconf openssl protobuf git curl ca-certificates clang
      elif has_cmd apk; then
        echo "Installing Bitty native build dependencies with apk..."
        run_privileged apk add build-base pkgconf openssl-dev protobuf-dev protobuf git curl ca-certificates clang
      elif has_cmd zypper; then
        echo "Installing Bitty native build dependencies with zypper..."
        run_privileged zypper install -y gcc gcc-c++ make pkg-config libopenssl-devel protobuf-devel protobuf git curl ca-certificates clang
      else
        print_manual_deps
        exit 1
      fi
      ;;
    Darwin)
      if ! has_cmd brew; then
        cat >&2 <<'EOF'
Homebrew is required for automatic macOS dependency installation.
Install it from https://brew.sh, or rerun with --no-system-deps after manually
installing pkgconf, openssl, protobuf, git, and curl.
EOF
        exit 1
      fi
      echo "Installing Bitty native build dependencies with Homebrew..."
      brew install pkgconf openssl protobuf git curl
      configure_macos_openssl
      ;;
    FreeBSD)
      if has_cmd pkg; then
        echo "Installing Bitty native build dependencies with pkg..."
        run_privileged pkg install -y pkgconf openssl protobuf git curl gmake llvm
      else
        print_manual_deps
        exit 1
      fi
      ;;
    *)
      print_manual_deps
      exit 1
      ;;
  esac
}

check_required_deps() {
  configure_macos_openssl
  missing="$(missing_cmds git curl pkg-config protoc)"
  if [ -n "$missing" ]; then
    echo "Missing required command(s):$missing" >&2
    print_manual_deps
    exit 1
  fi
  if ! pkg-config --exists openssl >/dev/null 2>&1; then
    echo "OpenSSL development files were not found by pkg-config." >&2
    print_manual_deps
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --join|--coordinator) JOIN="$2"; shift 2 ;;
    --node-id) NODE_ID="$2"; shift 2 ;;
    --listen) LISTEN="$2"; shift 2 ;;
    --worker-listen) WORKER_LISTEN="$2"; shift 2 ;;
    --public-endpoint) PUBLIC_ENDPOINT="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --cluster-token|--token) CLUSTER_TOKEN="$2"; shift 2 ;;
    --run-tests) RUN_TESTS=1; shift ;;
    --debug) BUILD_PROFILE=debug; shift ;;
    --no-system-deps) INSTALL_SYSTEM_DEPS=0; shift ;;
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

install_system_deps
check_required_deps

if ! has_cmd cargo; then
  if [ "$INSTALL_RUST" != "1" ]; then
    echo "cargo is missing and --no-rustup was used." >&2
    exit 1
  fi
  if ! has_cmd curl; then
    echo "curl is required to install Rust automatically." >&2
    exit 1
  fi
  echo "Installing Rust with rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

if ! has_cmd cargo; then
  cat >&2 <<'EOF'
Rust was installed, but cargo is still not available in this shell.
Try rerunning this installer after opening a new terminal, or run:
  source "$HOME/.cargo/env"
EOF
  exit 1
fi

if [ -d "$INSTALL_DIR/.git" ]; then
  echo "Updating Bitty checkout at $INSTALL_DIR (branch: $BRANCH)"
  git -C "$INSTALL_DIR" checkout "$BRANCH"
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
elif [ -e "$INSTALL_DIR" ]; then
  echo "install dir exists but is not a git checkout: $INSTALL_DIR" >&2
  exit 1
else
  echo "Cloning $REPO_URL (branch: $BRANCH) into $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# Stop any running bitty processes before building (prevents old daemons
# from staying alive with buggy code after reinstall)
echo "Stopping any running bitty processes..."
pkill -9 bitty 2>/dev/null || true
pkill -9 bitty-coordinator 2>/dev/null || true
pkill -9 bitty-worker 2>/dev/null || true
sleep 1

# Auto-detect GPU features for the host hardware
detect_gpu_features() {
  case "$(uname -s)" in
    Darwin) echo "--features gpu-metal" ;;
    Linux)
      if command -v nvidia-smi >/dev/null 2>&1; then
        echo "--features gpu-cuda"
      else
        echo ""
      fi
      ;;
    *) echo "" ;;
  esac
}
GPU_FEATURES="${GPU_FEATURES:-$(detect_gpu_features)}"

if [ "$BUILD_PROFILE" = "release" ]; then
  cargo build --release --workspace --bins $GPU_FEATURES
  TARGET_DIR="$INSTALL_DIR/target/release"
else
  cargo build --workspace --bins $GPU_FEATURES
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

Check the installed version:
  $BIN_DIR/bitty version

Update an existing install later with:
  git -C "$INSTALL_DIR" pull
  "$INSTALL_DIR/scripts/install_bitty.sh"

EOF

if [ -n "$CLUSTER_TOKEN" ]; then
  cat <<EOF
TCP cluster token:
  export BITTY_CLUSTER_TOKEN="$CLUSTER_TOKEN"

EOF
fi

case "$ROLE" in
  node)
    if [ -z "$MODEL_PATH" ]; then
      MODEL_PATH="$HOME/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf"
    fi
    if [ -z "$WORKER_LISTEN" ]; then
      WORKER_LISTEN="0.0.0.0:50061"
    fi
    cat <<EOF
Start Bitty:
  $BIN_DIR/bitty setup
  $BIN_DIR/bitty share home

Then use the cluster:
  $BIN_DIR/bitty run bitnet-b1.58
  $BIN_DIR/bitty cluster status
  $BIN_DIR/bitty cluster benchmark

On small CPU-only servers, keep orchestration cheap with:
  export BITTY_NODE_ROLE=coordinator
  export BITTY_DISABLE_MODEL_LAYERS=1

Bitty runs in the background. Use "$BIN_DIR/bitty stop" to stop it.
EOF
    ;;
  join)
    if [ -z "$MODEL_PATH" ]; then
      MODEL_PATH="$HOME/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf"
    fi
    if [ -z "$JOIN" ]; then
      JOIN="iroh://INVITE_FROM_BITTY_INVITE"
    fi
    cat <<EOF
Join an existing Bitty node:
  $BIN_DIR/bitty setup
  $BIN_DIR/bitty connect "$JOIN" --name home --model "$MODEL_PATH"

Then use Bitty without repeating the invite:
  $BIN_DIR/bitty run bitnet-b1.58
  $BIN_DIR/bitty cluster check
  $BIN_DIR/bitty cluster benchmark

To join as a coordinator/client helper without taking model layers:
  BITTY_DISABLE_MODEL_LAYERS=1 $BIN_DIR/bitty connect "$JOIN" --name home --model "$MODEL_PATH"
EOF
    ;;
  client)
    if [ -z "$JOIN" ]; then
      JOIN="iroh://INVITE_FROM_BITTY_INVITE"
    fi
    cat <<EOF
Save the cluster target:
  $BIN_DIR/bitty use "$JOIN" --name home

Send requests:
  $BIN_DIR/bitty run bitnet-b1.58 "Hello"
  $BIN_DIR/bitty cluster status
  $BIN_DIR/bitty cluster benchmark
EOF
    ;;
esac
