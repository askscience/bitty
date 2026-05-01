#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:askscience/bitty.git}"
DEST_DIR="${1:-"$HOME/bitty"}"
RUN_TESTS="${RUN_TESTS:-1}"
INSTALL_BITNET="${INSTALL_BITNET:-0}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

if [ -d "$DEST_DIR/.git" ]; then
  echo "Updating existing checkout: $DEST_DIR"
  git -C "$DEST_DIR" pull --ff-only
elif [ -e "$DEST_DIR" ]; then
  echo "Destination exists but is not a git repo: $DEST_DIR" >&2
  exit 1
else
  echo "Cloning $REPO_URL into $DEST_DIR"
  git clone "$REPO_URL" "$DEST_DIR"
fi

cd "$DEST_DIR"

if ! command -v cargo >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Rust/Cargo is required.

Install Rust from https://rustup.rs, then rerun:
  scripts/install_bitty.sh
EOF
  exit 1
fi

if [ "$RUN_TESTS" = "1" ]; then
  cargo test --workspace
fi

if [ "$INSTALL_BITNET" = "1" ]; then
  scripts/setup_bitnet.sh
fi

cat <<EOF

Bitty is installed at:
  $DEST_DIR

Useful commands:
  cargo run -p bitty-sim -- --nodes 8 --layers 16 --tokens 4
  cargo run -p bitty-inference --bin bitty-bitnet -- --prompt "Explain BitNet" --n-predict 80

For BitNet model setup, run:
  INSTALL_BITNET=1 scripts/install_bitty.sh "$DEST_DIR"
EOF
