#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BITNET_DIR="${BITNET_DIR:-"$ROOT_DIR/external/BitNet"}"
MODEL_DIR="${MODEL_DIR:-"$BITNET_DIR/models/BitNet-b1.58-2B-4T"}"
MODEL_REPO="${MODEL_REPO:-microsoft/BitNet-b1.58-2B-4T-gguf}"
QUANT_TYPE="${QUANT_TYPE:-i2_s}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-"$BITNET_DIR/.bitty-venv"}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required. Set PYTHON_BIN=/path/to/python if needed." >&2
  exit 1
fi

if [ ! -d "$BITNET_DIR/.git" ]; then
  mkdir -p "$(dirname "$BITNET_DIR")"
  git clone --recursive https://github.com/microsoft/BitNet.git "$BITNET_DIR"
else
  git -C "$BITNET_DIR" submodule update --init --recursive
fi

BITNET_MAD="$BITNET_DIR/src/ggml-bitnet-mad.cpp"
if grep -q "int8_t \\* y_col = y + col \\* by;" "$BITNET_MAD"; then
  "$PYTHON_BIN" - "$BITNET_MAD" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
path.write_text(text.replace(
    "int8_t * y_col = y + col * by;",
    "const int8_t * y_col = y + col * by;",
))
PY
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$BITNET_DIR/requirements.txt"
"$VENV_PYTHON" -m pip install "huggingface_hub>=0.34,<1.0"

"$VENV_PYTHON" - "$MODEL_REPO" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
PY

(
  cd "$BITNET_DIR"
  "$VENV_PYTHON" setup_env.py -md "$MODEL_DIR" -q "$QUANT_TYPE"
)

echo "BitNet runtime ready:"
echo "  runtime: $BITNET_DIR"
echo "  python:  $VENV_PYTHON"
echo "  model:   $MODEL_DIR/ggml-model-$QUANT_TYPE.gguf"
echo
echo "Try:"
echo "  cargo run -p bitty-inference --bin bitty-bitnet -- --prompt \"Explain BitNet in one paragraph\" --n-predict 80"
