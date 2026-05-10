# Installation

## Prerequisites

- **Rust 1.78+**: `rustup install 1.78 && rustup default 1.78`
- **C compiler**: GCC, Clang, or MSVC
- **Optional — GPU**: see [GPU Acceleration](05-gpu-acceleration.md)

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/askscience/bitty/gpu-vulkan/scripts/install_bitty.sh | bash
```

The installer auto-detects your GPU and builds with the right features.

**Installer options:**

```bash
# Specify branch
BRANCH=gpu-vulkan bash install_bitty.sh

# Debug build
bash install_bitty.sh --debug

# Custom paths
INSTALL_DIR="$HOME/src/bitty" BIN_DIR="$HOME/.local/bin" bash install_bitty.sh

# Skip system dependency installation
bash install_bitty.sh --no-system-deps
```

## Build from Source

```bash
git clone https://github.com/askscience/bitty.git
cd bitty

# CPU only
cargo build --release

# With Metal (macOS) — auto-detected by installer
cargo build --release --features gpu-metal

# With CUDA (Linux NVIDIA)
cargo build --release --features gpu-cuda

# With Vulkan/DX12/Metal via wgpu
cargo build --release --features gpu-wgpu

# All backends
cargo build --release --features all-gpu
```

## Post-Install

```bash
bitty setup                  # First-time setup
bitty pull gemma3:4b         # Pull a model
bitty run gemma3:4b          # Interactive chat (auto-detects GPU)
```

## Directory Layout

```
~/.bitty/
├── config.toml              # User configuration
├── iroh-secret.key          # P2P identity key
├── logs/bitty.log           # Application logs
├── models/                  # Downloaded GGUF models
│   ├── gemma-3-4b-it-Q4_K_M.gguf
│   └── ...
├── clusters.toml            # Saved cluster aliases
└── profiles/                # Custom Modelfile profiles
```

## Feature Flags

| Feature | Crate | What |
|---|---|---|
| `gpu-cuda` | bitty-cli → bitty-candle-runtime | NVIDIA CUDA |
| `gpu-rocm` | bitty-cli → bitty-candle-runtime | AMD ROCm (experimental) |
| `gpu-metal` | bitty-cli → bitty-candle-runtime | Apple Metal |
| `gpu-wgpu` | bitty-cli → bitty-wgpu-runtime | Vulkan / DX12 / Metal via wgpu |
| `all-gpu` | bitty-cli | All GPU backends |

## Uninstall

```bash
bitty reset          # Remove ~/.bitty
rm $(which bitty)    # Remove binary
```
