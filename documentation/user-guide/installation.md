# Installation

## Prerequisites

- **Rust toolchain**: 1.78+ (`rustup install 1.78 && rustup default 1.78`)
- **C compiler**: GCC, Clang, or MSVC (for linking C dependencies)
- **Optional — CUDA**: NVIDIA GPU with CUDA toolkit 12+
- **Optional — Metal**: macOS 13+ (Apple Silicon or AMD GPU)

## Quick Install (Linux, macOS, FreeBSD)

```bash
curl -fsSL https://raw.githubusercontent.com/askscience/bitty/main/scripts/install_bitty.sh | sh
```

## Build from Source

```bash
git clone https://github.com/askscience/bitty.git
cd bitty

# Basic build (CPU only)
cargo build --release

# With Metal support (macOS)
cargo build --release --features metal

# With CUDA support (Linux)
cargo build --release --features cuda
```

## Feature Flags

| Feature | Crate | Description |
|---------|-------|-------------|
| `metal` | bitty-candle-runtime | Apple Metal GPU support |
| `cuda` | bitty-candle-runtime | NVIDIA CUDA GPU support |
| `cpu-mkl` | bitty-candle-runtime | Intel MKL CPU acceleration |
| `cpu-accelerate` | bitty-candle-runtime | macOS Accelerate framework |

## Post-Install

```bash
# Run first-time setup
bitty setup

# Pull the default model
bitty pull bitnet-b1.58

# Verify installation
bitty run bitnet-b1.58 "Hello, world!"
```

## Directory Layout

```
~/.bitty/                    # Data directory
├── config.toml              # User configuration
├── iroh-secret.key          # Iroh P2P identity key
├── logs/
│   └── bitty.log            # Application logs (rotating)
├── models/                  # Downloaded GGUF models
│   ├── bitnet-b1.58/
│   │   └── model.gguf
│   └── ...
├── clusters.toml            # Saved cluster aliases
└── profiles/                # User-created Modelfile profiles
```

## Uninstall

```bash
# Remove all data
bitty reset

# Remove binary
rm $(which bitty)  # or delete from Cargo install directory
```
