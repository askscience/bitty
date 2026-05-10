# Quickstart

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/askscience/bitty/gpu-vulkan/scripts/install_bitty.sh | bash
```

The installer auto-detects your GPU (Metal on macOS, CUDA on NVIDIA Linux) and builds with the right features.

## First run

```bash
# Pull a model
bitty pull gemma3:4b

# Interactive chat (auto-detects GPU)
bitty run gemma3:4b

# One-shot prompt
bitty run gemma3:4b "Explain quantum computing in one sentence"

# Use CPU only
bitty run gemma3:4b --cpu

# Force specific GPU backend
bitty run gemma3:4b --cuda
bitty run gemma3:4b --vulkan
```

## What to expect

On startup you'll see the selected backend:
```
  running on candle-metal
```
or if no GPU is available:
```
  running on CPU
```

## More models

```bash
# List available models
bitty pull --list

# Pull a different one
bitty pull tinyllama:1.1b
bitty pull deepseek-r1:1.5b
```

## See also

- [Installation](01-installation.md) — prerequisites, build from source, uninstall
- [GPU Acceleration](05-gpu-acceleration.md) — backends, Vulkan, performance
- [CLI Reference](03-cli-reference.md) — all commands and flags
