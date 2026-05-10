# GPU Acceleration

Bitty automatically detects your GPU and uses it for inference. No flags needed.

## Auto-detection priority

| Host | Auto-selected backend | Notes |
|---|---|---|
| macOS (Apple Silicon / AMD) | Metal | macOS 13+ required |
| Linux + NVIDIA GPU | CUDA | CUDA toolkit 12+ required |
| Linux + AMD GPU | Vulkan (via wgpu) | Mesa or AMDVLK drivers required |
| Linux + Intel GPU | Vulkan (via wgpu) | Mesa drivers required |
| Windows + NVIDIA | CUDA | |
| Windows + AMD | Vulkan / DX12 (via wgpu) | |
| Any + no GPU | CPU | Silent fallback |

## Backends

### Metal (candle)

- macOS only, fastest on Apple Silicon
- Compiled with `--features gpu-metal`
- Installed automatically by the installer on macOS

### CUDA (candle)

- NVIDIA GPUs only, Linux recommended
- Compiled with `--features gpu-cuda`
- Requires CUDA toolkit 12+ and `nvcc` on PATH
- Installed automatically if `nvidia-smi` detected

### Vulkan / DX12 / Metal (wgpu)

- Cross-platform, via `wgpu` backend
- Compiled with `--features gpu-wgpu`
- Vulkan: Linux (AMD, Intel, NVIDIA)
- DX12: Windows (AMD, Intel, NVIDIA)
- Metal via wgpu: macOS (slower than candle-Metal, opt-in with `--metal-gpu`)

### CPU

- Always available, no features needed
- Explicit with `--cpu` flag

## Explicit control

```bash
# Auto-detect (default)
bitty run gemma3:4b

# Force CPU
bitty run gemma3:4b --cpu

# Force specific backend
bitty run gemma3:4b --cuda
bitty run gemma3:4b --metal
bitty run gemma3:4b --vulkan
bitty run gemma3:4b --dx12
bitty run gemma3:4b --metal-gpu     # wgpu-Metal (opt-in)
bitty run gemma3:4b --rocm          # AMD ROCm (experimental)

# Generic
bitty run gemma3:4b --backend vulkan
```

## Build from source with GPU

```bash
# macOS (Metal)
cargo build --release --features gpu-metal

# macOS + wgpu too (Vulkan/Metal/DX12)
cargo build --release --features "gpu-metal,gpu-wgpu"

# Linux NVIDIA (CUDA)
cargo build --release --features gpu-cuda

# Linux AMD (Vulkan via wgpu)
cargo build --release --features gpu-wgpu

# All backends (large compile)
cargo build --release --features all-gpu
```

## Performance

| Backend | Relative speed | Notes |
|---|---|---|
| candle-CUDA | 100% (baseline) | NVIDIA optimized |
| candle-Metal | 80-95% | M-series chips excel |
| wgpu-Vulkan | 60-80% | Still maturing |
| wgpu-DX12 | 55-75% | |
| wgpu-Metal | 40-60% | Use candle-Metal instead |
| CPU | 5-15% | Always available |

## Troubleshooting

### "running on CPU" when I have a GPU

1. Check if GPU features are compiled: `bitty --version` shows features
2. Reinstall: the installer auto-detects GPU
3. Build from source with explicit features (see above)

### Metal: "GPU not available"

- macOS 13+ required
- Check: `system_profiler SPDisplaysDataType | grep Metal`
- Build with: `--features gpu-metal`

### CUDA: "nvcc not found"

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Or install CUDA toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

### Vulkan: "wgpu adapter not found"

```bash
# Ubuntu/Debian
sudo apt install mesa-vulkan-drivers vulkan-tools

# Verify
vulkaninfo | grep "deviceName"
```

### ROCm

ROCm support is experimental. candle-core 0.10.x does not yet expose a ROCm backend.
When it ships, enable with `--features gpu-rocm`.
