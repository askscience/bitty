use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackendKind {
    Cuda,
    Rocm,
    Metal,    // candle-Metal
    Vulkan,   // wgpu-Vulkan
    Dx12,     // wgpu-DX12
    MetalWgpu,// wgpu-Metal (opt-in, slower than candle-Metal)
    Auto,
    Cpu,
}

/// Determine which GPU backend to try, in priority order.
/// `requested` is derived from the CLI flags (--vulkan -> Some(Vulkan), etc.).
/// `--gpu` alone results in Some(Auto).
pub fn select_backend(requested: Option<GpuBackendKind>) -> GpuBackendKind {
    if let Some(r) = requested {
        if r != GpuBackendKind::Auto {
            return r;
        }
    }
    // Auto-select
    auto_backend()
}

fn auto_backend() -> GpuBackendKind {
    // candle-CUDA: compiled in AND NVIDIA GPU present
    if cfg!(feature = "gpu-cuda") && has_cuda_device() {
        return GpuBackendKind::Cuda;
    }
    // candle-ROCm: compiled in AND AMD GPU on Linux
    if cfg!(feature = "gpu-rocm") && cfg!(target_os = "linux") {
        return GpuBackendKind::Rocm;
    }
    // candle-Metal: compiled in AND macOS
    if cfg!(feature = "gpu-metal") && cfg!(target_os = "macos") {
        return GpuBackendKind::Metal;
    }
    // wgpu
    if cfg!(feature = "gpu-wgpu") {
        if cfg!(target_os = "macos") {
            return GpuBackendKind::MetalWgpu;
        } else if cfg!(target_os = "windows") {
            return GpuBackendKind::Dx12;
        }
        return GpuBackendKind::Vulkan;
    }
    GpuBackendKind::Cpu
}

pub fn has_cuda_device() -> bool {
    // Quick probe: check for nvml or nvidia-smi
    std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Try to load the model on the selected GPU backend.
/// Returns Ok(text_result) if GPU succeeded, Err(original_error) if we should fall back to CPU.
pub async fn try_gpu_chat(
    backend: GpuBackendKind,
    path: &Path,
    hf_source: Option<&str>,
    messages: &[bitty_bitnet_runtime::ChatMessage],
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    repeat_penalty: f32,
    seed: Option<u64>,
    on_delta: impl FnMut(&str),
) -> Result<String, String> {
    match backend {
        GpuBackendKind::Cuda | GpuBackendKind::Metal | GpuBackendKind::Rocm => {
            // Use candle-runtime (GPU path for llama-backend models)
            try_candle_gpu(
                path, hf_source, messages, max_tokens, temperature, top_k, repeat_penalty, seed, on_delta,
            )
        }
        GpuBackendKind::Vulkan | GpuBackendKind::Dx12 | GpuBackendKind::MetalWgpu => {
            // Use wgpu-runtime
            try_wgpu_gpu(
                path, hf_source, messages, max_tokens, temperature, top_k, repeat_penalty, seed, on_delta,
            )
        }
        GpuBackendKind::Auto => try_auto_gpu(
            path, hf_source, messages, max_tokens, temperature, top_k, repeat_penalty, seed, on_delta,
        ).await,
        GpuBackendKind::Cpu => Err("CPU-only requested".to_string()),
    }
}

/// Try the candle GPU backend (CUDA/Metal/ROCm) for non-BitNet models.
fn try_candle_gpu(
    path: &Path,
    _hf_source: Option<&str>,
    _messages: &[bitty_bitnet_runtime::ChatMessage],
    _max_tokens: usize,
    _temperature: f32,
    _top_k: usize,
    _repeat_penalty: f32,
    _seed: Option<u64>,
    _on_delta: impl FnMut(&str),
) -> Result<String, String> {
    // Use the bitty-candle-runtime CandleModel directly for llama-backend models.
    // This path uses candle-core CUDA/Metal/ROCm for fast inference.
    let device = bitty_candle_runtime::auto_device();
    let _model = bitty_candle_runtime::CandleModel::load(
        &path.to_string_lossy(),
        &device,
    )
    .map_err(|e| format!("candle GPU load: {e}"))?;

    // TODO: wire up full generation loop with chat template
    // For now, this path loads successfully but the generate loop isn't wired.
    // Use the CPU backend as fallback for actual inference.

    Err("candle GPU path: model loaded, but generate loop not yet wired for llama-backend models".to_string())
}

/// Try the wgpu GPU backend (Vulkan/Metal/DX12).
fn try_wgpu_gpu(
    _path: &Path,
    _hf_source: Option<&str>,
    _messages: &[bitty_bitnet_runtime::ChatMessage],
    _max_tokens: usize,
    _temperature: f32,
    _top_k: usize,
    _repeat_penalty: f32,
    _seed: Option<u64>,
    _on_delta: impl FnMut(&str),
) -> Result<String, String> {
    #[cfg(feature = "gpu-wgpu")]
    {
        let _wgpu_backend = bitty_wgpu_runtime::WgpuDevice::new(bitty_wgpu_runtime::GpuBackend::Auto)
            .map_err(|e| format!("wgpu device: {e}"))?;
        Err("wgpu GPU path: device created, but generate loop not yet wired".to_string())
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        Err("wgpu GPU support not compiled in (enable gpu-wgpu feature)".to_string())
    }
}

/// Auto-detect: try multiple backends in priority order and return the first that works.
async fn try_auto_gpu(
    _path: &Path,
    _hf_source: Option<&str>,
    _messages: &[bitty_bitnet_runtime::ChatMessage],
    _max_tokens: usize,
    _temperature: f32,
    _top_k: usize,
    _repeat_penalty: f32,
    _seed: Option<u64>,
    _on_delta: impl FnMut(&str),
) -> Result<String, String> {
    let backends = auto_backend_order();
    for backend in backends {
        eprintln!("  trying {} backend...", backend.name());
        let result = try_gpu_chat(
            backend,
            _path,
            _hf_source,
            _messages,
            _max_tokens,
            _temperature,
            _top_k,
            _repeat_penalty,
            _seed,
            |_| {},
        )
        .await;
        match result {
            Ok(text) => return Ok(text),
            Err(e) => eprintln!("  {} failed: {e}", backend.name()),
        }
    }
    Err("no GPU backend available".to_string())
}

fn auto_backend_order() -> Vec<GpuBackendKind> {
    let mut order = Vec::new();
    if cfg!(feature = "gpu-cuda") && has_cuda_device() {
        order.push(GpuBackendKind::Cuda);
    }
    if cfg!(feature = "gpu-rocm") && cfg!(target_os = "linux") {
        order.push(GpuBackendKind::Rocm);
    }
    if cfg!(feature = "gpu-metal") && cfg!(target_os = "macos") {
        order.push(GpuBackendKind::Metal);
    }
    if cfg!(feature = "gpu-wgpu") {
        if cfg!(target_os = "macos") {
            order.push(GpuBackendKind::MetalWgpu);
        } else if cfg!(target_os = "windows") {
            order.push(GpuBackendKind::Dx12);
        }
        order.push(GpuBackendKind::Vulkan);
    }
    order
}

impl GpuBackendKind {
    pub fn name(&self) -> &'static str {
        match self {
            GpuBackendKind::Cuda => "candle-cuda",
            GpuBackendKind::Rocm => "candle-rocm",
            GpuBackendKind::Metal => "candle-metal",
            GpuBackendKind::Vulkan => "wgpu-vulkan",
            GpuBackendKind::Dx12 => "wgpu-dx12",
            GpuBackendKind::MetalWgpu => "wgpu-metal",
            GpuBackendKind::Auto => "auto",
            GpuBackendKind::Cpu => "cpu",
        }
    }

    pub fn from_cli_flag(backend: &str) -> Self {
        match backend {
            "cuda" => GpuBackendKind::Cuda,
            "rocm" => GpuBackendKind::Rocm,
            "metal" => GpuBackendKind::Metal,
            "vulkan" => GpuBackendKind::Vulkan,
            "dx12" => GpuBackendKind::Dx12,
            "metal-gpu" | "metal_wgpu" => GpuBackendKind::MetalWgpu,
            _ => GpuBackendKind::Auto,
        }
    }
}
