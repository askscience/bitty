//! Cross-GPU inference backend via wgpu.
//!
//! Supports Vulkan, Metal, DX12, and WebGPU through a single Rust API.
//! Quantized matmul kernels handle Q4_K, Q6_K, Q8_0, F16, and F32 formats
//! directly on the GPU without pre-dequantizing weights.
//!
//! Architecture:
//! - `device.rs`  — GPU device selection via wgpu adapter enumeration
//! - `model.rs`   — GGUF loading, forward pass, generation loop
//! - `sampler.rs` — GPU-accelerated sampling from logits
//! - `shaders/`   — Slang WGSL shader sources (generated WGSL committed to repo)

mod device;
mod model;
mod sampler;

pub use device::WgpuDevice;
pub use model::WgpuModel;
