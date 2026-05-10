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

pub use device::{GpuBackend, WgpuDevice};
pub use model::WgpuModel;

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires wgpu adapter (GPU hardware)"]
    fn wgpu_rmsnorm_parity_cpu() {
        // Test that the RMSNorm WGSL shader matches the CPU reference implementation
        use wgpu::util::DeviceExt;

        let device = crate::WgpuDevice::new(crate::GpuBackend::Auto);
        if device.is_err() { eprintln!("skipping: no wgpu adapter"); return; }
        let gpu = device.unwrap();

        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/rmsnorm.wgsl").into(),
            ),
        });
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });

        let dim = 64usize;
        let eps = 1e-5f32;
        let input: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
        let weight: Vec<f32> = vec![1.0f32; dim];

        // CPU reference
        let ms: f32 = input.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rms = (ms + eps).sqrt();
        let cpu: Vec<f32> = input.iter().zip(&weight).map(|(x, w)| x / rms * w).collect();

        // GPU dispatch
        let in_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&weight),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dim as u32, eps.to_bits(), 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
            ],
        });

        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
          pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(1, 1, 1); }
        gpu.queue.submit(std::iter::once(enc.finish()));

        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (dim * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut enc2 = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc2.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (dim * 4) as u64);
        gpu.queue.submit(std::iter::once(enc2.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_out: &[f32] = bytemuck::cast_slice(&data);
        let gpu_vec = gpu_out[..dim].to_vec();
        drop(data); staging.unmap();

        for i in 0..dim {
            let diff = (cpu[i] - gpu_vec[i]).abs();
            assert!(diff < 1e-3, "RMSNorm mismatch at {i}: CPU={} GPU={} diff={diff:e}", cpu[i], gpu_vec[i]);
        }
    }
}
