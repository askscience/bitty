use std::sync::Arc;

use wgpu::BufferUsages;

use crate::gpu::buffer_pool::{BufferPool, GpuBuf};
use crate::gpu::pipeline::PipelineManager;
use crate::nn::bg_cache::BgCache;

const RMSNORM_WGSL: &str = include_str!("../shaders/rmsnorm.wgsl");
const QUANTIZE_WGSL: &str = include_str!("../shaders/quantize.wgsl");
const TERNARY_GEMV_WGSL: &str = include_str!("../shaders/ternary_gemv.wgsl");
const TERNARY_GEMM_WGSL: &str = include_str!("../shaders/ternary_gemm.wgsl");
const F32_MATMUL_WGSL: &str = include_str!("../shaders/f32_matmul.wgsl");

/// Linear layer: supports both ternary BitNet I2_S matmul and standard F32 matmul.
///
/// In ternary mode: RMSNorm → Quantize → Ternary MatMul → Dequantize
///   Weights are ternary {-1,0,+1} packed as 2-bit values.
/// In F32 mode: RMSNorm → F32 MatMul
///   Weights are standard F32 values (dequantized from GGUF).
pub struct BitLinear {
    device: Arc<wgpu::Device>,
    packed_weights: GpuBuf,
    weight_scales: GpuBuf,
    norm_weight: Option<GpuBuf>,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
    k_packed: usize,
    bg_cache: BgCache,
    /// True for ternary (I2_S) weights; false for F32 weights.
    ternary: bool,
}

impl BitLinear {
    pub fn new(
        device: Arc<wgpu::Device>,
        packed_weights: GpuBuf,
        weight_scales: GpuBuf,
        norm_weight: Option<GpuBuf>,
        in_dim: usize,
        out_dim: usize,
    ) -> Self {
        Self {
            device,
            packed_weights,
            weight_scales,
            norm_weight,
            in_dim,
            out_dim,
            k_packed: in_dim.div_ceil(16),
            bg_cache: BgCache::new(),
            ternary: true,
        }
    }

    /// Create a F32 linear layer (no ternary matmul).
    pub fn new_f32(
        device: Arc<wgpu::Device>,
        weight: GpuBuf,
        norm_weight: Option<GpuBuf>,
        in_dim: usize,
        out_dim: usize,
    ) -> Self {
        let dummy_scale = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_scale"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        }));
        Self {
            device,
            packed_weights: weight,
            weight_scales: dummy_scale,
            norm_weight,
            in_dim,
            out_dim,
            k_packed: 0,
            bg_cache: BgCache::new(),
            ternary: false,
        }
    }

    /// Forward pass: input [N, in_dim] f32 → output [N, out_dim] f32
    pub fn forward(
        &mut self,
        input: &GpuBuf,
        n: usize,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        if self.ternary {
            self.forward_ternary(input, n, encoder, pipelines, pool)
        } else {
            self.forward_f32(input, n, encoder, pipelines, pool)
        }
    }

    fn forward_ternary(
        &mut self,
        input: &GpuBuf,
        n: usize,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let normed = if let Some(ref _norm_w) = self.norm_weight {
            let out = pool.acquire(
                (n * self.in_dim * 4) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            );
            self.dispatch_rmsnorm(encoder, input, &out, n, pipelines);
            Some(out)
        } else {
            None
        };
        let normed_ref = normed.as_ref().unwrap_or(input);

        let quantized = pool.acquire(
            (n * self.in_dim * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let input_scales = pool.acquire(
            (n * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::UNIFORM,
        );
        self.dispatch_quantize(encoder, normed_ref, &quantized, &input_scales, n, pipelines);

        let output = pool.acquire(
            (n * self.out_dim * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        if n == 1 {
            self.dispatch_gemv(encoder, &quantized, &input_scales, &output, pipelines);
        } else {
            self.dispatch_gemm(encoder, &quantized, &input_scales, &output, n, pipelines);
        }

        output
    }

    fn forward_f32(
        &mut self,
        input: &GpuBuf,
        n: usize,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let normed = if let Some(ref _norm_w) = self.norm_weight {
            let out = pool.acquire(
                (n * self.in_dim * 4) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            );
            self.dispatch_rmsnorm(encoder, input, &out, n, pipelines);
            out
        } else {
            input.clone()
        };
        let output = pool.acquire(
            (n * self.out_dim * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        self.dispatch_f32_matmul(encoder, &normed, &output, n, pipelines);
        output
    }

    fn dispatch_f32_matmul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: usize,
        pipelines: &mut PipelineManager,
    ) {
        let entry = pipelines.get_or_create_default("f32_matmul", F32_MATMUL_WGSL);

        let params_data = [
            (n as u32).to_le_bytes(),
            (self.out_dim as u32).to_le_bytes(),
            (self.in_dim as u32).to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("f32_linear"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, &self.packed_weights),
                buf_entry(2, output),
                buf_entry(3, &params),
            ],
        });

        let total = (n * self.out_dim) as u32;
        let wg_x = total.min(65535);
        let wg_y = total.div_ceil(65535);

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    fn dispatch_rmsnorm(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: usize,
        pipelines: &mut PipelineManager,
    ) {
        let entry = pipelines.get_or_create_default("rmsnorm", RMSNORM_WGSL);
        let params = create_uniform_u32_u32_f32(&self.device, n as u32, self.in_dim as u32, 1e-5);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitlinear_rmsnorm"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, self.norm_weight.as_ref().unwrap()),
                buf_entry(2, output),
                buf_entry(3, &params),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }

    fn dispatch_quantize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        n: usize,
        pipelines: &mut PipelineManager,
    ) {
        let entry = pipelines.get_or_create_default("quantize", QUANTIZE_WGSL);
        let params = create_uniform_u32_u32(&self.device, n as u32, self.in_dim as u32);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitlinear_quantize"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, output),
                buf_entry(2, scales),
                buf_entry(3, &params),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }

    fn dispatch_gemv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        input_scales: &wgpu::Buffer,
        output: &wgpu::Buffer,
        pipelines: &mut PipelineManager,
    ) {
        let entry = pipelines.get_or_create_default("ternary_gemv", TERNARY_GEMV_WGSL);

        let params_data = [
            (self.out_dim as u32).to_le_bytes(),
            (self.in_dim as u32).to_le_bytes(),
            (self.k_packed as u32).to_le_bytes(),
        ]
        .concat();
        let params_buf = create_uniform_raw(&self.device, &params_data);

        let scale_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemv_scale"),
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(input_scales, 0, &scale_uniform, 0, 4);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitlinear_gemv"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, &self.packed_weights),
                buf_entry(1, input),
                buf_entry(2, &self.weight_scales),
                buf_entry(3, &params_buf),
                buf_entry(4, &scale_uniform),
                buf_entry(5, output),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(self.out_dim as u32, 1, 1);
    }

    fn dispatch_gemm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        input_scales: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: usize,
        pipelines: &mut PipelineManager,
    ) {
        let entry = pipelines.get_or_create_default("ternary_gemm", TERNARY_GEMM_WGSL);

        let params_data = [
            (self.out_dim as u32).to_le_bytes(),
            (n as u32).to_le_bytes(),
            (self.in_dim as u32).to_le_bytes(),
            (self.k_packed as u32).to_le_bytes(),
        ]
        .concat();
        let params_buf = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitlinear_gemm"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, &self.packed_weights),
                buf_entry(1, input),
                buf_entry(2, &self.weight_scales),
                buf_entry(3, &params_buf),
                buf_entry(4, input_scales),
                buf_entry(5, output),
            ],
        });

        let wg_m = self.out_dim.div_ceil(64) as u32;
        let wg_n = n.div_ceil(64) as u32;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(wg_m, wg_n, 1);
    }

    pub fn clear_bg_cache(&mut self) {
        self.bg_cache.clear();
    }
}

// --- Uniform buffer helpers ---

pub(crate) fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) fn create_uniform_raw(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(4).div_ceil(4) * 4) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = buffer.slice(..).get_mapped_range_mut();
        view[..data.len()].copy_from_slice(data);
    }
    buffer.unmap();
    buffer
}

pub(crate) fn create_uniform_u32_u32(device: &wgpu::Device, a: u32, b: u32) -> wgpu::Buffer {
    let data = [a.to_le_bytes(), b.to_le_bytes()].concat();
    create_uniform_raw(device, &data)
}

pub(crate) fn create_uniform_u32_u32_f32(
    device: &wgpu::Device,
    a: u32,
    b: u32,
    c: f32,
) -> wgpu::Buffer {
    let data = [a.to_le_bytes(), b.to_le_bytes(), c.to_le_bytes()].concat();
    create_uniform_raw(device, &data)
}
