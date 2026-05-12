use crate::device::{GpuBackend, WgpuDevice};
use crate::sampler;
use bitty_candle_runtime::Tokenizer;
use bitty_gguf_loader::{BackendKind, InferenceBackend, RopeStyle};
use std::collections::HashMap;
use std::path::Path;
use wgpu::util::DeviceExt;

pub struct WgpuModel {
    device: WgpuDevice,
    metadata: GpuMetadata,
    tokenizer: Tokenizer,
    weights: GpuWeights,
    pipelines: GpuPipelines,
    gbuf: GpuBuffers,
}

#[derive(Debug, Clone)]
struct GpuMetadata {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_seq_len: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_style: RopeStyle,
    embedding_scale: Option<f32>,
}

struct GpuWeights {
    embed_tokens: wgpu::Buffer,
    final_norm: wgpu::Buffer,
    layers: Vec<GpuLayer>,
    lm_head: Option<wgpu::Buffer>,
    cpu_embed: Vec<f32>,
    cpu_lm_head: Option<Vec<f32>>,
    cpu_final_norm: Vec<f32>,
}

struct GpuLayer {
    input_ln: wgpu::Buffer,
    post_attn_ln: wgpu::Buffer,
    q_proj: QuantTensor,
    k_proj: QuantTensor,
    v_proj: QuantTensor,
    o_proj: QuantTensor,
    up_proj: QuantTensor,
    gate_proj: QuantTensor,
    down_proj: QuantTensor,
    q_norm: Option<wgpu::Buffer>,
    k_norm: Option<wgpu::Buffer>,
    cpu_input_ln: Vec<f32>,
    cpu_post_attn_ln: Vec<f32>,
}

struct QuantTensor {
    buffer: wgpu::Buffer,
    ggml_type: u32,
    in_dim: u32,
    out_dim: u32,
}

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum WgpuModelError {
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Load error: {0}")]
    Load(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
}

// Compute pipeline handles
struct GpuPipelines {
    rmsnorm: wgpu::ComputePipeline,
    embedding: wgpu::ComputePipeline,
    matmul_f32: wgpu::ComputePipeline,
    matmul_q4k: wgpu::ComputePipeline,
    matmul_q6k: wgpu::ComputePipeline,
    matmul_q8_0: wgpu::ComputePipeline,
    matmul_q4_0: wgpu::ComputePipeline,
    matmul_q5k: wgpu::ComputePipeline,
    matmul_f16: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
    swiglu: wgpu::ComputePipeline,
    add_bias: wgpu::ComputePipeline,
    softcap: wgpu::ComputePipeline,
}

impl GpuPipelines {
    fn create(device: &wgpu::Device) -> Self {
        let make = |label: &str, src: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let quant_shader = |t: u32, label: &str| -> wgpu::ComputePipeline {
            let src = crate::matmul_shader::generate_matmul_shader(t)
                .unwrap_or_else(|| panic!("no shader template for GGML type {t}"));
            make(label, &src)
        };

        Self {
            rmsnorm:     make("rmsnorm",    include_str!("shaders/rmsnorm.wgsl")),
            embedding:   make("embedding",  include_str!("shaders/embedding.wgsl")),
            matmul_f32:  make("matmul_f32", include_str!("shaders/matmul_f32.wgsl")),
            matmul_q4k:  quant_shader(12, "matmul_q4k"),
            matmul_q6k:  quant_shader(14, "matmul_q6k"),
            matmul_q8_0: quant_shader(8,  "matmul_q8_0"),
            matmul_q4_0: quant_shader(2,  "matmul_q4_0"),
            matmul_q5k:  quant_shader(13, "matmul_q5k"),
            matmul_f16:  make("matmul_f16", include_str!("shaders/matmul_f16.wgsl")),
            rope:        make("rope",       include_str!("shaders/rope.wgsl")),
            attention:   make("attention",  include_str!("shaders/attention.wgsl")),
            swiglu:      make("swiglu",     include_str!("shaders/swiglu.wgsl")),
            add_bias:    make("add_bias",   include_str!("shaders/add_bias.wgsl")),
            softcap:     make("softcap",    include_str!("shaders/softcap.wgsl")),
        }
    }
}

struct GpuBuffers {
    hidden: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    ffn_inter: wgpu::Buffer,
    ffn_out: wgpu::Buffer,
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    dim: usize,
    kv_hd: usize,
    nk: usize,
    max_seq: usize,
    num_layers: usize,
}

impl GpuBuffers {
    fn new(device: &wgpu::Device, dim: usize, num_heads: usize, nk: usize, hd: usize, max_seq: usize, num_layers: usize) -> Self {
        let buf = |label: &str, count: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label), size: (count.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let qk_dim = num_heads * hd;
        let kv_stride = num_layers * max_seq * nk * hd;
        Self {
            hidden: buf("gbuf_hidden", dim),
            q: buf("gbuf_q", qk_dim),
            k: buf("gbuf_k", qk_dim),
            v: buf("gbuf_v", qk_dim),
            attn_out: buf("gbuf_attn_out", dim),
            ffn_inter: buf("gbuf_ffn_inter", dim * 4),
            ffn_out: buf("gbuf_ffn_out", dim),
            k_cache: buf("gbuf_k_cache", kv_stride),
            v_cache: buf("gbuf_v_cache", kv_stride),
            dim, kv_hd: nk * hd, nk, max_seq, num_layers,
        }
    }
}

impl WgpuModel {
    pub fn load(path: &Path, hf_model_id: Option<&str>, backend: GpuBackend) -> Result<Self, String> {
        let gpu = WgpuDevice::new(backend)
            .map_err(|e| format!("wgpu device: {e}"))?;
        let tokenizer = Tokenizer::from_gguf_path(path, hf_model_id)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let pipelines = GpuPipelines::create(&gpu.device);

        let file = std::fs::File::open(path)
            .map_err(|e| format!("open: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("mmap: {e}"))?;

        let (meta, weights) = load_gguf_to_gpu(&gpu, &mmap)?;

        let gbuf = GpuBuffers::new(&gpu.device, meta.hidden_size, meta.num_heads, meta.num_kv_heads, meta.head_dim, meta.max_seq_len, meta.num_layers);

        Ok(Self { device: gpu, metadata: meta, tokenizer, weights, pipelines, gbuf })
    }

    pub fn generate_from_ids(
        &mut self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        on_delta: impl FnMut(&str),
    ) -> Result<String, String> {
        let d = self.metadata.hidden_size;
        let num_layers = self.metadata.num_layers;
        let n_kv = self.metadata.num_kv_heads;
        let hd = self.metadata.head_dim;

        // Allocate KV cache (CPU-side for now)
        let mut k_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::new()).collect();
        let mut v_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::new()).collect();

        // Allocate working buffers
        let mut hidden = vec![0f32; d];
        let mut generated = Vec::new();
        let mut emitted = String::new();

        let eos = self.tokenizer.eos_token_id();
        let eot = self.tokenizer.eot_token_id();
        let im_end = self.tokenizer.im_end_token_id();

        // Process prompt
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            hidden = embed_token(tid as usize, &self.weights.cpu_embed, d, self.metadata.vocab_size, self.metadata.embedding_scale);

            for li in 0..num_layers {
                let layer = &self.weights.layers[li];
                hidden = self.forward_layer(
                    &hidden, pos, li, layer, &mut k_cache[li], &mut v_cache[li],
                    &self.metadata,
                )?;
            }

            // First generated token: sample from final hidden
            if pos == prompt_ids.len() - 1 {
                let _logits = compute_logits_on_cpu(
                    &hidden, &self.weights.cpu_final_norm,
                    self.weights.cpu_lm_head.as_deref(), &self.weights.cpu_embed,
                    d, self.metadata.vocab_size, self.metadata.rms_norm_eps,
                );
                return self.generate_loop(
                    max_tokens, temperature, top_k,
                    &mut hidden, d, num_layers, n_kv, hd,
                    &mut k_cache, &mut v_cache,
                    &mut generated, &mut emitted,
                    eos, eot, im_end,
                    on_delta,
                );
            }
        }
        Ok(emitted)
    }

    fn forward_layer(
        &self, hidden: &[f32], pos: usize, _li: usize,
        layer: &GpuLayer,
        k_cache: &mut Vec<f32>, v_cache: &mut Vec<f32>,
        meta: &GpuMetadata,
    ) -> Result<Vec<f32>, String> {
        let d = meta.hidden_size;
        let nh = meta.num_heads;
        let nk = meta.num_kv_heads;
        let hd = meta.head_dim;
        let eps = meta.rms_norm_eps;

        // 1. RMSNorm → Q/K/V matmul (GPU)
        let normed = gpu_rmsnorm(&self.device, &self.pipelines.rmsnorm, hidden, &layer.input_ln, d, eps)?;
        let q = gpu_matmul(&self.device, &self.pipelines, &normed, &layer.q_proj, d)?;
        let k = gpu_matmul(&self.device, &self.pipelines, &normed, &layer.k_proj, d)?;
        let v = gpu_matmul(&self.device, &self.pipelines, &normed, &layer.v_proj, d)?;

        // 2. Q/K norms (if present — CPU since norm weights are tiny)
        let mut q = if let Some(ref _qn) = layer.q_norm { apply_norm_on_cpu(&q, &layer.cpu_input_ln, eps) } else { q };
        let mut k = if let Some(ref _kn) = layer.k_norm { apply_norm_on_cpu(&k, &layer.cpu_input_ln, eps) } else { k };

        // 3. RoPE on GPU (Q and K concatenated, dispatched once)
        let mut qk = q.clone();
        qk.extend_from_slice(&k);
        let num_heads = nh as u32;
        let num_kv_heads = nk as u32;
        let rope_style = match meta.rope_style { RopeStyle::Neox => 0u32, RopeStyle::Interleaved => 1u32 };
        gpu_rope_inplace(&self.device, &self.pipelines.rope, &mut qk, hd, num_heads, num_kv_heads, pos, meta.rope_theta, rope_style)?;
        let q_len = num_heads as usize * hd;
        q = qk[..q_len].to_vec();
        k = qk[q_len..].to_vec();

        // 4. Store K/V in CPU cache for attention (GPU-resident cache planned for Phase 1.6 follow-up)
        let kv_pos = pos * nk * hd;
        if kv_pos + k.len() > k_cache.len() { k_cache.resize(kv_pos + k.len(), 0.0); }
        if kv_pos + v.len() > v_cache.len() { v_cache.resize(kv_pos + v.len(), 0.0); }
        k_cache[kv_pos..kv_pos + k.len()].copy_from_slice(&k);
        v_cache[kv_pos..kv_pos + v.len()].copy_from_slice(&v);

        let seq_len = pos + 1;
        // GPU fused attention
        let attn_out = gpu_attention(
            &self.device, &self.pipelines.attention,
            &q, &k_cache, &v_cache,
            nh as u32, nk as u32, hd as u32, seq_len as u32,
        )?;

        // 5. O projection (GPU)
        let block_out = gpu_matmul(&self.device, &self.pipelines, &attn_out, &layer.o_proj, d)?;

        // 6. Residual (CPU — tiny op)
        let mut x1 = vec![0f32; d];
        for i in 0..d { x1[i] = hidden[i] + block_out[i]; }

        // 7. FFN: RMSNorm → gate/up matmuls (GPU) → SwiGLU (GPU)
        let ffn_normed = gpu_rmsnorm(&self.device, &self.pipelines.rmsnorm, &x1, &layer.post_attn_ln, d, eps)?;
        let gate = gpu_matmul(&self.device, &self.pipelines, &ffn_normed, &layer.gate_proj, d)?;
        let up = gpu_matmul(&self.device, &self.pipelines, &ffn_normed, &layer.up_proj, d)?;

        let inter = gate.len().min(up.len());
        let activated = gpu_swiglu(&self.device, &self.pipelines.swiglu, &gate, &up, inter)?;

        // 8. Down projection (GPU)
        let ffn_out = gpu_matmul(&self.device, &self.pipelines, &activated, &layer.down_proj, inter)?;

        // 9. Residual (CPU — tiny op)
        let mut out = vec![0f32; d];
        for i in 0..d { out[i] = x1[i] + ffn_out[i]; }
        Ok(out)
    }

    fn generate_loop(
        &self,
        max_tokens: usize, temperature: f32, top_k: usize,
        hidden: &mut Vec<f32>, d: usize, num_layers: usize, _n_kv: usize, _hd: usize,
        k_cache: &mut [Vec<f32>], v_cache: &mut [Vec<f32>],
        generated: &mut Vec<u32>, emitted: &mut String,
        eos: u32, eot: Option<u32>, im_end: Option<u32>,
        mut on_delta: impl FnMut(&str),
    ) -> Result<String, String> {
        for _step in 0..max_tokens {
            let normed_vec = gpu_rmsnorm(&self.device, &self.pipelines.rmsnorm, hidden, &self.weights.final_norm, d, self.metadata.rms_norm_eps)?;
            let logits = compute_logits_on_cpu(
                &normed_vec, &self.weights.cpu_final_norm,
                self.weights.cpu_lm_head.as_deref(), &self.weights.cpu_embed,
                d, self.metadata.vocab_size, self.metadata.rms_norm_eps,
            );

            let next = sampler::sample(&logits, temperature, top_k);
            if next == eos || Some(next) == eot || Some(next) == im_end { break; }
            generated.push(next);

            // Stream decode
            if let Ok(full) = self.tokenizer.decode(generated) {
                if full.len() > emitted.len() && full.starts_with(emitted.as_str()) {
                    let tail = &full[emitted.len()..];
                    if !tail.ends_with('\u{FFFD}') { on_delta(tail); *emitted = full; }
    }
}

impl InferenceBackend for WgpuModel {
    type Error = String;

    fn load(path: &Path, hf_source: Option<&str>) -> std::result::Result<Self, Self::Error> {
        Self::load(path, hf_source, GpuBackend::Auto)
    }

    fn forward(&mut self, token_ids: &[u32]) -> std::result::Result<Vec<f32>, Self::Error> {
        // Generate one step and return logits
        let temp = 0.0;
        let result = self.generate_from_ids(token_ids, 1, temp, 1, |_| {})?;
        Ok(vec![0.0]) // placeholder — WGPU path uses full generate loop
    }

    fn reset_kv_cache(&mut self) {
        // KV cache is managed per-request via Vec<Vec<f32>> in generate_from_ids
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.metadata.vocab_size
    }
}

            // Embed next token
            *hidden = embed_token(next as usize, &self.weights.cpu_embed, d, self.metadata.vocab_size, self.metadata.embedding_scale);
            let pos = prompt_ids_len_from_cache(k_cache, self.metadata.num_kv_heads, self.metadata.head_dim);

            for li in 0..num_layers {
                let layer = &self.weights.layers[li];
                *hidden = self.forward_layer(
                    hidden, pos, li, layer, &mut k_cache[li], &mut v_cache[li],
                    &self.metadata,
                )?;
            }
        }
        Ok(emitted.clone())
    }
}

fn prompt_ids_len_from_cache(k_cache: &[Vec<f32>], nk: usize, hd: usize) -> usize {
    let kv_stride = nk * hd;
    if kv_stride == 0 { return 0; }
    k_cache.first().map(|k| k.len() / kv_stride).unwrap_or(0)
}

// ─── GPU dispatch helpers ───

fn gpu_rmsnorm(
    gpu: &WgpuDevice, pipeline: &wgpu::ComputePipeline,
    input: &[f32], weight: &wgpu::Buffer, dim: usize, eps: f32,
) -> Result<Vec<f32>, String> {
    let in_buf = upload_f32(&gpu.device, input, "rms_in");
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rms_out"),
        size: (dim * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let config_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rms_cfg"),
        contents: bytemuck::cast_slice(&[dim as u32, eps.to_bits(), 0u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: config_buf.as_entire_binding() },
        ],
    });

    dispatch_compute(&gpu, pipeline, &bind_group, 1);
    readback_f32(&gpu, &out_buf, dim)
}

fn gpu_matmul(
    gpu: &WgpuDevice, pipelines: &GpuPipelines,
    input: &[f32], weight: &QuantTensor, _in_dim: usize,
) -> Result<Vec<f32>, String> {
    let actual_in = weight.in_dim as usize;
    let out_dim = weight.out_dim as usize;
    if input.len() < actual_in { return Err(format!("matmul dim mismatch: input {} < weight in_dim {}", input.len(), actual_in)); }

    let pipeline = match weight.ggml_type {
        0  => &pipelines.matmul_f32,   // F32
        1  => &pipelines.matmul_f16,   // F16
        2  => &pipelines.matmul_q4_0,  // Q4_0
        8  => &pipelines.matmul_q8_0,  // Q8_0
        12 => &pipelines.matmul_q4k,   // Q4_K
        13 => &pipelines.matmul_q5k,   // Q5_K
        14 => &pipelines.matmul_q6k,   // Q6_K
        _  => &pipelines.matmul_f32,   // fallback
    };

    let in_buf = upload_f32(&gpu.device, &input[..actual_in], "matmul_in");
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul_out"),
        size: (out_dim * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let config_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_cfg"),
        contents: bytemuck::cast_slice(&[actual_in as u32, out_dim as u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: weight.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: config_buf.as_entire_binding() },
        ],
    });

    dispatch_compute(gpu, pipeline, &bind_group, out_dim.div_ceil(256));
    readback_f32(gpu, &out_buf, out_dim)
}

fn gpu_embedding(
    gpu: &WgpuDevice, pipeline: &wgpu::ComputePipeline,
    tokens: &[u32], embed_tokens: &wgpu::Buffer, dim: usize, scale: Option<f32>,
) -> Result<Vec<f32>, String> {
    let n = tokens.len();
    let tok_buf = upload_u32(&gpu.device, tokens, "emb_tokens");
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("emb_out"),
        size: (n * dim * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let config_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("emb_cfg"),
        contents: bytemuck::cast_slice(&[dim as u32, scale.unwrap_or(1.0).to_bits()]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tok_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: embed_tokens.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: config_buf.as_entire_binding() },
        ],
    });

    dispatch_compute(gpu, pipeline, &bind_group, (n * dim).div_ceil(64));
    readback_f32(gpu, &out_buf, n * dim)
}

fn gpu_attention(
    gpu: &WgpuDevice, pipeline: &wgpu::ComputePipeline,
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    nh: u32, nk: u32, hd: u32, seq_len: u32,
) -> Result<Vec<f32>, String> {
    let out_dim = (nh * hd) as usize;
    let q_buf = upload_f32(&gpu.device, q, "attn_q");
    let k_buf = upload_f32(&gpu.device, k_cache, "attn_k");
    let v_buf = upload_f32(&gpu.device, v_cache, "attn_v");
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn_out"),
        size: (out_dim * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let scale = 1.0 / (hd as f32).sqrt();
    let config_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attn_cfg"),
        contents: bytemuck::cast_slice(&[nh, nk, hd, seq_len, scale.to_bits()]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: config_buf.as_entire_binding() },
        ],
    });

    dispatch_compute(gpu, pipeline, &bind_group, nh.div_ceil(64) as usize);
    readback_f32(gpu, &out_buf, out_dim)
}

fn gpu_swiglu(
    gpu: &WgpuDevice, pipeline: &wgpu::ComputePipeline,
    gate: &[f32], up: &[f32], dim: usize,
) -> Result<Vec<f32>, String> {
    let gate_buf = upload_f32(&gpu.device, gate, "swiglu_gate");
    let up_buf = upload_f32(&gpu.device, up, "swiglu_up");
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("swiglu_out"),
        size: (dim * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dim_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("swiglu_dim"),
        contents: bytemuck::cast_slice(&[dim as u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gate_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: up_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() },
        ],
    });

    dispatch_compute(gpu, pipeline, &bind_group, dim.div_ceil(64));
    readback_f32(gpu, &out_buf, dim)
}

fn gpu_attention_buf(
    gpu: &WgpuDevice, pipeline: &wgpu::ComputePipeline,
    q_buf: &wgpu::Buffer, k_cache_buf: &wgpu::Buffer, v_cache_buf: &wgpu::Buffer,
    out_buf: &wgpu::Buffer, nh: u32, nk: u32, hd: u32, seq_len: u32,
) {
    let scale = 1.0 / (hd as f32).sqrt();
    let config_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attn_cfg"),
        contents: bytemuck::cast_slice(&[nh, nk, hd, seq_len, scale.to_bits()]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_cache_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_cache_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: config_buf.as_entire_binding() },
        ],
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(nh.div_ceil(64), 1, 1);
    }
    gpu.queue.submit(std::iter::once(encoder.finish()));
}

fn dispatch_compute(gpu: &WgpuDevice, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, workgroups: usize) {
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.max(1) as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(encoder.finish()));
}

fn readback_f32(gpu: &WgpuDevice, buf: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
    let size = (count * 4) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    rx.recv().map_err(|_| "map failed")?.map_err(|e| format!("map: {e}"))?;

    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    let result = floats[..count].to_vec();
    drop(data);
    staging.unmap();
    Ok(result)
}

fn upload_f32(device: &wgpu::Device, data: &[f32], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn gpu_rope_inplace(
    gpu: &WgpuDevice, pipeline: &wgpu::ComputePipeline,
    qk: &mut Vec<f32>, head_dim: usize, num_q_heads: u32, num_kv_heads: u32,
    pos: usize, theta: f32, style: u32,
) -> Result<(), String> {
    let rp = head_dim / 2;
    // Precompute cos/sin for this position
    let mut cos_sin = vec![0f32; rp * 2];
    for i in 0..rp {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        cos_sin[i * 2] = (pos as f32 * freq).cos();
        cos_sin[i * 2 + 1] = (pos as f32 * freq).sin();
    }

    let qk_len = qk.len();
    let qk_buf = upload_f32(&gpu.device, qk, "rope_qk");
    let cs_buf = upload_f32(&gpu.device, &cos_sin, "rope_cos_sin");
    let config_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_cfg"),
        contents: bytemuck::cast_slice(&[head_dim as u32, num_q_heads, num_kv_heads, style]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope_out"),
        size: (qk_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: qk_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cs_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: config_buf.as_entire_binding() },
        ],
    });

    // Copy input → output in a single dispatch
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&qk_buf, 0, &out_buf, 0, (qk_len * 4) as u64);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        // need to bind output buffer instead — let me rework this
    }

    // Since the shader reads from binding 0 and writes back, I need a different approach.
    // For now: upload, dispatch in-place (binding 0 is read_write), then read back.
    // The shader writes directly to binding 0.
    let bind_group2 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: qk_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cs_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: config_buf.as_entire_binding() },
        ],
    });

    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group2, &[]);
        let pairs = (num_q_heads + num_kv_heads) as usize * rp;
        pass.dispatch_workgroups(pairs.div_ceil(64).max(1) as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));

    let result = readback_f32(gpu, &qk_buf, qk_len)?;
    qk.copy_from_slice(&result[..qk_len]);
    Ok(())
}

fn upload_u32(device: &wgpu::Device, data: &[u32], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

// ─── CPU-side helpers ───

fn compute_logits_on_cpu(
    hidden: &[f32], cpu_final_norm: &[f32],
    cpu_lm_head: Option<&[f32]>, cpu_embed: &[f32],
    d: usize, vocab: usize, eps: f32,
) -> Vec<f32> {
    let normed = apply_norm_on_cpu(hidden, cpu_final_norm, eps);
    let head_weight = cpu_lm_head.unwrap_or(cpu_embed);
    let mut logits = vec![0f32; vocab];
    for v in 0..vocab {
        let offset = v * d;
        let end = (offset + d).min(head_weight.len());
        let mut dot = 0f32;
        for i in 0..d.min(end - offset) {
            dot += normed[i] * head_weight[offset + i];
        }
        logits[v] = dot;
    }
    logits
}

fn embed_token(tid: usize, cpu_embed: &[f32], d: usize, vocab: usize, scale: Option<f32>) -> Vec<f32> {
    let emb_offset = tid.min(vocab - 1) * d;
    let end = (emb_offset + d).min(cpu_embed.len());
    if emb_offset >= cpu_embed.len() {
        return vec![0f32; d];
    }
    let mut h = cpu_embed[emb_offset..end].to_vec();
    if h.len() < d { h.resize(d, 0.0); }
    if let Some(s) = scale { for v in &mut h { *v *= s; } }
    h
}

fn apply_norm_on_cpu(x: &[f32], cpu_norm: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    x.iter().zip(cpu_norm.iter())
        .map(|(xi, wi)| xi / rms * wi)
        .collect()
}

// ─── GGUF loader ───

fn load_gguf_to_gpu(gpu: &WgpuDevice, mmap: &[u8]) -> Result<(GpuMetadata, GpuWeights), String> {
    use bitty_model::gguf::parse_gguf_bytes;
    let gguf = parse_gguf_bytes(mmap).map_err(|e| format!("GGUF: {e}"))?;
    let meta = extract_gpu_metadata(&gguf)?;

    let (tensor_map, _data_offset) = build_tensor_map(mmap, &gguf);

    let get_f32 = |name: &str, n: usize| -> Result<(wgpu::Buffer, Vec<f32>), String> {
        let (off, len, _ggml) = tensor_map.get(name).ok_or_else(|| format!("Missing: {name}"))?;
        let off = *off;
        let len = *len;
        let data: &[f32] = bytemuck::cast_slice(&mmap[off..off + len.min(n * 4)]);
        let cpu = data[..n].to_vec();
        let buf = upload_f32(&gpu.device, &cpu, name);
        Ok((buf, cpu))
    };

    let (embed_tokens, cpu_embed) = get_f32("token_embd.weight", meta.vocab_size * meta.hidden_size)?;
    let (final_norm, cpu_final_norm) = get_f32("output_norm.weight", meta.hidden_size)?;

    let mut layers = Vec::with_capacity(meta.num_layers);
    for i in 0..meta.num_layers {
        let p = format!("blk.{}.", i);
        let (input_ln, cpu_input_ln) = get_f32(&format!("{p}attn_norm.weight"), meta.hidden_size)?;
        let (post_attn_ln, cpu_post_attn_ln) = get_f32(&format!("{p}ffn_norm.weight"), meta.hidden_size)?;

        let q_dim = meta.num_heads * meta.head_dim;
        let k_dim = meta.num_kv_heads * meta.head_dim;
        let v_dim = meta.num_kv_heads * meta.head_dim;
        let o_dim = meta.num_heads * meta.head_dim;

        let q_proj = load_quant_tensor(gpu, mmap, &tensor_map, &format!("{p}attn_q.weight"), meta.hidden_size as u32, q_dim as u32)?;
        let k_proj = load_quant_tensor(gpu, mmap, &tensor_map, &format!("{p}attn_k.weight"), meta.hidden_size as u32, k_dim as u32)?;
        let v_proj = load_quant_tensor(gpu, mmap, &tensor_map, &format!("{p}attn_v.weight"), meta.hidden_size as u32, v_dim as u32)?;
        let o_proj = load_quant_tensor(gpu, mmap, &tensor_map, &format!("{p}attn_output.weight"), o_dim as u32, meta.hidden_size as u32)?;
        let up_proj = load_quant_tensor(gpu, mmap, &tensor_map, &format!("{p}ffn_up.weight"), meta.hidden_size as u32, meta.intermediate_size as u32)?;
        let gate_proj = load_quant_tensor(gpu, mmap, &tensor_map, &format!("{p}ffn_gate.weight"), meta.hidden_size as u32, meta.intermediate_size as u32)?;
        let down_proj = load_quant_tensor(gpu, mmap, &tensor_map, &format!("{p}ffn_down.weight"), meta.intermediate_size as u32, meta.hidden_size as u32)?;

        layers.push(GpuLayer {
            input_ln, post_attn_ln,
            q_proj, k_proj, v_proj, o_proj,
            up_proj, gate_proj, down_proj,
            q_norm: None, k_norm: None,
            cpu_input_ln, cpu_post_attn_ln,
        });
    }

    let (lm_head, cpu_lm_head) = tensor_map.get("output.weight")
        .map(|(off, len, _ggml)| {
            let off = *off;
            let len = *len;
            let data: &[f32] = bytemuck::cast_slice(&mmap[off..off + len.min(meta.vocab_size * meta.hidden_size * 4)]);
            let cpu = data[..(meta.vocab_size * meta.hidden_size).min(data.len())].to_vec();
            let buf = upload_f32(&gpu.device, &cpu, "lm_head");
            (Some(buf), Some(cpu))
        })
        .unwrap_or((None, None));

    Ok((meta, GpuWeights { embed_tokens, final_norm, layers, lm_head, cpu_embed, cpu_lm_head, cpu_final_norm }))
}

fn load_quant_tensor(
    gpu: &WgpuDevice, mmap: &[u8], tensor_map: &HashMap<String, (usize, usize, u32)>,
    name: &str, in_dim: u32, out_dim: u32,
) -> Result<QuantTensor, String> {
    let (off, len, ggml_type) = tensor_map.get(name).copied().ok_or_else(|| format!("Missing: {name}"))?;

    let buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(name),
        contents: &mmap[off..(off + len).min(mmap.len())],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    Ok(QuantTensor { buffer: buf, ggml_type, in_dim, out_dim })
}

fn extract_gpu_metadata(gguf: &bitty_model::gguf::GgufFileMetadata) -> Result<GpuMetadata, String> {
    let m = &gguf.metadata;
    let arch = m.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("llama");
    let get = |k: &str| m.get(k).and_then(|v| v.as_u32()).unwrap_or(0) as usize;

    let hidden_size = get(&format!("{arch}.embedding_length")).max(2048);
    let num_layers = get(&format!("{arch}.block_count")).max(1);
    let nh = get(&format!("{arch}.attention.head_count")).max(1);
    let nk = get(&format!("{arch}.attention.head_count_kv")).max(1);
    let hd = if nh > 0 { hidden_size / nh } else { 64 };
    let inter = get(&format!("{arch}.feed_forward_length")).max(1);
    let vocab = m.get("tokenizer.ggml.tokens").and_then(|v| v.as_string_array()).map(|a| a.len()).unwrap_or(128256);
    let max_seq = get(&format!("{arch}.context_length")).max(2048);
    let eps = m.get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(1e-5);
    let theta = m.get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(10000.0);
    let rope_style = match arch {
        "llama" | "mistral" | "tinyllama" | "smollm" => RopeStyle::Interleaved,
        _ => RopeStyle::Neox,
    };
    let embedding_scale = if arch.starts_with("gemma") { Some((hidden_size as f32).sqrt()) } else { None };

    Ok(GpuMetadata { hidden_size, num_layers, num_heads: nh, num_kv_heads: nk, head_dim: hd,
        intermediate_size: inter, vocab_size: vocab, max_seq_len: max_seq, rms_norm_eps: eps,
        rope_theta: theta, rope_style, embedding_scale })
}

fn build_tensor_map(mmap: &[u8], gguf: &bitty_model::gguf::GgufFileMetadata) -> (HashMap<String, (usize, usize, u32)>, usize) {
    use bitty_model::gguf::bytes_per_element;
    let mut map = HashMap::new();
    let mut pos: usize = 12;
    let tensor_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
    let metadata_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap());
    pos += 8;
    for _ in 0..metadata_count {
        let key_len = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8 + key_len;
        let val_type = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
        pos += 4;
        pos = skip_metadata_val(mmap, pos, val_type);
    }
    for _ in 0..tensor_count {
        let name_len = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let name = std::str::from_utf8(&mmap[pos..pos + name_len]).unwrap_or("").to_string();
        pos += name_len;
        let dim_count = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mut dims = Vec::new();
        for _ in 0..dim_count { dims.push(u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize); pos += 8; }
        let ggml_type = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let offset = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let elem_count: usize = dims.iter().product();
        let data_offset = compute_data_offset(mmap, gguf.alignment);
        let tensor_start = data_offset + offset as usize;
        let byte_len = (elem_count as f64 * bytes_per_element(ggml_type)).ceil() as usize;
        map.insert(name, (tensor_start, byte_len, ggml_type));
    }
    (map, compute_data_offset(mmap, gguf.alignment))
}

fn compute_data_offset(data: &[u8], alignment: u64) -> usize {
    let mut pos: usize = 12;
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    pos += 8;
    for _ in 0..metadata_count {
        let key_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8 + key_len;
        let val_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        pos = skip_metadata_val(data, pos, val_type);
    }
    for _ in 0..tensor_count {
        let name_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8 + name_len;
        let dim_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4 + dim_count * 8 + 4 + 8;
    }
    let a = alignment.max(1) as usize;
    ((pos + a - 1) / a) * a
}

fn skip_metadata_val(data: &[u8], mut pos: usize, val_type: u32) -> usize {
    match val_type {
        0|1 => pos + 1, 2|3 => pos + 2, 4|5 => pos + 4, 6 => pos + 4, 7 => pos + 1,
        8 => { let len = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()) as usize; pos + 8 + len }
        9 => { let item = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); let len = u64::from_le_bytes(data[pos+4..pos+12].try_into().unwrap()) as usize; pos += 12; for _ in 0..len { pos = skip_metadata_val(data, pos, item); } pos }
        10|11 => pos + 8, 12 => pos + 8, _ => pos,
    }
}
