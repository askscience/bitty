use crate::device::{GpuBackend, WgpuDevice};
use crate::sampler;
use bitty_candle_runtime::Tokenizer;
use std::collections::HashMap;
use std::path::Path;
use wgpu::util::DeviceExt;

pub struct WgpuModel {
    device: WgpuDevice,
    metadata: CpuModelMetadata,
    tokenizer: Tokenizer,
    weights: GpuWeights,
    kv_cache: GpuKvCache,
}

#[derive(Debug, Clone)]
struct CpuModelMetadata {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_seq_len: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_style: RopeStyle,
    embedding_scale: Option<f32>,
    final_logit_softcap: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RopeStyle {
    Neox,
    Interleaved,
}

struct GpuWeights {
    embed_tokens: wgpu::Buffer,
    final_norm: wgpu::Buffer,
    layers: Vec<GpuLayerWeights>,
    lm_head: Option<wgpu::Buffer>,
}

struct GpuLayerWeights {
    input_ln: wgpu::Buffer,
    post_attn_ln: wgpu::Buffer,
    post_attention_norm: Option<wgpu::Buffer>,
    pre_ffn_norm: Option<wgpu::Buffer>,
    post_ffn_norm: Option<wgpu::Buffer>,
    q_proj: GpuQuantizedTensor,
    k_proj: GpuQuantizedTensor,
    v_proj: GpuQuantizedTensor,
    o_proj: GpuQuantizedTensor,
    up_proj: GpuQuantizedTensor,
    gate_proj: GpuQuantizedTensor,
    down_proj: GpuQuantizedTensor,
}

struct GpuQuantizedTensor {
    buffer: wgpu::Buffer,
    ggml_type: u32,
    /// Bytes per block in the native GGML format.
    block_size: u32,
    /// Elements per block.
    elements_per_block: u32,
    out_dim: u32,
    in_dim: u32,
}

struct GpuKvCache {
    keys: Vec<wgpu::Buffer>,
    values: Vec<wgpu::Buffer>,
    seq_len: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum WgpuModelError {
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("GGUF load error: {0}")]
    Load(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
}

impl WgpuModel {
    pub fn load(
        path: &Path,
        hf_model_id: Option<&str>,
        backend: GpuBackend,
    ) -> Result<Self, WgpuModelError> {
        let gpu = WgpuDevice::new(backend)
            .map_err(|e| WgpuModelError::Gpu(e))?;
        let tokenizer = Tokenizer::from_gguf_path(path, hf_model_id)
            .map_err(|e| WgpuModelError::Tokenizer(e.to_string()))?;

        let file = std::fs::File::open(path)
            .map_err(|e| WgpuModelError::Load(format!("Cannot open model: {e}")))?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| WgpuModelError::Load(format!("Cannot mmap: {e}")))?
        };

        let (meta, weights) = load_gguf_to_gpu(&gpu, &mmap)
            .map_err(|e| WgpuModelError::Load(e))?;

        let kv_cache = GpuKvCache::new(&gpu, meta.num_layers, meta.num_kv_heads, meta.head_dim, meta.max_seq_len);

        Ok(Self {
            device: gpu,
            metadata: meta,
            tokenizer,
            weights,
            kv_cache,
        })
    }

    pub fn generate_from_ids(
        &mut self,
        prompt_ids: &[u32],
        _max_tokens: usize,
        temperature: f32,
        top_k: usize,
        _on_delta: impl FnMut(&str),
    ) -> Result<String, WgpuModelError> {
        // Upload prompt
        let mut kv_cache = GpuKvCacheState { seq_len: 0 };

        // Process prompt tokens
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            let _hidden = self.forward_single_token(tid, pos, &mut kv_cache)?;
            if pos == prompt_ids.len() - 1 {
                // Last prompt token: sample from logits
                let _logits = self.compute_logits(&[])?;
                let _next = sampler::sample(&_logits, temperature, top_k);
                // TODO: full generation loop with streaming decode
            }
        }

        Ok(String::new())
    }

    fn forward_single_token(&self, _tid: u32, _pos: usize, _cache: &mut GpuKvCacheState) -> Result<Vec<f32>, WgpuModelError> {
        // TODO: implement with compute shader dispatches
        Ok(Vec::new())
    }

    fn compute_logits(&self, _hidden: &[f32]) -> Result<Vec<f32>, WgpuModelError> {
        // TODO: lm_head matmul + readback
        Ok(Vec::new())
    }

    fn is_stop(&self, token: u32) -> bool {
        token == self.tokenizer.eos_token_id()
            || self.tokenizer.eot_token_id() == Some(token)
            || self.tokenizer.im_end_token_id() == Some(token)
    }
}

struct GpuKvCacheState {
    seq_len: usize,
}

impl GpuKvCache {
    fn new(gpu: &WgpuDevice, num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        let kv_size = num_kv_heads * head_dim * max_seq;
        for _ in 0..num_layers {
            keys.push(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv_cache_keys"),
                size: (kv_size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            values.push(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv_cache_values"),
                size: (kv_size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        Self { keys, values, seq_len: 0 }
    }
}

fn load_gguf_to_gpu(
    gpu: &WgpuDevice,
    mmap: &[u8],
) -> Result<(CpuModelMetadata, GpuWeights), String> {
    use bitty_model::gguf::{parse_gguf_bytes, GgufFileMetadata};

    let gguf = parse_gguf_bytes(mmap).map_err(|e| format!("GGUF parse: {e}"))?;
    let meta = extract_metadata(&gguf)?;

    // Build tensor map
    let mut tensor_map: HashMap<String, (usize, usize)> = HashMap::new(); // offset, byte_len
    let data_offset = compute_data_offset(mmap, gguf.alignment);
    for t in &gguf.tensors {
        let offset = data_offset + t.offset as usize;
        tensor_map.insert(t.name.clone(), (offset, t.byte_len as usize));
    }

    let get_f32 = |name: &str, n: usize| -> Result<wgpu::Buffer, String> {
        let (off, len) = tensor_map
            .get(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        let data: &[f32] = bytemuck::cast_slice(&mmap[*off..*off + *len]);
        let buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(&data[..n]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        Ok(buf)
    };

    // Load embedding and final norm
    let embed_tokens = get_f32("token_embd.weight", meta.vocab_size * meta.hidden_size)?;
    let final_norm = get_f32("output_norm.weight", meta.hidden_size)?;

    // TODO: Load all layers' quantized tensors
    let layers = Vec::new();

    Ok((
        meta,
        GpuWeights {
            embed_tokens,
            final_norm,
            layers,
            lm_head: None,
        },
    ))
}

fn extract_metadata(gguf: &bitty_model::gguf::GgufFileMetadata) -> Result<CpuModelMetadata, String> {
    let m = &gguf.metadata;
    let arch = m.get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    let get_u32 = |key: &str| -> u32 {
        m.get(key).and_then(|v| v.as_u32()).unwrap_or(0)
    };

    let hidden_size = get_u32(&format!("{arch}.embedding_length")).max(2048) as usize;
    let num_layers = get_u32(&format!("{arch}.block_count")).max(1) as usize;
    let num_heads = get_u32(&format!("{arch}.attention.head_count")).max(1) as usize;
    let num_kv_heads = get_u32(&format!("{arch}.attention.head_count_kv")).max(1) as usize;
    let head_dim = if num_heads > 0 { hidden_size / num_heads } else { 64 };
    let rope_dim = get_u32(&format!("{arch}.rope.dimension_count")).max(2) as usize;
    let rope_dim = if rope_dim > 0 { rope_dim } else { head_dim };
    let intermediate_size = get_u32(&format!("{arch}.feed_forward_length")).max(1) as usize;
    let vocab_size = m.get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_string_array())
        .map(|a| a.len())
        .unwrap_or(128256);
    let max_seq_len = get_u32(&format!("{arch}.context_length")).max(2048) as usize;
    let rms_norm_eps = m.get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1e-5);
    let rope_theta = m.get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(10000.0);

    let rope_style = match arch {
        "llama" | "mistral" | "phi3" | "phi" | "tinyllama" | "smollm" | "stablelm" => {
            RopeStyle::Interleaved
        }
        _ => RopeStyle::Neox,
    };

    let embedding_scale = if arch.starts_with("gemma") {
        Some((hidden_size as f32).sqrt())
    } else {
        None
    };

    let final_logit_softcap = m.get(&format!("{arch}.final_logit_softcapping"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    Ok(CpuModelMetadata {
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_dim,
        intermediate_size,
        vocab_size,
        max_seq_len,
        rms_norm_eps,
        rope_theta,
        rope_style,
        embedding_scale,
        final_logit_softcap,
    })
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
        pos = skip_val(data, pos, val_type);
    }
    for _ in 0..tensor_count {
        let name_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8 + name_len;
        let dim_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4 + dim_count * 8 + 4 + 8;
    }
    let align = alignment.max(1) as usize;
    ((pos + align - 1) / align) * align
}

fn skip_val(data: &[u8], mut pos: usize, val_type: u32) -> usize {
    match val_type {
        0 | 1 => pos + 1,
        2 | 3 => pos + 2,
        4 | 5 => pos + 4,
        6 => pos + 4,
        7 => pos + 1,
        8 => {
            let len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
            pos + 8 + len
        }
        9 => {
            let item_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            let len = u64::from_le_bytes(data[pos + 4..pos + 12].try_into().unwrap()) as usize;
            pos += 12;
            for _ in 0..len {
                pos = skip_val(data, pos, item_type);
            }
            pos
        }
        10 | 11 => pos + 8,
        12 => pos + 8,
        _ => pos,
    }
}
