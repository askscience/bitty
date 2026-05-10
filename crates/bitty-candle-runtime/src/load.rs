use bitty_model::gguf::{
    self, GgufFileMetadata, GgufTensorInfo, GGML_TYPE_F16,
};
use candle_core::{Device, Tensor};
use std::path::Path;

use crate::layers::{ModelConfig, RopeStyle};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("GGUF parse error: {0}")]
    Gguf(#[from] bitty_model::gguf::GgufError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Unsupported GGML type: {0}")]
    UnsupportedGgmlType(u32),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Model error: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, LoadError>;

pub struct LoadedModel {
    pub config: ModelConfig,
    pub weights: WeightStore,
}

pub struct WeightStore {
    mmap: memmap2::Mmap,
    pub tensors: Vec<GgufTensorInfo>,
    data_offset: u64,
    pub device: Device,
}

impl WeightStore {
    pub fn get_raw(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.iter().find(|t| t.name == name)?;
        let start = (self.data_offset + info.offset) as usize;
        let end = (start + info.byte_len as usize).min(self.mmap.len());
        Some(&self.mmap[start..end])
    }

    pub fn get_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    pub fn get_f32(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let raw = self.get_raw(name).ok_or_else(|| LoadError::MissingWeight(name.to_string()))?;
        let floats: &[f32] = bytemuck::cast_slice(raw);
        let data = floats.to_vec();
        Ok(Tensor::from_vec(data, shape, &self.device)?)
    }

    pub fn get_f16_to_f32(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let raw = self.get_raw(name).ok_or_else(|| LoadError::MissingWeight(name.to_string()))?;
        let halfs: &[half::f16] = bytemuck::cast_slice(raw);
        let data: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
        Ok(Tensor::from_vec(data, shape, &self.device)?)
    }
}

pub fn load_gguf(source: &str, device: &Device) -> Result<LoadedModel> {
    let path = Path::new(source);
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gguf = gguf::parse_gguf_bytes(&mmap)?;

    let config = extract_config(&gguf)?;
    let data_offset = compute_data_offset(&mmap, gguf.alignment);

    let weights = WeightStore {
        mmap,
        tensors: gguf.tensors.clone(),
        data_offset,
        device: device.clone(),
    };

    Ok(LoadedModel { config, weights })
}

fn extract_config(gguf: &GgufFileMetadata) -> Result<ModelConfig> {
    let m = &gguf.metadata;
    let get_f64 = |keys: &[&str]| -> Option<f64> {
        keys.iter().find_map(|k| match m.get(*k)? {
            bitty_model::gguf::GgufMetadataValue::F64(v) => Some(*v),
            _ => None,
        })
    };

    let hidden_size = m.get("bitnet.embedding_length")
        .or_else(|| m.get("llama.embedding_length"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let intermediate_size = m.get("bitnet.feed_forward_length")
        .or_else(|| m.get("llama.feed_forward_length"))
        .and_then(|v| v.as_u64())
        .unwrap_or((hidden_size as u64 * 8 / 3).max(1)) as usize;

    let num_hidden_layers = m.get("bitnet.block_count")
        .or_else(|| m.get("llama.block_count"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let num_attention_heads = m.get("bitnet.attention.head_count")
        .or_else(|| m.get("llama.attention.head_count"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    let num_key_value_heads = m.get("bitnet.attention.head_count_kv")
        .or_else(|| m.get("llama.attention.head_count_kv"))
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads as u64) as usize;

    let vocab_size = m.get("llama.vocab_size")
        .or_else(|| m.get("tokenizer.ggml.tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(32_000) as usize;

    let max_position_embeddings = m.get("bitnet.context_length")
        .or_else(|| m.get("llama.context_length"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;

    let rms_norm_eps = get_f64(&["bitnet.attention.layer_norm_epsilon", "llama.attention.layer_norm_epsilon"])
        .unwrap_or(1e-5) as f32;

    let rope_theta = get_f64(&["bitnet.rope.theta", "llama.rope.theta"])
        .unwrap_or(10000.0) as f32;

    let arch_str = m.get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let is_bitnet = arch_str.contains("bitnet");
    let is_gemma3 = arch_str.starts_with("gemma3") || arch_str.starts_with("gemma-3");

    let rope_style = match arch_str {
        "llama" | "mistral" | "phi3" | "phi" | "tinyllama" | "smollm" | "stablelm" => {
            RopeStyle::Interleaved
        }
        _ => RopeStyle::Neox,
    };

    let embedding_scale = if arch_str.starts_with("gemma") {
        Some((hidden_size as f32).sqrt())
    } else {
        None
    };

    let final_logit_softcap = m
        .get(&format!("{arch_str}.final_logit_softcapping"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    let tie_word_embeddings = is_bitnet
        || gguf.tensors.iter().all(|t| t.name != "lm_head.weight");

    let lm_head_f16 = gguf.tensors.iter().any(|t| t.name == "lm_head.weight"
        && t.ggml_type == GGML_TYPE_F16);

    let is_qwen = arch_str.contains("qwen2") || arch_str.contains("qwen3");

    Ok(ModelConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        rope_style,
        tie_word_embeddings,
        lm_head_f16,
        is_qwen,
        embedding_scale,
        final_logit_softcap,
        is_gemma3,
    })
}

fn compute_data_offset(data: &[u8], alignment: u64) -> u64 {
    let mut pos: u64 = 0;
    if data.len() < 4 { return 0; }
    pos += 4;
    if data.len() < 8 { return 0; }
    pos += 4;
    if data.len() < 24 { return 0; }
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    pos += 16;
    for _ in 0..metadata_count {
        if (pos as usize) + 8 > data.len() { return 0; }
        let key_len = u64::from_le_bytes(data[pos as usize..(pos as usize + 8)].try_into().unwrap());
        pos += 8 + key_len;
        if (pos as usize) + 4 > data.len() { return 0; }
        let val_type = u32::from_le_bytes(data[pos as usize..(pos as usize + 4)].try_into().unwrap());
        pos += 4;
        pos = skip_metadata_value(data, pos, val_type);
    }
    for _ in 0..tensor_count {
        if (pos as usize) + 8 > data.len() { return 0; }
        let name_len = u64::from_le_bytes(data[pos as usize..(pos as usize + 8)].try_into().unwrap());
        pos += 8 + name_len;
        if (pos as usize) + 4 > data.len() { return 0; }
        let dim_count = u32::from_le_bytes(data[pos as usize..(pos as usize + 4)].try_into().unwrap());
        pos += 4;
        pos += dim_count as u64 * 8;
        pos += 4;
        pos += 8;
    }
    let alignment = alignment.max(1);
    ((pos + alignment - 1) / alignment) * alignment
}

fn skip_metadata_value(data: &[u8], mut pos: u64, val_type: u32) -> u64 {
    match val_type {
        0 | 1 => pos + 1,
        2 | 3 => pos + 2,
        4 | 5 => pos + 4,
        6 => pos + 4,
        7 => pos + 1,
        8 => {
            let len = u64::from_le_bytes(data[pos as usize..(pos as usize + 8)].try_into().unwrap());
            pos + 8 + len
        }
        9 => {
            let item_type = u32::from_le_bytes(data[pos as usize..(pos as usize + 4)].try_into().unwrap());
            let len = u64::from_le_bytes(data[(pos as usize + 4)..(pos as usize + 12)].try_into().unwrap());
            pos += 12;
            for _ in 0..len {
                pos = skip_metadata_value(data, pos, item_type);
            }
            pos
        }
        10 | 11 => pos + 8,
        12 => pos + 8,
        _ => pos,
    }
}
