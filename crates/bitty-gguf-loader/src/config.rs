use bitty_model::gguf::{GgufFileMetadata, GgufTensorInfo, GGML_TYPE_F16};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeStyle {
    Neox,
    Interleaved,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_style: RopeStyle,
    pub tie_word_embeddings: bool,
    pub lm_head_f16: bool,
    pub is_qwen: bool,
    pub embedding_scale: Option<f32>,
    pub final_logit_softcap: Option<f32>,
    pub is_gemma3: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Extract model configuration from GGUF metadata.
/// This is the canonical implementation used by all backends.
pub fn extract_model_config(gguf: &GgufFileMetadata, tensors: &[GgufTensorInfo]) -> ModelConfig {
    let m = &gguf.metadata;
    let arch_str = m.get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let is_bitnet = arch_str.contains("bitnet");
    let is_gemma3 = arch_str.starts_with("gemma3") || arch_str.starts_with("gemma-3");

    let get = |suffix: &str| -> Option<u64> {
        m.get(&format!("{arch_str}.{suffix}"))
            .or_else(|| m.get(&format!("llama.{suffix}")))
            .or_else(|| m.get(&format!("bitnet.{suffix}")))
            .and_then(|v| v.as_u64())
    };

    let hidden_size = get("embedding_length").unwrap_or(0) as usize;
    let intermediate_size = get("feed_forward_length")
        .unwrap_or((hidden_size as u64 * 8 / 3).max(1)) as usize;
    let num_hidden_layers = get("block_count").unwrap_or(0) as usize;
    let num_attention_heads = get("attention.head_count").unwrap_or(1) as usize;
    let num_key_value_heads = get("attention.head_count_kv")
        .unwrap_or(num_attention_heads as u64) as usize;
    let vocab_size = get("vocab_size")
        .or_else(|| m.get("tokenizer.ggml.tokens").and_then(|v| v.as_u64()))
        .unwrap_or(32_000) as usize;
    let max_position_embeddings = get("context_length").unwrap_or(2048) as usize;

    let rms_norm_eps = m.get(&format!("{arch_str}.attention.layer_norm_rms_epsilon"))
        .or_else(|| m.get(&format!("{arch_str}.attention.layer_norm_epsilon")))
        .or_else(|| m.get("llama.attention.layer_norm_rms_epsilon"))
        .or_else(|| m.get("llama.attention.layer_norm_epsilon"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;

    let rope_theta = m.get(&format!("{arch_str}.rope.freq_base"))
        .or_else(|| m.get(&format!("{arch_str}.rope.theta")))
        .or_else(|| m.get("llama.rope.freq_base"))
        .or_else(|| m.get("llama.rope.theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

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
        || tensors.iter().all(|t| t.name != "lm_head.weight");

    let lm_head_f16 = tensors.iter().any(|t| {
        t.name == "lm_head.weight" && t.ggml_type == GGML_TYPE_F16
    });

    let is_qwen = arch_str.contains("qwen2") || arch_str.contains("qwen3");

    ModelConfig {
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
    }
}
