//! Architecture detection and metadata extraction from GGUF files.
//!
//! All values come from the GGUF header via architecture-specific keys
//! (e.g. `llama.embedding_length`, `qwen2.attention.head_count`). We intentionally
//! do NOT clamp values upward — earlier versions did `get_u32(key).max(2048)` as a
//! default, but `.max()` is a numeric floor and silently inflated smaller models'
//! dims so they stopped matching tensor shapes. Fall back with `unwrap_or` only.

use crate::cpu_backend::types::{ActivationFn, CpuModelMetadata};
use oxbitnet::model::gguf::GgufMetadata;

/// Detect model architecture from GGUF metadata.
pub fn detect_architecture(meta: &GgufMetadata) -> String {
    meta.get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string()
}

/// Extract model hyperparameters from GGUF metadata.
pub fn extract_config(meta: &GgufMetadata, num_layers: usize) -> CpuModelMetadata {
    let arch = detect_architecture(meta);

    // Lookup helper: try architecture-specific key, then llama fallback, then bitnet.
    let get_u32 = |suffix: &str| -> Option<u32> {
        meta.get(&format!("{arch}.{suffix}"))
            .and_then(|v| v.as_u32())
            .or_else(|| meta.get(&format!("llama.{suffix}")).and_then(|v| v.as_u32()))
            .or_else(|| meta.get(&format!("bitnet.{suffix}")).and_then(|v| v.as_u32()))
    };

    let hidden_size = get_u32("embedding_length").unwrap_or(2048) as usize;

    let num_heads = get_u32("attention.head_count").unwrap_or(32) as usize;

    // Grouped-query attention: if the file specifies kv-head count, use it as-is.
    // Only fall back to `num_heads` (no GQA) when the key is absent.
    let num_kv_heads = get_u32("attention.head_count_kv")
        .or_else(|| {
            // Older key spelling some converters emit.
            meta.get("llama.attention.key_value_head_count")
                .and_then(|v| v.as_u32())
        })
        .unwrap_or(num_heads as u32) as usize;

    // Head dim: prefer the explicit value if the architecture exposes it
    // (Qwen uses `attention.key_length`); otherwise derive it from hidden/heads.
    let head_dim = get_u32("attention.key_length")
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            if num_heads > 0 {
                hidden_size / num_heads
            } else {
                64
            }
        });

    let intermediate_size = get_u32("feed_forward_length").unwrap_or(8192) as usize;

    let vocab_size = meta
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_string_array())
        .map(|a| a.len())
        .unwrap_or(128256);

    let rms_eps = meta
        .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.as_f32())
        .unwrap_or(1e-5);

    let rope_theta = meta
        .get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.as_f32())
        .unwrap_or(if arch.contains("bitnet-25") { 500000.0 } else { 10000.0 });

    let max_seq_len = meta
        .get(&format!("{arch}.context_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(
            meta.get("llama.context_length")
                .and_then(|v| v.as_u32())
                .unwrap_or(4096),
        ) as usize;

    let activation = if arch.contains("bitnet-25") {
        ActivationFn::Relu2
    } else {
        ActivationFn::Silu
    };

    CpuModelMetadata {
        architecture: arch,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_seq_len,
        rms_norm_eps: rms_eps,
        rope_theta,
        activation,
    }
}
