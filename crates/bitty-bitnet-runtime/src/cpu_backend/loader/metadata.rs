//! Architecture detection and metadata extraction from GGUF files.
//!
//! All values come from the GGUF header via architecture-specific keys
//! (e.g. `llama.embedding_length`, `qwen2.attention.head_count`). We intentionally
//! do NOT clamp values upward — earlier versions did `get_u32(key).max(2048)` as a
//! default, but `.max()` is a numeric floor and silently inflated smaller models'
//! dims so they stopped matching tensor shapes. Fall back with `unwrap_or` only.

use crate::cpu_backend::types::{ActivationFn, CpuModelMetadata};
use bitty_model::gguf::GgufFileMetadata;

/// Detect model architecture from GGUF metadata.
pub fn detect_architecture(meta: &GgufFileMetadata) -> String {
    meta.metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string()
}

/// Extract model hyperparameters from GGUF metadata.
pub fn extract_config(meta: &GgufFileMetadata, num_layers: usize) -> CpuModelMetadata {
    let arch = detect_architecture(meta);

    // Lookup helper: try architecture-specific key, then llama fallback, then bitnet.
    let get_u32 = |suffix: &str| -> Option<u32> {
        meta.metadata
            .get(&format!("{arch}.{suffix}"))
            .and_then(|v| v.as_u32())
            .or_else(|| {
                meta.metadata
                    .get(&format!("llama.{suffix}"))
                    .and_then(|v| v.as_u32())
            })
            .or_else(|| {
                meta.metadata
                    .get(&format!("bitnet.{suffix}"))
                    .and_then(|v| v.as_u32())
            })
    };

    let hidden_size = get_u32("embedding_length").unwrap_or(2048) as usize;

    let num_heads = get_u32("attention.head_count").unwrap_or(32) as usize;

    // Grouped-query attention: if the file specifies kv-head count, use it as-is.
    // Only fall back to `num_heads` (no GQA) when the key is absent.
    let num_kv_heads = get_u32("attention.head_count_kv")
        .or_else(|| {
            // Older key spelling some converters emit.
            meta.metadata
                .get("llama.attention.key_value_head_count")
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
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_string_array())
        .map(|a| a.len())
        .unwrap_or(128256);

    let rms_eps = meta
        .metadata
        .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1e-5);

    let rope_theta = meta
        .metadata
        .get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(if arch.contains("bitnet-25") {
            500000.0
        } else {
            10000.0
        });

    let max_seq_len = meta
        .metadata
        .get(&format!("{arch}.context_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(
            meta.metadata
                .get("llama.context_length")
                .and_then(|v| v.as_u32())
                .unwrap_or(4096),
        ) as usize;

    let activation = if arch.contains("bitnet-25") {
        ActivationFn::Relu2
    } else {
        ActivationFn::Silu
    };

    let is_qwen35 = arch == "qwen35"
        || meta
            .metadata
            .keys()
            .any(|k| k.starts_with("qwen35.") || k.starts_with("Qwen35."));

    let rope_dim = get_u32("rope.dimension_count")
        .map(|v| v as usize)
        .unwrap_or(head_dim);

    let default_mrope: [u32; 4] = [11, 11, 10, 0];
    let rope_sections = default_mrope;

    let full_attention_interval = get_u32("attention.full_attention_interval").unwrap_or(4);

    let ssm_d_conv = get_u32("ssm.conv_kernel").unwrap_or(0) as usize;
    let ssm_d_inner = get_u32("ssm.inner_size").unwrap_or(0) as usize;
    let ssm_d_state = get_u32("ssm.state_size").unwrap_or(0) as usize;
    let ssm_dt_rank = get_u32("ssm.time_step_rank").unwrap_or(0) as usize;
    let ssm_n_group = get_u32("ssm.group_count").unwrap_or(0) as usize;

    CpuModelMetadata {
        architecture: arch,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_dim,
        intermediate_size,
        vocab_size,
        max_seq_len,
        rms_norm_eps: rms_eps,
        rope_theta,
        activation,
        is_qwen35,
        full_attention_interval,
        rope_sections,
        ssm_d_conv,
        ssm_d_inner,
        ssm_d_state,
        ssm_dt_rank,
        ssm_n_group,
    }
}
