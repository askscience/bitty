//! Tensor name classification for GGUF models.
//! Maps raw GGUF tensor names to logical roles used by the CPU backend.
//!
//! Handles all naming conventions: `blk.N.component`, `model.layers.N.component`,
//! and architecture-specific variants (SSM, fused QKV, QK norms, etc.).

#[derive(Debug, Clone)]
pub enum TensorRole {
    EmbedTokens,
    FinalNorm,
    LmHead,
    InputNorm(usize),
    PostAttnNorm(usize),
    /// Gemma3: norm after attention, before residual.
    PostAttentionNorm(usize),
    /// Gemma3: norm before FFN (after first residual).
    PreFfnNorm(usize),
    /// Gemma3: norm after FFN, before second residual.
    PostFfnNorm(usize),
    QProj(usize),
    KProj(usize),
    VProj(usize),
    OProj(usize),
    QkvFused(usize),
    QNorm(usize),
    KNorm(usize),
    UpProj(usize),
    GateProj(usize),
    DownProj(usize),
    // SSM/Mamba tensors
    SsmInProj(usize),
    SsmConv1dWeight(usize),
    SsmConv1dBias(usize),
    SsmDtProjWeight(usize),
    SsmDtProjBias(usize),
    SsmA(usize),
    SsmAlphaWeight(usize),
    SsmBetaWeight(usize),
    SsmDParam(usize),
    SsmOutProj(usize),
    SsmNorm(usize),
    /// Qwen3.5 gated attention gate weight
    AttnGate(usize),
    Ignored,
}

/// Classify a raw GGUF tensor name into its logical role.
/// Returns `None` if the name doesn't match any known pattern.
pub fn classify(name: &str) -> Option<TensorRole> {
    if name == "token_embd.weight" {
        return Some(TensorRole::EmbedTokens);
    }
    if name == "output_norm.weight" || name == "model.norm.weight" {
        return Some(TensorRole::FinalNorm);
    }
    if name == "output.weight" || name == "lm_head.weight" {
        return Some(TensorRole::LmHead);
    }

    // Layer-specific tensors
    if let Some((layer, comp)) = parse_layer_component(name) {
        return match comp {
            "attn_q.weight" | "self_attn.q_proj.weight" => Some(TensorRole::QProj(layer)),
            "attn_k.weight" | "self_attn.k_proj.weight" => Some(TensorRole::KProj(layer)),
            "attn_v.weight" | "self_attn.v_proj.weight" => Some(TensorRole::VProj(layer)),
            "attn_output.weight" | "self_attn.o_proj.weight" => Some(TensorRole::OProj(layer)),
            "attn_qkv.weight" => Some(TensorRole::QkvFused(layer)),
            "attn_norm.weight" | "input_layernorm.weight" => Some(TensorRole::InputNorm(layer)),
            "ffn_norm.weight" | "post_attention_norm.weight" => {
                Some(TensorRole::PostAttnNorm(layer))
            }
            "post_attention_layernorm.weight" => Some(TensorRole::PostAttentionNorm(layer)),
            "pre_feedforward_layernorm.weight" => Some(TensorRole::PreFfnNorm(layer)),
            "post_feedforward_layernorm.weight" => Some(TensorRole::PostFfnNorm(layer)),
            "attn_q_norm.weight" => Some(TensorRole::QNorm(layer)),
            "attn_k_norm.weight" => Some(TensorRole::KNorm(layer)),
            "ffn_up.weight" | "mlp.up_proj.weight" => Some(TensorRole::UpProj(layer)),
            "ffn_gate.weight" | "mlp.gate_proj.weight" => Some(TensorRole::GateProj(layer)),
            "ffn_down.weight" | "mlp.down_proj.weight" => Some(TensorRole::DownProj(layer)),
            // SSM tensors
            "ssm_in_proj.weight" => Some(TensorRole::SsmInProj(layer)),
            "ssm_conv1d.weight" => Some(TensorRole::SsmConv1dWeight(layer)),
            "ssm_conv1d.bias" => Some(TensorRole::SsmConv1dBias(layer)),
            "ssm_dt_proj.weight" => Some(TensorRole::SsmDtProjWeight(layer)),
            "ssm_dt_proj.bias" | "ssm_dt.bias" => Some(TensorRole::SsmDtProjBias(layer)),
            "ssm_a" => Some(TensorRole::SsmA(layer)),
            "ssm_alpha.weight" => Some(TensorRole::SsmAlphaWeight(layer)),
            "ssm_beta.weight" => Some(TensorRole::SsmBetaWeight(layer)),
            "ssm_d.weight" | "ssm_d_param" => Some(TensorRole::SsmDParam(layer)),
            "ssm_out.weight" => Some(TensorRole::SsmOutProj(layer)),
            "ssm_norm.weight" => Some(TensorRole::SsmNorm(layer)),
            "attn_gate.weight" => Some(TensorRole::AttnGate(layer)),
            // Sub-norms and extras — ignore
            "attn_sub_norm.weight"
            | "ffn_sub_norm.weight"
            | "self_attn.sub_norm.weight"
            | "mlp.sub_norm.weight" => Some(TensorRole::Ignored),
            _ => None,
        };
    }
    None
}

/// Parse a tensor name like `blk.3.attn_q.weight` into (3, "attn_q.weight").
/// Also handles `model.layers.3.self_attn.q_proj.weight` and `layers.3.attn.output.weight`.
fn parse_layer_component(name: &str) -> Option<(usize, &str)> {
    // blk.N.component
    if let Some(rest) = name.strip_prefix("blk.") {
        let dot = rest.find('.')?;
        let layer: usize = rest[..dot].parse().ok()?;
        return Some((layer, &rest[dot + 1..]));
    }
    // model.layers.N.component
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let dot = rest.find('.')?;
        let layer: usize = rest[..dot].parse().ok()?;
        return Some((layer, &rest[dot + 1..]));
    }
    // layers.N.component
    if let Some(rest) = name.strip_prefix("layers.") {
        let dot = rest.find('.')?;
        let layer: usize = rest[..dot].parse().ok()?;
        return Some((layer, &rest[dot + 1..]));
    }
    None
}
