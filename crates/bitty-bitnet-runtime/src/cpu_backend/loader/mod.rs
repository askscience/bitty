//! GGUF model loading for the CPU inference backend.
//!
//! Reads GGUF files, classifies tensors by their role, and builds
//! the CpuWeights structure used by all layers.

pub mod metadata;
pub mod names;

use crate::cpu_backend::dequant::dequantize_slice;
use crate::cpu_backend::types::{
    AttentionWeights, CpuLayer, CpuWeights, LayerKind, LinearAttnWeights, LmHead, MlpBlock,
    PackedTensor, SsmWeights,
};
use names::TensorRole;
use bytes::Bytes;
use bitty_model::gguf::{
    parse_gguf_bytes, GgufFileMetadata, GGML_TYPE_I2_S, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
};
use std::collections::HashMap;

pub fn ggml_type_size(ty: u32) -> Result<f64, String> {
    match ty {
        bitty_model::gguf::GGML_TYPE_F32 => Ok(4.0),
        bitty_model::gguf::GGML_TYPE_F16 | bitty_model::gguf::GGML_TYPE_BF16 => Ok(2.0),
        bitty_model::gguf::GGML_TYPE_Q4_0 | bitty_model::gguf::GGML_TYPE_Q4_1 | bitty_model::gguf::GGML_TYPE_Q4_K => Ok(0.5),
        bitty_model::gguf::GGML_TYPE_Q4_0_4_4 | bitty_model::gguf::GGML_TYPE_Q4_0_4_8 | bitty_model::gguf::GGML_TYPE_Q4_0_8_8 => Ok(0.5),
        bitty_model::gguf::GGML_TYPE_IQ4_NL | bitty_model::gguf::GGML_TYPE_IQ4_XS => Ok(0.5),
        bitty_model::gguf::GGML_TYPE_Q5_0 | bitty_model::gguf::GGML_TYPE_Q5_1 | bitty_model::gguf::GGML_TYPE_Q5_K => Ok(0.625),
        bitty_model::gguf::GGML_TYPE_Q8_0 | bitty_model::gguf::GGML_TYPE_Q8_1 | bitty_model::gguf::GGML_TYPE_Q8_K => Ok(1.0),
        bitty_model::gguf::GGML_TYPE_Q2_K | bitty_model::gguf::GGML_TYPE_IQ2_XXS | bitty_model::gguf::GGML_TYPE_IQ2_XS | bitty_model::gguf::GGML_TYPE_IQ2_S => Ok(0.25),
        bitty_model::gguf::GGML_TYPE_Q3_K | bitty_model::gguf::GGML_TYPE_IQ3_XXS | bitty_model::gguf::GGML_TYPE_IQ3_S | bitty_model::gguf::GGML_TYPE_IQ3_M => Ok(0.375),
        bitty_model::gguf::GGML_TYPE_Q6_K => Ok(0.75),
        bitty_model::gguf::GGML_TYPE_IQ1_S | bitty_model::gguf::GGML_TYPE_IQ1_M => Ok(0.125),
        bitty_model::gguf::GGML_TYPE_I8 => Ok(1.0),
        bitty_model::gguf::GGML_TYPE_I16 => Ok(2.0),
        bitty_model::gguf::GGML_TYPE_I32 => Ok(4.0),
        bitty_model::gguf::GGML_TYPE_I64 | bitty_model::gguf::GGML_TYPE_F64 => Ok(8.0),
        bitty_model::gguf::GGML_TYPE_TQ1_0 => Ok(54.0 / 256.0),
        bitty_model::gguf::GGML_TYPE_I2_S => Ok(0.25),
        bitty_model::gguf::GGML_TYPE_TQ2_0 => Ok(66.0 / 256.0),
        _ => Err(format!("Unsupported GGML type: {ty}")),
    }
}

fn compute_tensor_data_offset(data: &[u8], alignment: u64) -> usize {
    let mut pos: usize = 0;
    if data.len() < 8 { return 0; }
    pos += 4;
    let _version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    pos += 4;
    if data.len() < 24 { return 0; }
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    pos += 16;

    for _ in 0..metadata_count {
        if pos + 8 > data.len() { return 0; }
        let key_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8 + key_len;
        if pos + 4 > data.len() { return 0; }
        let val_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        pos = skip_metadata_val(data, pos, val_type);
    }

    for _ in 0..tensor_count {
        if pos + 8 > data.len() { return 0; }
        let name_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8 + name_len;
        if pos + 4 > data.len() { return 0; }
        let dim_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        pos += dim_count * 8;
        pos += 4;
        pos += 8;
    }

    let alignment = alignment.max(1) as usize;
    ((pos + alignment - 1) / alignment) * alignment
}

fn skip_metadata_val(data: &[u8], mut pos: usize, val_type: u32) -> usize {
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
                pos = skip_metadata_val(data, pos, item_type);
            }
            pos
        }
        10 | 11 => pos + 8,
        12 => pos + 8,
        _ => pos,
    }
}

/// Load a GGUF file and return metadata and weights. Tokenizer must be loaded separately.
pub fn load_gguf(data: &[u8]) -> Result<(GgufFileMetadata, CpuWeights), String> {
    let gguf = parse_gguf_bytes(data)
        .map_err(|e| format!("GGUF parse error: {e}"))?;

    let metadata = gguf.clone();
    let tensor_data_offset = compute_tensor_data_offset(data, gguf.alignment);

    // Collect tensors by layer
    let mut embed_tokens = Vec::new();
    let mut final_norm = Vec::new();
    let mut lm_head: Option<LmHead> = None;
    let mut layer_builders: HashMap<usize, LayerBuilder> = HashMap::new();

    for tensor in &gguf.tensors {
        let data_offset = tensor_data_offset + tensor.offset as usize;
        let num_elements: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
        let (byte_size, _) = packed_byte_size(tensor.ggml_type, num_elements);
        let tensor_data = &data[data_offset..(data_offset + byte_size).min(data.len())];

        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let packed = PackedTensor {
            data: Bytes::copy_from_slice(tensor_data),
            ggml_type: tensor.ggml_type,
            shape,
            name: tensor.name.clone(),
        };

        if let Some(role) = names::classify(&tensor.name) {
            match role {
                TensorRole::EmbedTokens => {
                    embed_tokens = dequant_to_f32(tensor_data, tensor.ggml_type, num_elements);
                }
                TensorRole::FinalNorm => {
                    final_norm = dequant_to_f32(tensor_data, tensor.ggml_type, num_elements);
                }
                TensorRole::LmHead => {
                    lm_head = Some(LmHead::Packed(packed));
                }
                TensorRole::InputNorm(l) => {
                    layer_builders.entry(l).or_default().input_ln =
                        dequant_to_f32(tensor_data, tensor.ggml_type, num_elements);
                }
                TensorRole::PostAttnNorm(l) => {
                    layer_builders.entry(l).or_default().post_attn_ln =
                        dequant_to_f32(tensor_data, tensor.ggml_type, num_elements);
                }
                TensorRole::QProj(l) => {
                    layer_builders.entry(l).or_default().q_proj = Some(packed);
                }
                TensorRole::KProj(l) => {
                    layer_builders.entry(l).or_default().k_proj = Some(packed);
                }
                TensorRole::VProj(l) => {
                    layer_builders.entry(l).or_default().v_proj = Some(packed);
                }
                TensorRole::QkvFused(l) => {
                    let b = layer_builders.entry(l).or_default();
                    b.q_proj = Some(packed.clone());
                    b.k_proj = Some(packed.clone());
                    b.v_proj = Some(packed);
                    b.is_fused_qkv = true;
                    b.has_attention = true;
                }
                TensorRole::OProj(l) => {
                    layer_builders.entry(l).or_default().o_proj = Some(packed);
                    layer_builders.entry(l).or_default().has_attention = true;
                }
                TensorRole::QNorm(l) => {
                    layer_builders.entry(l).or_default().q_norm = Some(dequant_to_f32(
                        tensor_data,
                        tensor.ggml_type,
                        num_elements,
                    ));
                }
                TensorRole::KNorm(l) => {
                    layer_builders.entry(l).or_default().k_norm = Some(dequant_to_f32(
                        tensor_data,
                        tensor.ggml_type,
                        num_elements,
                    ));
                }
                TensorRole::AttnGate(l) => {
                    layer_builders.entry(l).or_default().attn_gate = Some(packed.clone());
                }
                TensorRole::UpProj(l) => {
                    layer_builders.entry(l).or_default().up_proj = Some(packed);
                }
                TensorRole::GateProj(l) => {
                    layer_builders.entry(l).or_default().gate_proj = Some(packed);
                }
                TensorRole::DownProj(l) => {
                    layer_builders.entry(l).or_default().down_proj = Some(packed);
                }
                TensorRole::SsmInProj(l) => {
                    layer_builders.entry(l).or_default().ssm_in_proj = Some(packed);
                    layer_builders.entry(l).or_default().has_ssm = true;
                }
                TensorRole::SsmConv1dWeight(l) => {
                    layer_builders.entry(l).or_default().ssm_conv1d_weight = Some(packed);
                    layer_builders.entry(l).or_default().has_ssm = true;
                }
                TensorRole::SsmConv1dBias(l) => {
                    layer_builders.entry(l).or_default().ssm_conv1d_bias = Some(dequant_to_f32(
                        tensor_data,
                        tensor.ggml_type,
                        num_elements,
                    ));
                }
                TensorRole::SsmDtProjWeight(l) => {
                    layer_builders.entry(l).or_default().ssm_dt_proj_weight = Some(packed);
                    layer_builders.entry(l).or_default().has_ssm = true;
                }
                TensorRole::SsmDtProjBias(l) => {
                    layer_builders.entry(l).or_default().ssm_dt_proj_bias = Some(dequant_to_f32(
                        tensor_data,
                        tensor.ggml_type,
                        num_elements,
                    ));
                }
                TensorRole::SsmA(l) => {
                    layer_builders.entry(l).or_default().ssm_a_log = Some(dequant_to_f32(
                        tensor_data,
                        tensor.ggml_type,
                        num_elements,
                    ));
                }
                TensorRole::SsmAlphaWeight(l) => {
                    let b = layer_builders.entry(l).or_default();
                    b.ssm_alpha = Some(packed);
                    b.has_ssm = true;
                }
                TensorRole::SsmBetaWeight(l) => {
                    let b = layer_builders.entry(l).or_default();
                    b.ssm_beta = Some(packed);
                    b.has_ssm = true;
                }
                TensorRole::SsmDParam(l) => {
                    layer_builders.entry(l).or_default().ssm_d_param = Some(dequant_to_f32(
                        tensor_data,
                        tensor.ggml_type,
                        num_elements,
                    ));
                }
                TensorRole::SsmOutProj(l) => {
                    layer_builders.entry(l).or_default().ssm_out_proj = Some(packed);
                    layer_builders.entry(l).or_default().has_ssm = true;
                }
                TensorRole::SsmNorm(l) => {
                    layer_builders.entry(l).or_default().ssm_norm = Some(dequant_to_f32(
                        tensor_data,
                        tensor.ggml_type,
                        num_elements,
                    ));
                }
                TensorRole::Ignored => {}
            }
        }
    }

    // Assemble layers
    let mut indices: Vec<usize> = layer_builders.keys().copied().collect();
    indices.sort();
    let num_cfg_layers = indices.iter().max().map(|x| x + 1).unwrap_or(1);
    let meta_cfg = metadata::extract_config(&metadata, num_cfg_layers);

    let mut layers = Vec::new();
    let mut ssm_count = 0;
    let mut linear_attn_count = 0;

    for i in indices {
        let b = layer_builders.remove(&i).unwrap();
        let builder = b;

        let mlp = MlpBlock {
            up_proj: builder
                .up_proj
                .as_ref()
                .cloned()
                .unwrap_or_else(PackedTensor::dummy),
            gate_proj: builder
                .gate_proj
                .as_ref()
                .cloned()
                .unwrap_or_else(PackedTensor::dummy),
            down_proj: builder
                .down_proj
                .as_ref()
                .cloned()
                .unwrap_or_else(PackedTensor::dummy),
        };

        let input_ln = if builder.input_ln.is_empty() {
            vec![1.0f32; 1]
        } else {
            builder.input_ln.clone()
        };
        let post_attn_ln = if builder.post_attn_ln.is_empty() {
            vec![1.0f32; 1]
        } else {
            builder.post_attn_ln.clone()
        };

        let kind = if meta_cfg.is_qwen35 {
            let recurrent = (i + 1) % (meta_cfg.full_attention_interval as usize) != 0;
            if recurrent {
                linear_attn_count += 1;
                eprintln!("info: layer {i} uses Qwen linear attention");
                LayerKind::LinearAttn(build_linear_attn(builder)?)
            } else {
                LayerKind::Attention(AttentionWeights {
                    q_proj: builder.q_proj.unwrap_or_else(PackedTensor::dummy),
                    k_proj: builder.k_proj.unwrap_or_else(PackedTensor::dummy),
                    v_proj: builder.v_proj.unwrap_or_else(PackedTensor::dummy),
                    o_proj: builder.o_proj.unwrap_or_else(PackedTensor::dummy),
                    q_norm: builder.q_norm,
                    k_norm: builder.k_norm,
                    attn_gate: builder.attn_gate,
                    is_fused_qkv: builder.is_fused_qkv,
                })
            }
        } else if builder.has_ssm {
            ssm_count += 1;
            eprintln!("info: layer {i} uses SSM");
            let proj = builder
                .ssm_in_proj
                .or(builder.q_proj.clone())
                .unwrap_or_else(PackedTensor::dummy);
            let d_inner = proj.shape.last().copied().unwrap_or(1) / 2;
            let d_state = builder
                .ssm_a_log
                .as_ref()
                .map(|a| a.len() / d_inner.max(1))
                .unwrap_or(16);
            let kernel_size = builder
                .ssm_conv1d_weight
                .as_ref()
                .map(|w| w.shape.last().copied().unwrap_or(4))
                .unwrap_or(4);

            LayerKind::Ssm(SsmWeights {
                in_proj: proj,
                conv1d_weight: builder
                    .ssm_conv1d_weight
                    .unwrap_or_else(PackedTensor::dummy),
                conv1d_bias: builder.ssm_conv1d_bias,
                dt_proj_weight: builder
                    .ssm_dt_proj_weight
                    .unwrap_or_else(PackedTensor::dummy),
                dt_proj_bias: builder.ssm_dt_proj_bias,
                a_log: builder
                    .ssm_a_log
                    .unwrap_or_else(|| vec![0.0f32; d_inner * d_state]),
                d_param: builder
                    .ssm_d_param
                    .unwrap_or_else(|| vec![0.0f32; d_inner * d_state]),
                out_proj: builder.ssm_out_proj.unwrap_or_else(PackedTensor::dummy),
                ssm_norm: builder.ssm_norm.unwrap_or_else(|| vec![1.0f32; d_inner]),
                d_state,
                d_inner,
                kernel_size: kernel_size as usize,
            })
        } else if builder.has_attention {
            LayerKind::Attention(AttentionWeights {
                q_proj: builder.q_proj.unwrap_or_else(PackedTensor::dummy),
                k_proj: builder.k_proj.unwrap_or_else(PackedTensor::dummy),
                v_proj: builder.v_proj.unwrap_or_else(PackedTensor::dummy),
                o_proj: builder.o_proj.unwrap_or_else(PackedTensor::dummy),
                q_norm: builder.q_norm,
                k_norm: builder.k_norm,
                attn_gate: builder.attn_gate,
                is_fused_qkv: builder.is_fused_qkv,
            })
        } else {
            eprintln!("warning: layer {i} has neither attention nor SSM weights");
            continue;
        };

        layers.push(CpuLayer {
            kind,
            input_ln,
            post_attn_ln,
            mlp,
            layer_idx: i,
        });
    }

    if ssm_count > 0 {
        eprintln!("info: {} Mamba SSM layers loaded", ssm_count);
    }
    if linear_attn_count > 0 {
        eprintln!("info: {} Qwen linear attention layers loaded", linear_attn_count);
    }

    let weights = CpuWeights {
        embed_tokens,
        final_norm,
        layers,
        lm_head,
    };
    Ok((metadata, weights))
}

fn build_linear_attn(b: LayerBuilder) -> Result<LinearAttnWeights, String> {
    let wqkv = b
        .q_proj
        .ok_or_else(|| "Qwen linear layer: missing attn_qkv".to_string())?;
    let wqkv_gate = b
        .attn_gate
        .ok_or_else(|| "Qwen linear layer: missing attn_gate (z)".to_string())?;
    let ssm_conv1d = b
        .ssm_conv1d_weight
        .ok_or_else(|| "Qwen linear layer: missing ssm_conv1d".to_string())?;
    let ssm_alpha = b
        .ssm_alpha
        .ok_or_else(|| "Qwen linear layer: missing ssm_alpha".to_string())?;
    let ssm_beta = b
        .ssm_beta
        .ok_or_else(|| "Qwen linear layer: missing ssm_beta".to_string())?;
    let ssm_out = b
        .ssm_out_proj
        .ok_or_else(|| "Qwen linear layer: missing ssm_out".to_string())?;
    let ssm_a = b
        .ssm_a_log
        .ok_or_else(|| "Qwen linear layer: missing ssm_a".to_string())?;
    let ssm_norm = b
        .ssm_norm
        .ok_or_else(|| "Qwen linear layer: missing ssm_norm".to_string())?;
    let ssm_dt_bias = b.ssm_dt_proj_bias.unwrap_or_default();

    Ok(LinearAttnWeights {
        wqkv,
        wqkv_gate,
        ssm_conv1d,
        ssm_dt_bias,
        ssm_a,
        ssm_alpha,
        ssm_beta,
        ssm_norm,
        ssm_out,
    })
}

#[derive(Default)]
struct LayerBuilder {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    q_proj: Option<PackedTensor>,
    k_proj: Option<PackedTensor>,
    v_proj: Option<PackedTensor>,
    o_proj: Option<PackedTensor>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    attn_gate: Option<PackedTensor>,
    is_fused_qkv: bool,
    has_attention: bool,
    up_proj: Option<PackedTensor>,
    gate_proj: Option<PackedTensor>,
    down_proj: Option<PackedTensor>,
    // SSM
    has_ssm: bool,
    ssm_in_proj: Option<PackedTensor>,
    ssm_conv1d_weight: Option<PackedTensor>,
    ssm_conv1d_bias: Option<Vec<f32>>,
    ssm_dt_proj_weight: Option<PackedTensor>,
    ssm_dt_proj_bias: Option<Vec<f32>>,
    ssm_a_log: Option<Vec<f32>>,
    ssm_d_param: Option<Vec<f32>>,
    ssm_out_proj: Option<PackedTensor>,
    ssm_norm: Option<Vec<f32>>,
    ssm_alpha: Option<PackedTensor>,
    ssm_beta: Option<PackedTensor>,
}

fn packed_byte_size(ggml_type: u32, num_elements: usize) -> (usize, f64) {
    if ggml_type == GGML_TYPE_I2_S {
        return (num_elements.div_ceil(4) + 32, 0.25);
    }
    let elem_size = ggml_type_size(ggml_type).unwrap_or(2.0);
    let payload = (num_elements as f64 * elem_size).ceil() as usize;
    let blocks = (num_elements as f64 / 256.0).ceil() as usize;
    let overhead = match ggml_type {
        GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q3_K => blocks * 16,
        GGML_TYPE_Q8_K => blocks * 10,
        GGML_TYPE_Q6_K => blocks * 18,
        GGML_TYPE_Q2_K => blocks * 20,
        _ => 0,
    };
    (payload + overhead, elem_size)
}

fn dequant_to_f32(data: &[u8], ggml_type: u32, num_elements: usize) -> Vec<f32> {
    dequantize_slice(data, ggml_type, num_elements)
}
