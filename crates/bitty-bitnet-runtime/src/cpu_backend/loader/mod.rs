//! GGUF model loading for the CPU inference backend.
//!
//! Reads GGUF files, classifies tensors by their role, and builds
//! the CpuWeights structure used by all layers.

pub mod metadata;
pub mod names;

use crate::cpu_backend::dequant::dequantize_slice;
use crate::cpu_backend::types::*;
use names::TensorRole;
use bytes::Bytes;
use oxbitnet::model::gguf::{
    self, GgufMetadata, GgufParser, GGML_TYPE_I2_S, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
};
use std::collections::HashMap;

/// Load a GGUF file and return metadata, tokenizer, and weights.
pub fn load_gguf(data: &[u8]) -> Result<(GgufMetadata, oxbitnet::Tokenizer, CpuWeights), String> {
    let mut parser = GgufParser::new(data);
    let gguf = parser
        .parse()
        .map_err(|e| format!("GGUF parse error: {e}"))?;

    let tokenizer = oxbitnet::Tokenizer::from_gguf_metadata(&gguf.metadata)
        .map_err(|e| format!("Tokenizer error: {e}"))?;

    let metadata = gguf.metadata.clone();

    // Collect tensors by layer
    let mut embed_tokens = Vec::new();
    let mut final_norm = Vec::new();
    let mut lm_head: Option<LmHead> = None;
    let mut layer_builders: HashMap<usize, LayerBuilder> = HashMap::new();

    for tensor in &gguf.tensors {
        let data_offset = gguf.tensor_data_offset + tensor.offset as usize;
        let num_elements: usize = tensor.shape.iter().map(|&d| d as usize).product();
        let (byte_size, _) = packed_byte_size(tensor.tensor_type, num_elements);
        let tensor_data = &data[data_offset..(data_offset + byte_size).min(data.len())];

        let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
        let packed = PackedTensor {
            data: Bytes::copy_from_slice(tensor_data),
            ggml_type: tensor.tensor_type,
            shape,
            name: tensor.name.clone(),
        };

        if let Some(role) = names::classify(&tensor.name) {
            match role {
                TensorRole::EmbedTokens => {
                    embed_tokens = dequant_to_f32(tensor_data, tensor.tensor_type, num_elements);
                }
                TensorRole::FinalNorm => {
                    final_norm = dequant_to_f32(tensor_data, tensor.tensor_type, num_elements);
                }
                TensorRole::LmHead => {
                    lm_head = Some(LmHead::Packed(packed));
                }
                TensorRole::InputNorm(l) => {
                    layer_builders.entry(l).or_default().input_ln =
                        dequant_to_f32(tensor_data, tensor.tensor_type, num_elements);
                }
                TensorRole::PostAttnNorm(l) => {
                    layer_builders.entry(l).or_default().post_attn_ln =
                        dequant_to_f32(tensor_data, tensor.tensor_type, num_elements);
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
                        tensor.tensor_type,
                        num_elements,
                    ));
                }
                TensorRole::KNorm(l) => {
                    layer_builders.entry(l).or_default().k_norm = Some(dequant_to_f32(
                        tensor_data,
                        tensor.tensor_type,
                        num_elements,
                    ));
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
                // SSM tensors
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
                        tensor.tensor_type,
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
                        tensor.tensor_type,
                        num_elements,
                    ));
                }
                TensorRole::SsmA(l) => {
                    layer_builders.entry(l).or_default().ssm_a_log = Some(dequant_to_f32(
                        tensor_data,
                        tensor.tensor_type,
                        num_elements,
                    ));
                }
                TensorRole::SsmAlphaWeight(l) | TensorRole::SsmBetaWeight(l) => {
                    // alpha/beta are auxiliary SSM params — store or ignore
                    layer_builders.entry(l).or_default().has_ssm = true;
                }
                TensorRole::SsmDParam(l) => {
                    layer_builders.entry(l).or_default().ssm_d_param = Some(dequant_to_f32(
                        tensor_data,
                        tensor.tensor_type,
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
                        tensor.tensor_type,
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
    let mut layers = Vec::new();
    let mut ssm_count = 0;

    for i in indices {
        let b = layer_builders.remove(&i).unwrap();
        let builder = b; // take ownership

        let mlp = MlpBlock {
            up_proj: builder.up_proj.unwrap_or_else(PackedTensor::dummy),
            gate_proj: builder.gate_proj.unwrap_or_else(PackedTensor::dummy),
            down_proj: builder.down_proj.unwrap_or_else(PackedTensor::dummy),
        };

        let kind = if builder.has_ssm {
            ssm_count += 1;
            eprintln!("info: layer {i} uses SSM");
            // For Qwen-style SSM, the QKV fused tensor serves as in_proj
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
                is_fused_qkv: builder.is_fused_qkv,
            })
        } else {
            // Passthrough: identity (shouldn't happen)
            eprintln!("warning: layer {i} has neither attention nor SSM weights");
            continue;
        };

        layers.push(CpuLayer {
            kind,
            input_ln: if builder.input_ln.is_empty() {
                vec![1.0f32; 1]
            } else {
                builder.input_ln
            },
            post_attn_ln: if builder.post_attn_ln.is_empty() {
                vec![1.0f32; 1]
            } else {
                builder.post_attn_ln
            },
            mlp,
            layer_idx: i,
        });
    }

    if ssm_count > 0 {
        eprintln!("info: {} SSM layers loaded", ssm_count);
    }

    let weights = CpuWeights {
        embed_tokens,
        final_norm,
        layers,
        lm_head,
    };
    Ok((metadata, tokenizer, weights))
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
}

fn packed_byte_size(ggml_type: u32, num_elements: usize) -> (usize, f64) {
    if ggml_type == GGML_TYPE_I2_S {
        return (num_elements.div_ceil(4) + 32, 0.25);
    }
    let elem_size = gguf::ggml_type_size(ggml_type).unwrap_or(2.0);
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
