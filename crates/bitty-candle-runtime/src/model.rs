use candle_core::{Device, Tensor};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_nn::ops::rms_norm;
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::layers::{Attention, FFN, TransformerBlock};
use crate::load::{DeviceWeightStore, LoadError, LoadedModel};

use bitty_gguf_loader::{BackendKind, InferenceBackend};
use bitty_model::gguf::{GGML_TYPE_F16, GGML_TYPE_F32};

pub use crate::layers::ModelConfig;

fn ggml_to_ggml_dtype(ggml_type: u32) -> GgmlDType {
    match ggml_type {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => GgmlDType::F32, // fallback
    }
}

/// Build a QMatMul from raw GGUF bytes. For quantized types, stores
/// packed bytes on the GPU via QTensor for fused dequant+matmul.
/// For F32/F16 types, uploads directly as a regular Tensor.
fn build_qmatmul(
    ws: &DeviceWeightStore,
    name: &str,
    shape: &[usize],
    device: &Device,
) -> std::result::Result<QMatMul, ModelError> {
    let info = ws.get_info(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    let raw = ws.get_raw(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;

    let ggml_type = info.ggml_type;

    // F32/F16: use zero-copy raw buffer tensor path
    if ggml_type == GGML_TYPE_F32 || ggml_type == GGML_TYPE_F16 {
        let t = ws.get_raw_tensor(name, shape, ggml_type)
            .map_err(ModelError::Load)?;
        return Ok(QMatMul::Tensor(t));
    }

    // Quantized: use QTensor → QMatMul for fused dequant+matmul
    let dtype = ggml_to_ggml_dtype(ggml_type);
    let storage = QStorage::from_data(
        std::borrow::Cow::Borrowed(raw),
        device,
        dtype,
    ).map_err(|e| ModelError::Load(LoadError::Io(
        std::io::Error::new(std::io::ErrorKind::Other, format!("QStorage: {e}"))
    )))?;

    let shape: candle_core::Shape = shape.into();
    let qtensor = QTensor::new(storage, shape)
        .map_err(|e| ModelError::Candle(e))?;

    QMatMul::from_qtensor(qtensor)
        .map_err(|e| ModelError::Candle(e))
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

enum LmHead {
    Tied,
    F16(Tensor),
    Separate(Tensor),
}

pub struct CandleModel {
    device: Device,
    pub config: ModelConfig,
    embed_tokens: Tensor,
    layers: Vec<TransformerBlock>,
    final_norm: Tensor,
    lm_head: LmHead,
    kv_caches: Vec<KvCache>,
}

impl CandleModel {
    pub fn load(source: &str, device: &Device) -> Result<Self> {
        let loaded = crate::load::load_gguf(source, device)?;
        Self::build(loaded, device)
    }

    fn build(loaded: LoadedModel, device: &Device) -> Result<Self> {
        let config = loaded.config;
        let weights = &loaded.weights;

        let require_f32 = |ws: &DeviceWeightStore, name: &str, shape: &[usize]| -> Result<Tensor> {
            let info = ws.get_info(name)
                .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
            let raw = ws.get_raw(name)
                .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
            // Zero-copy path for native F32/F16 types (avoids intermediate Vec allocation)
            if info.ggml_type == GGML_TYPE_F32 || info.ggml_type == GGML_TYPE_F16 {
                return ws.get_raw_tensor(name, shape, info.ggml_type)
                    .map_err(ModelError::Load);
            }
            let element_count: usize = shape.iter().product();
            let data = crate::dequant::dequantize_tensor(raw, info.ggml_type, element_count);
            Ok(Tensor::from_vec(data, shape, device)?)
        };

        let require_i2s_with_scale = |ws: &DeviceWeightStore, weight_name: &str,
                                      scale_name: &str, out_dim: usize, in_dim: usize| -> Result<Tensor> {
            let weight_raw = ws.get_raw(weight_name)
                .ok_or_else(|| ModelError::MissingWeight(weight_name.to_string()))?;
            let scale_raw = ws.get_raw(scale_name)
                .ok_or_else(|| ModelError::MissingWeight(scale_name.to_string()))?;
            let scale: &[f32] = bytemuck::cast_slice(scale_raw);
            let dequantized = dequantize_i2_s_with_scale(weight_raw, scale, out_dim, in_dim);
            Ok(Tensor::from_vec(dequantized, &[out_dim, in_dim], device)?)
        };

        let embed_tokens = require_f32(weights, "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size])?;
        let final_norm = require_f32(weights, "model.norm.weight",
            &[config.hidden_size])?;

        let is_ternary = weights.get_info("model.layers.0.self_attn.q_proj.weight")
            .map(|info| info.ggml_type == bitty_model::gguf::GGML_TYPE_I2_S)
            .unwrap_or(false);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut kv_caches = Vec::with_capacity(config.num_hidden_layers);
        let head_dim = config.head_dim();

        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let q_out = config.num_attention_heads * head_dim;
            let k_out = config.num_key_value_heads * head_dim;
            let v_out = config.num_key_value_heads * head_dim;
            let o_in = config.num_attention_heads * head_dim;

            let (q_proj, k_proj, v_proj, o_proj) = if is_ternary {
                (
                    QMatMul::Tensor(require_i2s_with_scale(weights,
                        &format!("{p}.self_attn.q_proj.weight"),
                        &format!("{p}.self_attn.q_proj.weight_scale"),
                        q_out, config.hidden_size)?),
                    QMatMul::Tensor(require_i2s_with_scale(weights,
                        &format!("{p}.self_attn.k_proj.weight"),
                        &format!("{p}.self_attn.k_proj.weight_scale"),
                        k_out, config.hidden_size)?),
                    QMatMul::Tensor(require_i2s_with_scale(weights,
                        &format!("{p}.self_attn.v_proj.weight"),
                        &format!("{p}.self_attn.v_proj.weight_scale"),
                        v_out, config.hidden_size)?),
                    QMatMul::Tensor(require_i2s_with_scale(weights,
                        &format!("{p}.self_attn.o_proj.weight"),
                        &format!("{p}.self_attn.o_proj.weight_scale"),
                        config.hidden_size, o_in)?),
                )
            } else {
                (
                    build_qmatmul(weights, &format!("{p}.self_attn.q_proj.weight"),
                        &[q_out, config.hidden_size], device)?,
                    build_qmatmul(weights, &format!("{p}.self_attn.k_proj.weight"),
                        &[k_out, config.hidden_size], device)?,
                    build_qmatmul(weights, &format!("{p}.self_attn.v_proj.weight"),
                        &[v_out, config.hidden_size], device)?,
                    build_qmatmul(weights, &format!("{p}.self_attn.o_proj.weight"),
                        &[config.hidden_size, o_in], device)?,
                )
            };

            let (up_proj, down_proj, gate_proj) = if is_ternary {
                let up = QMatMul::Tensor(require_i2s_with_scale(weights,
                    &format!("{p}.mlp.up_proj.weight"),
                    &format!("{p}.mlp.up_proj.weight_scale"),
                    config.intermediate_size, config.hidden_size)?);
                let down = QMatMul::Tensor(require_i2s_with_scale(weights,
                    &format!("{p}.mlp.down_proj.weight"),
                    &format!("{p}.mlp.down_proj.weight_scale"),
                    config.hidden_size, config.intermediate_size)?);
                let gate = weights.has(&format!("{p}.mlp.gate_proj.weight")).then(|| {
                    require_i2s_with_scale(weights,
                        &format!("{p}.mlp.gate_proj.weight"),
                        &format!("{p}.mlp.gate_proj.weight_scale"),
                        config.intermediate_size, config.hidden_size)
                        .map(QMatMul::Tensor)
                }).transpose()?;
                (up, down, gate)
            } else {
                let up = build_qmatmul(weights, &format!("{p}.mlp.up_proj.weight"),
                    &[config.intermediate_size, config.hidden_size], device)?;
                let down = build_qmatmul(weights, &format!("{p}.mlp.down_proj.weight"),
                    &[config.hidden_size, config.intermediate_size], device)?;
                let gate = weights.has(&format!("{p}.mlp.gate_proj.weight")).then(|| {
                    build_qmatmul(weights, &format!("{p}.mlp.gate_proj.weight"),
                        &[config.intermediate_size, config.hidden_size], device)
                }).transpose()?;
                (up, down, gate)
            };

            let q_bias = weights.has(&format!("{p}.self_attn.q_proj.bias"))
                .then(|| require_f32(weights, &format!("{p}.self_attn.q_proj.bias"), &[q_out]))
                .transpose()?;
            let k_bias = weights.has(&format!("{p}.self_attn.k_proj.bias"))
                .then(|| require_f32(weights, &format!("{p}.self_attn.k_proj.bias"), &[k_out]))
                .transpose()?;
            let v_bias = weights.has(&format!("{p}.self_attn.v_proj.bias"))
                .then(|| require_f32(weights, &format!("{p}.self_attn.v_proj.bias"), &[v_out]))
                .transpose()?;

            let attention = Attention::new(
                q_proj, k_proj, v_proj, o_proj,
                q_bias, k_bias, v_bias,
                config.num_attention_heads, config.num_key_value_heads,
                head_dim, config.is_qwen,
                config.rope_theta, config.rope_style,
                config.max_position_embeddings,
            );

            let ffn = FFN::new(up_proj, gate_proj, down_proj);

            let input_ln = require_f32(weights,
                &format!("{p}.input_layernorm.weight"), &[config.hidden_size])?;
            let post_attn_ln = require_f32(weights,
                &format!("{p}.post_attention_layernorm.weight"), &[config.hidden_size])?;

            let post_attention_norm = if config.is_gemma3 {
                weights.has(&format!("{p}.post_attention_layernorm.weight"))
                    .then(|| require_f32(weights,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.hidden_size]))
                    .transpose()
                    .ok()
                    .flatten()
            } else {
                None
            };
            let pre_ffn_norm = if config.is_gemma3 {
                weights.has(&format!("{p}.pre_feedforward_layernorm.weight"))
                    .then(|| require_f32(weights,
                        &format!("{p}.pre_feedforward_layernorm.weight"),
                        &[config.hidden_size]))
                    .transpose()
                    .ok()
                    .flatten()
            } else {
                None
            };
            let post_ffn_norm = if config.is_gemma3 {
                weights.has(&format!("{p}.post_feedforward_layernorm.weight"))
                    .then(|| require_f32(weights,
                        &format!("{p}.post_feedforward_layernorm.weight"),
                        &[config.hidden_size]))
                    .transpose()
                    .ok()
                    .flatten()
            } else {
                None
            };

            layers.push(TransformerBlock::new(
                input_ln, post_attn_ln,
                post_attention_norm, pre_ffn_norm, post_ffn_norm,
                attention, ffn, config.rms_norm_eps,
            ));
            kv_caches.push(KvCache::new(config.max_position_embeddings));
        }

        let lm_head = if config.tie_word_embeddings || !weights.has("lm_head.weight") {
            LmHead::Tied
        } else if config.lm_head_f16 {
            LmHead::F16(
                require_f32(weights, "lm_head.weight", &[config.vocab_size, config.hidden_size])?
            )
        } else if is_ternary {
            LmHead::Separate(
                require_i2s_with_scale(weights,
                    "lm_head.weight", "lm_head.weight_scale",
                    config.vocab_size, config.hidden_size)?
            )
        } else {
            LmHead::Separate(
                require_f32(weights, "lm_head.weight", &[config.vocab_size, config.hidden_size])?
            )
        };

        Ok(Self {
            device: device.clone(),
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            kv_caches,
        })
    }

    pub fn reset_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
    }

    pub fn forward(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let n = token_ids.len();
        let ids_tensor = Tensor::from_vec(
            token_ids.iter().map(|&i| i as u32).collect(),
            n,
            &self.device,
        )?;

        let mut hidden = self.embed_tokens.embedding(&ids_tensor)?;
        if let Some(scale) = self.config.embedding_scale {
            hidden = (hidden * scale as f64)?;
        }

        for i in 0..self.layers.len() {
            let out = self.layers[i].forward(&hidden, n, &mut self.kv_caches[i])?;
            hidden = out;
        }

        let normed = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;

        let last_token = if n > 1 {
            normed.narrow(0, n - 1, 1)?
        } else {
            normed
        };

        let logits = match &self.lm_head {
            LmHead::Tied => {
                let w = self.embed_tokens.clone();
                last_token.reshape((1, self.config.hidden_size))?
                    .matmul(&w.t()?)?
                    .reshape(self.config.vocab_size)?
            }
            LmHead::F16(w) | LmHead::Separate(w) => {
                last_token.reshape((1, self.config.hidden_size))?
                    .matmul(&w.t()?)?
                    .reshape(self.config.vocab_size)?
            }
        };

        let mut logits_f32 = logits.to_vec1::<f32>()?;
        if let Some(softcap) = self.config.final_logit_softcap {
            for v in logits_f32.iter_mut() {
                *v = softcap * (*v / softcap).tanh();
            }
        }
        Ok(logits_f32)
    }

    pub fn embed_tokens(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        let ids_tensor = Tensor::from_vec(
            token_ids.iter().map(|&i| i as u32).collect(),
            token_ids.len(),
            &self.device,
        )?;
        let emb = self.embed_tokens.embedding(&ids_tensor)?;
        if let Some(scale) = self.config.embedding_scale {
            Ok((emb * scale as f64)?)
        } else {
            Ok(emb)
        }
    }

    pub fn forward_layers<F>(
        &mut self,
        activation: &Tensor,
        range: std::ops::Range<usize>,
        on_layer: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(usize, &Tensor) -> Result<()>,
    {
        let n = activation.dims()[0];
        let mut hidden = activation.clone();

        for layer_id in range {
            let out = self.layers[layer_id].forward(&hidden, n, &mut self.kv_caches[layer_id])?;
            self.kv_caches[layer_id].seq_len += n;
            hidden = out;
            on_layer(layer_id, &hidden)?;
        }

        Ok(hidden)
    }

    pub fn final_norm_and_logits(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let normed = rms_norm(hidden, &self.final_norm, self.config.rms_norm_eps)?;
        let last_token = normed.narrow(0, hidden.dim(0)? - 1, 1)?;
        let logits = match &self.lm_head {
            LmHead::Tied => {
                let w = self.embed_tokens.clone();
                last_token.matmul(&w.t()?)?.reshape(self.config.vocab_size)?
            }
            LmHead::F16(w) | LmHead::Separate(w) => {
                last_token.matmul(&w.t()?)?.reshape(self.config.vocab_size)?
            }
        };
        let mut logits_f32 = logits.to_vec1::<f32>()?;
        if let Some(softcap) = self.config.final_logit_softcap {
            for v in logits_f32.iter_mut() {
                *v = softcap * (*v / softcap).tanh();
            }
        }
        Ok(logits_f32)
    }

    pub fn layers_and_caches_mut(&mut self) -> (&mut Vec<TransformerBlock>, &mut Vec<KvCache>) {
        (&mut self.layers, &mut self.kv_caches)
    }

    pub fn layers_mut(&mut self) -> &mut Vec<TransformerBlock> {
        &mut self.layers
    }

    pub fn kv_caches_mut(&mut self) -> &mut Vec<KvCache> {
        &mut self.kv_caches
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn final_norm(&self) -> &Tensor {
        &self.final_norm
    }

    pub fn embed_tokens_tensor(&self) -> &Tensor {
        &self.embed_tokens
    }

    pub fn lm_head_weight(&self) -> Option<&Tensor> {
        match &self.lm_head {
            LmHead::F16(w) | LmHead::Separate(w) => Some(w),
            LmHead::Tied => Some(&self.embed_tokens),
        }
    }
}

fn dequantize_i2_s_with_scale(raw: &[u8], scale: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let blocks_per_row = (in_dim + 127) / 128;
    let mut result = vec![0.0f32; out_dim * in_dim];

    for r in 0..out_dim {
        let row_scale = if r < scale.len() { scale[r] } else { 1.0f32 };
        for b in 0..blocks_per_row {
            let block_start = (r * blocks_per_row + b) * 32;
            if block_start + 32 > raw.len() { break; }
            let block: &[u8; 32] = raw[block_start..block_start + 32].try_into().unwrap();
            let decoded = bitty_model::gguf::decode_i2_s_block(block);
            let base_idx = r * in_dim + b * 128;
            let count = 128.min(in_dim - b * 128);
            for i in 0..count {
                result[base_idx + i] = decoded[i] as f32 * row_scale;
            }
        }
    }
    result
}

impl InferenceBackend for CandleModel {
    type Error = ModelError;

    fn load(path: &Path, _hf_source: Option<&str>) -> std::result::Result<Self, Self::Error> {
        let source = path.to_string_lossy().to_string();
        let device = crate::auto_device();
        Self::load(&source, &device)
    }

    fn forward(&mut self, token_ids: &[u32]) -> std::result::Result<Vec<f32>, Self::Error> {
        self.forward(token_ids)
    }

    fn reset_kv_cache(&mut self) {
        self.reset_kv_cache();
    }

    fn backend_kind(&self) -> BackendKind {
        match self.device() {
            Device::Cpu => BackendKind::Cpu,
            #[cfg(feature = "cuda")]
            _ if self.device().is_cuda() => BackendKind::Cuda,
            #[cfg(feature = "metal")]
            _ if self.device().is_metal() => BackendKind::Metal,
            _ => BackendKind::Cpu,
        }
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
