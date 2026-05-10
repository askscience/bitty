use crate::{BitNetRuntimeError, Result};
use bitty_candle_runtime::{CandleModel, Tokenizer, rms_norm};
use candle_core::Tensor;
use std::ops::Range;

pub struct SplitBitNetModel {
    model: CandleModel,
    pub config: bitty_candle_runtime::ModelConfig,
    tokenizer: Tokenizer,
}

pub struct GpuActivation {
    pub buffer: Tensor,
    pub tokens: usize,
}

impl SplitBitNetModel {
    pub async fn load(source: &str, _max_seq_len: usize, hf_model_id: Option<&str>) -> Result<Self> {
        let device = bitty_candle_runtime::auto_device();
        let model = CandleModel::load(source, &device)
            .map_err(|err| BitNetRuntimeError::Backend(err.to_string()))?;
        let config = model.config.clone();
        let tokenizer = Tokenizer::from_gguf_path(std::path::Path::new(source), hf_model_id)
            .map_err(|err| BitNetRuntimeError::Backend(format!("tokenizer: {err}")))?;

        Ok(Self { model, config, tokenizer })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn reset_kv_cache(&mut self) {
        self.model.reset_kv_cache();
    }

    pub fn embed_tokens(&mut self, token_ids: &[u32]) -> GpuActivation {
        let n = token_ids.len();
        let tensor = self.model.embed_tokens(token_ids)
            .expect("embed_tokens failed");
        GpuActivation { buffer: tensor, tokens: n }
    }

    pub fn upload_activation(&self, payload: &[u8], tokens: usize) -> GpuActivation {
        let floats: &[f32] = bytemuck::cast_slice(payload);
        let tensor = Tensor::from_vec(
            floats.to_vec(),
            &[tokens, self.config.hidden_size],
            self.model.device(),
        )
        .expect("upload_activation failed");
        GpuActivation { buffer: tensor, tokens }
    }

    pub fn forward_layers(
        &mut self,
        activation: GpuActivation,
        range: Range<usize>,
    ) -> GpuActivation {
        let mut hidden = activation.buffer;
        let n = activation.tokens;
        let (layers, caches) = self.model.layers_and_caches_mut();

        for layer_id in range {
            if layer_id >= layers.len() || layer_id >= caches.len() {
                break;
            }
            let out = layers[layer_id].forward(&hidden, n, &mut caches[layer_id])
                .expect("layer forward failed");
            hidden = out;
        }
        GpuActivation { buffer: hidden, tokens: activation.tokens }
    }

    pub fn final_logits(&self, activation: GpuActivation) -> Tensor {
        let normed = rms_norm(
            &activation.buffer,
            self.model.final_norm(),
            self.model.config.rms_norm_eps,
        )
        .expect("final_norm failed");

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let n = activation.tokens;

        let last_token = if n > 1 {
            normed.narrow(0, n - 1, 1).expect("narrow failed")
        } else {
            normed
        };

        let lm_head = self.model.lm_head_weight()
            .expect("lm_head weight not found")
            .clone();

        last_token
            .reshape((1, hidden_size))
            .expect("reshape failed")
            .matmul(&lm_head.t().expect("transpose failed"))
            .expect("matmul failed")
            .reshape(vocab_size)
            .expect("reshape failed")
    }

    pub async fn read_activation(&self, activation: &GpuActivation) -> Result<Vec<u8>> {
        let floats = activation.buffer.flatten_all()
            .map_err(|e| BitNetRuntimeError::Backend(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| BitNetRuntimeError::Backend(e.to_string()))?;
        Ok(bytemuck::cast_slice(&floats).to_vec())
    }

    pub async fn read_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        logits.to_vec1::<f32>()
            .map_err(|e| BitNetRuntimeError::Backend(e.to_string()))
    }
}
