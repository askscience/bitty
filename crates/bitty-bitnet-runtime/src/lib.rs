pub mod cpu_backend;
mod split_model;

use bitty_model::{BitNetMetadataError, BitNetModelMetadata};
use bitty_protocol::{ActivationDType, ActivationTensor, AssignedLayerRange, BitNetLogits};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::path::{Path, PathBuf};
use thiserror::Error;

use split_model::SplitBitNetModel;

pub type Result<T> = std::result::Result<T, BitNetRuntimeError>;

pub struct BitNetRuntime {
    model_path: PathBuf,
    metadata: BitNetModelMetadata,
    model: SplitBitNetModel,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitNetShard {
    pub range: AssignedLayerRange,
    pub owns_embedding: bool,
    pub owns_lm_head: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BitNetActivation {
    pub request_id: String,
    pub token_position: u32,
    pub source_layer: u32,
    pub target_layer: u32,
    pub token_count: u32,
    pub hidden_size: u32,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitNetKvCache;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
        }
    }
}

pub type BitNetTokenizer = bitty_candle_runtime::Tokenizer;

pub fn load_tokenizer(path: &Path, hf_model_id: Option<&str>) -> Result<BitNetTokenizer> {
    bitty_candle_runtime::Tokenizer::from_gguf_path(path, hf_model_id)
        .map_err(|err| BitNetRuntimeError::Backend(format!("tokenizer: {err}")))
}

impl BitNetRuntime {
    pub async fn load(path: &Path, hf_model_id: Option<&str>) -> Result<Self> {
        let metadata = BitNetModelMetadata::from_gguf_path(path)?;
        let source = path.to_string_lossy().to_string();
        let model = SplitBitNetModel::load(&source, 4096, hf_model_id).await?;
        Ok(Self {
            model_path: path.to_path_buf(),
            metadata,
            model,
        })
    }

    pub fn metadata(&self) -> BitNetModelMetadata {
        self.metadata.clone()
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn tokenizer(&self) -> &BitNetTokenizer {
        self.model.tokenizer()
    }

    pub fn reset_kv_cache(&mut self) {
        self.model.reset_kv_cache();
    }

    pub fn load_shard(&self, range: Range<u32>) -> Result<BitNetShard> {
        if range.start >= range.end || range.end > self.metadata.layer_count {
            return Err(BitNetRuntimeError::InvalidRange {
                start: range.start,
                end: range.end,
                layers: self.metadata.layer_count,
            });
        }

        Ok(BitNetShard {
            range: AssignedLayerRange {
                start_layer: range.start,
                end_layer_exclusive: range.end,
                quantization: self.metadata.quantization,
            },
            owns_embedding: range.start == 0,
            owns_lm_head: range.end == self.metadata.layer_count,
        })
    }

    pub async fn embed_tokens(
        &mut self,
        request_id: &str,
        token_position: u32,
        tokens: &[u32],
    ) -> Result<BitNetActivation> {
        let activation = self.model.embed_tokens(tokens);
        let payload = self.model.read_activation(&activation).await?;
        Ok(BitNetActivation {
            request_id: request_id.into(),
            token_position,
            source_layer: 0,
            target_layer: 0,
            token_count: tokens.len() as u32,
            hidden_size: self.metadata.hidden_size,
            payload,
        })
    }

    pub async fn forward_layers(
        &mut self,
        shard: &mut BitNetShard,
        cache: &mut BitNetKvCache,
        activation: BitNetActivation,
    ) -> Result<BitNetActivation> {
        if activation.target_layer > shard.range.start_layer {
            return Err(BitNetRuntimeError::StaleActivation {
                target_layer: activation.target_layer,
                shard_start: shard.range.start_layer,
            });
        }

        let _ = cache;
        let gpu_activation = if shard.owns_embedding && activation.source_layer == 0 {
            let tokens = activation
                .payload
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            self.model.embed_tokens(&tokens)
        } else {
            self.model
                .upload_activation(&activation.payload, activation.token_count as usize)
        };
        let gpu_activation = self.model.forward_layers(
            gpu_activation,
            shard.range.start_layer as usize..shard.range.end_layer_exclusive as usize,
        );
        let payload = self.model.read_activation(&gpu_activation).await?;
        let mut output = activation;
        output.source_layer = shard.range.start_layer;
        output.target_layer = shard.range.end_layer_exclusive;
        output.token_count = gpu_activation.tokens as u32;
        output.hidden_size = self.metadata.hidden_size;
        output.payload = payload;
        Ok(output)
    }

    pub async fn final_logits(&mut self, activation: BitNetActivation) -> Result<BitNetLogits> {
        let gpu_activation = self
            .model
            .upload_activation(&activation.payload, activation.token_count as usize);
        let logits_buffer = self.model.final_logits(gpu_activation);
        let logits = self.model.read_logits(&logits_buffer).await?;
        Ok(BitNetLogits::new(
            activation.request_id,
            activation.token_position,
            logits,
        ))
    }

    pub fn sample(&self, logits: &BitNetLogits, options: SamplingOptions) -> Result<u32> {
        if !logits.verify_checksum() {
            return Err(BitNetRuntimeError::ChecksumFailed);
        }
        if options.temperature <= 0.0 {
            return logits
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as u32)
                .ok_or(BitNetRuntimeError::EmptyLogits);
        }

        let mut scaled: Vec<f32> = logits
            .logits
            .iter()
            .map(|l| l / options.temperature)
            .collect();
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for val in scaled.iter_mut() {
            *val = (*val - max_logit).exp();
            sum += *val;
        }
        if sum <= 0.0 {
            return logits
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as u32)
                .ok_or(BitNetRuntimeError::EmptyLogits);
        }
        for val in scaled.iter_mut() {
            *val /= sum;
        }

        let dist = rand::distr::weighted::WeightedIndex::new(scaled)
            .map_err(|_| BitNetRuntimeError::EmptyLogits)?;
        let mut rng = rand::rng();
        use rand::Rng;
        Ok(rng.sample(&dist) as u32)
    }

    pub async fn generate_full(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        self.generate_stream(prompt, max_tokens, temperature, |_| {})
            .await
    }

    pub async fn generate_stream<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        mut on_delta: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.reset_kv_cache();
        let mut current_input = self
            .tokenizer()
            .encode(prompt, true)
            .map_err(|err| BitNetRuntimeError::Backend(err.to_string()))?;
        let mut generated_ids: Vec<u32> = Vec::new();
        let mut emitted: String = String::new();
        let mut cache = BitNetKvCache;
        let mut shard = self.load_shard(0..self.metadata.layer_count)?;
        for position in 0..max_tokens {
            let activation =
                BitNetActivation::from_tokens("local", position as u32, &current_input);
            let activation = self
                .forward_layers(&mut shard, &mut cache, activation)
                .await?;
            let logits = self.final_logits(activation).await?;
            let token = self.sample(
                &logits,
                SamplingOptions {
                    temperature,
                    top_p: 1.0,
                },
            )?;
            if self.is_stop_token(token) {
                break;
            }
            current_input.clear();
            current_input.push(token);
            generated_ids.push(token);

            let full = self
                .tokenizer()
                .decode(&generated_ids)
                .map_err(|err| BitNetRuntimeError::Backend(err.to_string()))?;
            if full.len() > emitted.len() && full.starts_with(&emitted) {
                let tail = &full[emitted.len()..];
                if !tail.ends_with('\u{FFFD}') {
                    on_delta(tail);
                    emitted = full;
                }
            }
        }
        let final_text = self
            .tokenizer()
            .decode(&generated_ids)
            .map_err(|err| BitNetRuntimeError::Backend(err.to_string()))?;
        if final_text.len() > emitted.len() && final_text.starts_with(&emitted) {
            on_delta(&final_text[emitted.len()..]);
        }
        Ok(final_text)
    }

    fn is_stop_token(&self, token: u32) -> bool {
        token == self.tokenizer().eos_token_id()
            || self.tokenizer().eot_token_id() == Some(token)
            || self.tokenizer().im_end_token_id() == Some(token)
    }
}

impl BitNetActivation {
    pub fn from_tokens(request_id: impl Into<String>, token_position: u32, tokens: &[u32]) -> Self {
        Self {
            request_id: request_id.into(),
            token_position,
            source_layer: 0,
            target_layer: 0,
            token_count: tokens.len() as u32,
            hidden_size: 0,
            payload: tokens
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect(),
        }
    }
}

impl From<BitNetActivation> for ActivationTensor {
    fn from(activation: BitNetActivation) -> Self {
        ActivationTensor::new(
            activation.request_id,
            activation.token_position,
            activation.source_layer,
            activation.target_layer,
            vec![activation.token_count, activation.hidden_size],
            ActivationDType::Fp16,
            activation.payload,
        )
    }
}

impl TryFrom<ActivationTensor> for BitNetActivation {
    type Error = BitNetRuntimeError;

    fn try_from(activation: ActivationTensor) -> Result<Self> {
        if !activation.verify_checksum() {
            return Err(BitNetRuntimeError::ChecksumFailed);
        }
        Ok(Self {
            request_id: activation.request_id,
            token_position: activation.token_position,
            source_layer: activation.source_layer,
            target_layer: activation.target_layer,
            token_count: activation.shape.first().copied().unwrap_or(1),
            hidden_size: activation
                .shape
                .get(1)
                .copied()
                .unwrap_or_else(|| activation.shape.first().copied().unwrap_or_default()),
            payload: activation.payload,
        })
    }
}

#[derive(Debug, Error)]
pub enum BitNetRuntimeError {
    #[error(transparent)]
    Metadata(#[from] BitNetMetadataError),
    #[error("BitNet backend failed: {0}")]
    Backend(String),
    #[error("missing model weight: {0}")]
    MissingWeight(String),
    #[error("invalid layer range {start}..{end} for model with {layers} layers")]
    InvalidRange { start: u32, end: u32, layers: u32 },
    #[error("activation checksum failed")]
    ChecksumFailed,
    #[error("activation target layer {target_layer} is stale for shard starting at {shard_start}")]
    StaleActivation { target_layer: u32, shard_start: u32 },
    #[error("empty logits")]
    EmptyLogits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_tensor_checksum_gate_rejects_corruption() {
        let mut activation =
            ActivationTensor::from(BitNetActivation::from_tokens("req", 0, &[1, 2]));
        activation.payload[0] = activation.payload[0].wrapping_add(1);
        let err = BitNetActivation::try_from(activation).unwrap_err();
        assert!(matches!(err, BitNetRuntimeError::ChecksumFailed));
    }

    #[tokio::test]
    #[ignore = "requires BITTY_GGUF_MODEL=/path/to/ggml-model-i2_s.gguf and a compatible device"]
    async fn split_local_logits_match_full_local_logits_for_temperature_zero() {
        let model_path = std::env::var("BITTY_GGUF_MODEL").expect("BITTY_GGUF_MODEL must be set");
        let path = Path::new(&model_path);

        let mut full = BitNetRuntime::load(path, None).await.unwrap();
        let mut split = BitNetRuntime::load(path, None).await.unwrap();
        let tokens = full.tokenizer().encode("Hello", true).unwrap();

        let mut full_cache = BitNetKvCache;
        let mut full_shard = full.load_shard(0..full.metadata.layer_count).unwrap();
        let full_activation = BitNetActivation::from_tokens("req", 0, &tokens);
        let full_activation = full
            .forward_layers(&mut full_shard, &mut full_cache, full_activation)
            .await
            .unwrap();
        let full_logits = full.final_logits(full_activation).await.unwrap();

        let midpoint = (split.metadata.layer_count / 2).max(1);
        let mut first_cache = BitNetKvCache;
        let mut second_cache = BitNetKvCache;
        let mut first_shard = split.load_shard(0..midpoint).unwrap();
        let mut second_shard = split
            .load_shard(midpoint..split.metadata.layer_count)
            .unwrap();
        let split_activation = BitNetActivation::from_tokens("req", 0, &tokens);
        let split_activation = split
            .forward_layers(&mut first_shard, &mut first_cache, split_activation)
            .await
            .unwrap();
        let split_activation = split
            .forward_layers(&mut second_shard, &mut second_cache, split_activation)
            .await
            .unwrap();
        let split_logits = split.final_logits(split_activation).await.unwrap();

        let full_token = full
            .sample(&full_logits, SamplingOptions::default())
            .unwrap();
        let split_token = split
            .sample(&split_logits, SamplingOptions::default())
            .unwrap();
        assert_eq!(split_token, full_token);
    }
}
