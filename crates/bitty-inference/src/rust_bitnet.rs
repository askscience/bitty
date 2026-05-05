use crate::{ExecutorError, LayerExecutor};
use async_trait::async_trait;
use bitty_bitnet_runtime::{BitNetKvCache, BitNetRuntime, BitNetRuntimeError};
use bitty_model::{BitNetMetadataError, BitNetModelMetadata, ShardPlan};
use bitty_protocol::{
    ActivationTensor, AssignedLayerRange, BitNetLogits, LayerAssignment, TokenOutput,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;
use tokio::sync::Mutex as AsyncMutex;

#[cfg(feature = "oxbitnet-backend")]
use futures::StreamExt;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BitNetBackendProbe {
    pub metadata: BitNetModelMetadata,
    pub tokenizer_available: bool,
}

impl BitNetBackendProbe {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, RustBitNetError> {
        let metadata = BitNetModelMetadata::from_gguf_path(model_path)?;
        let tokenizer_available = metadata.tokenizer_path.is_some();
        Ok(Self {
            metadata,
            tokenizer_available,
        })
    }

    pub fn layer_metadata(&self) -> Vec<bitty_protocol::LayerMetadata> {
        self.metadata.layer_metadata()
    }

    pub fn shard_plan(&self, assignment: &LayerAssignment) -> ShardPlan {
        self.metadata.shard_plan(assignment)
    }
}

#[derive(Clone)]
pub struct BitNetLayerExecutor {
    metadata: Arc<BitNetModelMetadata>,
    runtime: Option<Arc<AsyncMutex<BitNetRuntime>>>,
    kv_cache: Arc<Mutex<HashMap<KvCacheKey, Vec<u8>>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct KvCacheKey {
    request_id: String,
    layer_id: u32,
    token_position: u32,
}

impl BitNetLayerExecutor {
    pub fn new(metadata: BitNetModelMetadata) -> Self {
        Self {
            metadata: Arc::new(metadata),
            runtime: None,
            kv_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn load(model_path: impl AsRef<Path>) -> Result<Self, RustBitNetError> {
        let runtime = BitNetRuntime::load(model_path.as_ref()).await?;
        let metadata = runtime.metadata();
        Ok(Self {
            metadata: Arc::new(metadata),
            runtime: Some(Arc::new(AsyncMutex::new(runtime))),
            kv_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn metadata(&self) -> &BitNetModelMetadata {
        &self.metadata
    }

    pub fn clear_request(&self, request_id: &str) {
        self.kv_cache
            .lock()
            .expect("kv cache poisoned")
            .retain(|key, _| key.request_id != request_id);
    }

    pub fn deterministic_tokens(
        &self,
        request_id: &str,
        prompt_tokens: &[u32],
        max_new_tokens: u32,
    ) -> Vec<TokenOutput> {
        let mut state = prompt_tokens.iter().fold(0_u32, |acc, token| {
            acc.wrapping_mul(31).wrapping_add(*token)
        });
        (0..max_new_tokens.max(1))
            .map(|position| {
                state = state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223)
                    .wrapping_add(self.metadata.layer_count);
                TokenOutput {
                    request_id: request_id.into(),
                    token_position: position,
                    token_id: state % 32_000,
                    text: format!("<bitnet-rs:{}>", state % 32_000),
                    finished: position + 1 == max_new_tokens.max(1),
                    log_prob: 0.0,
                    gen_latency_us: 0,
                }
            })
            .collect()
    }
}

#[async_trait]
impl LayerExecutor for BitNetLayerExecutor {
    async fn execute_range(
        &self,
        range: &AssignedLayerRange,
        activation: ActivationTensor,
    ) -> Result<ActivationTensor, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }

        let started = Instant::now();
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ExecutorError::Failed(
                "BitNetLayerExecutor was created from metadata only; load a runtime-backed executor with BitNetLayerExecutor::load".into(),
            )
        })?;
        let mut runtime = runtime.lock().await;
        let mut shard = runtime
            .load_shard(range.start_layer..range.end_layer_exclusive)
            .map_err(|err| ExecutorError::Failed(err.to_string()))?;
        let runtime_activation = activation
            .clone()
            .try_into()
            .map_err(|err: BitNetRuntimeError| ExecutorError::Failed(err.to_string()))?;
        let mut cache = BitNetKvCache;
        let output = runtime
            .forward_layers(&mut shard, &mut cache, runtime_activation)
            .await
            .map_err(|err| ExecutorError::Failed(err.to_string()))?;
        let payload = output.payload.clone();
        {
            let mut cache = self.kv_cache.lock().expect("kv cache poisoned");
            for layer_id in range.start_layer..range.end_layer_exclusive {
                cache.insert(
                    KvCacheKey {
                        request_id: activation.request_id.clone(),
                        layer_id,
                        token_position: activation.token_position,
                    },
                    payload.clone(),
                );
            }
        }

        let mut output = ActivationTensor::from(output);
        output.compression = activation.compression;
        metrics::histogram!("dlm_bitnet_layer_executor_us")
            .record(started.elapsed().as_micros() as f64);
        Ok(output)
    }

    async fn decode_token_text(&self, token_id: u32) -> String {
        let Some(runtime) = self.runtime.as_ref() else {
            return format!("<bitnet-rs:{token_id}>");
        };
        let runtime = runtime.lock().await;
        match runtime.tokenizer().decode_one(token_id) {
            Ok(text) => text,
            Err(_) => char::REPLACEMENT_CHARACTER.to_string(),
        }
    }

    async fn final_logits(
        &self,
        activation: ActivationTensor,
    ) -> Result<BitNetLogits, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ExecutorError::Failed(
                "BitNetLayerExecutor was created from metadata only; load a runtime-backed executor with BitNetLayerExecutor::load".into(),
            )
        })?;
        let mut runtime = runtime.lock().await;
        let runtime_activation = activation
            .try_into()
            .map_err(|err: BitNetRuntimeError| ExecutorError::Failed(err.to_string()))?;
        runtime
            .final_logits(runtime_activation)
            .await
            .map_err(|err| ExecutorError::Failed(err.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum RustBitNetError {
    #[error(transparent)]
    Metadata(#[from] BitNetMetadataError),
    #[error(transparent)]
    Runtime(#[from] BitNetRuntimeError),
    #[cfg(feature = "oxbitnet-backend")]
    #[error("oxbitnet backend failed: {0}")]
    OxBitNet(String),
}

#[cfg(feature = "oxbitnet-backend")]
pub struct OxBitNetGenerator {
    inner: oxbitnet::BitNet,
}

#[cfg(feature = "oxbitnet-backend")]
impl OxBitNetGenerator {
    pub async fn load(model_path: impl AsRef<Path>) -> Result<Self, RustBitNetError> {
        let source = model_path.as_ref().to_string_lossy().to_string();
        let inner = oxbitnet::BitNet::load(&source, Default::default())
            .await
            .map_err(|err| RustBitNetError::OxBitNet(err.to_string()))?;
        Ok(Self { inner })
    }

    pub async fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, RustBitNetError> {
        let mut stream = self.inner.generate(
            prompt,
            oxbitnet::GenerateOptions {
                max_tokens,
                temperature,
                ..Default::default()
            },
        );
        let mut output = String::new();
        while let Some(piece) = stream.next().await {
            output.push_str(&piece);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_model::{BitNetModelFamily, BitNetTensorMetadata};
    use bitty_protocol::{ActivationDType, Quantization};

    fn metadata() -> BitNetModelMetadata {
        BitNetModelMetadata {
            architecture: BitNetModelFamily::BitNetB158,
            layer_count: 2,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            activation_bytes: 8,
            quantization: Quantization::Bit1,
            vocab_size: 32000,
            max_seq_len: 2048,
            rope_dimension_count: 4,
            tokenizer_path: None,
            tensors: vec![
                BitNetTensorMetadata {
                    name: "blk.0.attn_q.weight".into(),
                    layer_id: Some(0),
                    dimensions: vec![4, 4],
                    ggml_type: 36,
                    byte_len: 32,
                    offset: 0,
                },
                BitNetTensorMetadata {
                    name: "blk.1.ffn_down.weight".into(),
                    layer_id: Some(1),
                    dimensions: vec![4, 4],
                    ggml_type: 36,
                    byte_len: 32,
                    offset: 32,
                },
            ],
        }
    }

    #[tokio::test]
    async fn metadata_only_executor_rejects_layer_execution() {
        let executor = BitNetLayerExecutor::new(metadata());
        let range = AssignedLayerRange {
            start_layer: 0,
            end_layer_exclusive: 1,
            quantization: Quantization::Bit1,
        };
        let activation = ActivationTensor::new(
            "req",
            0,
            0,
            0,
            vec![4],
            ActivationDType::Fp16,
            vec![1, 2, 3, 4],
        );

        let err = executor
            .execute_range(&range, activation)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("runtime-backed executor"));
    }
}
