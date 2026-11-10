//! Executor for one slice ("stage") of the model: a contiguous range of transformer blocks.
//!
//! The compute engine is a patched llama.cpp that runs a layer range on hidden states. That
//! engine is not linked yet, so this type only loads and validates model metadata; every
//! compute call fails with [`ExecutorError::Failed`] naming the missing engine instead of
//! returning made-up activations.

use crate::{ExecutorError, LayerExecutor};
use async_trait::async_trait;
use bitty_model::{MetadataError, ModelMetadata};
use bitty_protocol::{ActivationTensor, AssignedLayerRange, Logits};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Backend id a node announces when it serves stages through llama.cpp.
pub const STAGE_BACKEND_ID: &str = "llama-cpp";

const ENGINE_MISSING: &str =
    "the llama.cpp stage engine is not built into this binary yet; no layer range can be executed";

#[derive(Debug, Error)]
pub enum StageError {
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("model has no transformer blocks")]
    NoLayers,
}

#[derive(Clone, Debug)]
pub struct StageExecutor {
    metadata: Arc<ModelMetadata>,
}

impl StageExecutor {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata: Arc::new(metadata),
        }
    }

    /// Reads the GGUF header only; tensor data stays on disk.
    pub async fn load(model_path: impl AsRef<Path>) -> Result<Self, StageError> {
        let path = model_path.as_ref().to_path_buf();
        let metadata = tokio::task::spawn_blocking(move || ModelMetadata::from_gguf_path(path))
            .await
            .expect("metadata task panicked")?;
        if metadata.layer_count == 0 {
            return Err(StageError::NoLayers);
        }
        Ok(Self::new(metadata))
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Whether this binary can actually run layers.
    pub fn engine_available(&self) -> bool {
        false
    }
}

#[async_trait]
impl LayerExecutor for StageExecutor {
    async fn execute_range(
        &self,
        range: &AssignedLayerRange,
        activation: ActivationTensor,
    ) -> Result<ActivationTensor, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }
        if range.end_layer_exclusive > self.metadata.layer_count || range.start_layer >= range.end_layer_exclusive {
            return Err(ExecutorError::Failed(format!(
                "layer range {}..{} is outside the model's {} blocks",
                range.start_layer, range.end_layer_exclusive, self.metadata.layer_count
            )));
        }
        Err(ExecutorError::Failed(ENGINE_MISSING.into()))
    }

    async fn final_logits(&self, activation: ActivationTensor) -> Result<Logits, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }
        Err(ExecutorError::Failed(ENGINE_MISSING.into()))
    }
}
