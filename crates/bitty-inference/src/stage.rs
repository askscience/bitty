//! Executor for one slice ("stage") of the model: a contiguous range of transformer blocks.
//!
//! The compute engine is a patched llama.cpp running as a child process (`bitty-stage`, see
//! `engine/`). It loads only the tensors of its range, takes token ids (first stage) or hidden
//! states (later stages) and returns hidden states, or logits on the final stage. Without the
//! engine binary every compute call fails with [`ExecutorError::Failed`] instead of returning
//! made-up activations.

use crate::stage_process::{
    find_stage_engine, StageOutput, StageProcess, StageProcessConfig, StageProcessError,
};
use crate::{ExecutorError, LayerExecutor};
use async_trait::async_trait;
use bitty_model::{MetadataError, ModelMetadata};
use bitty_protocol::{ActivationDType, ActivationTensor, AssignedLayerRange, Logits};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Backend id a node announces when it serves stages through llama.cpp.
pub const STAGE_BACKEND_ID: &str = "llama-cpp";

/// Request id the worker keepalive uses; it never reaches the model.
const KEEPALIVE_REQUEST: &str = "keepalive";

const ENGINE_MISSING: &str = "the llama.cpp stage engine (bitty-stage) was not found: set \
     BITTY_STAGE_BIN or install it next to this binary; no layer range can be executed";

#[derive(Debug, Error)]
pub enum StageError {
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("model has no transformer blocks")]
    NoLayers,
}

/// Tuning of the engine process.
#[derive(Clone, Copy, Debug)]
pub struct StageOptions {
    /// 0 lets the engine choose.
    pub threads: u32,
    /// Context length of one request, in tokens.
    pub context: u32,
    /// Requests the stage keeps state for at the same time.
    pub sessions: u32,
}

impl Default for StageOptions {
    fn default() -> Self {
        Self {
            threads: 0,
            context: 8192,
            sessions: 4,
        }
    }
}

struct Session {
    id: u64,
    next_position: u32,
}

#[derive(Default)]
struct Engine {
    process: Option<StageProcess>,
    sessions: HashMap<String, Session>,
    next_session_id: u64,
}

#[derive(Clone)]
pub struct StageExecutor {
    metadata: Arc<ModelMetadata>,
    model_path: Option<PathBuf>,
    engine_path: Option<PathBuf>,
    options: StageOptions,
    engine: Arc<Mutex<Engine>>,
}

impl std::fmt::Debug for StageExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageExecutor")
            .field("model_path", &self.model_path)
            .field("engine_path", &self.engine_path)
            .finish_non_exhaustive()
    }
}

impl StageExecutor {
    /// Metadata only: such an executor has no model file and cannot run layers.
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata: Arc::new(metadata),
            model_path: None,
            engine_path: None,
            options: StageOptions::default(),
            engine: Arc::default(),
        }
    }

    /// Reads the GGUF header only; the engine process starts on the first compute call, once
    /// the layer range is known.
    pub async fn load(model_path: impl AsRef<Path>) -> Result<Self, StageError> {
        let path = model_path.as_ref().to_path_buf();
        let header_path = path.clone();
        let metadata =
            tokio::task::spawn_blocking(move || ModelMetadata::from_gguf_path(header_path))
                .await
                .expect("metadata task panicked")?;
        if metadata.layer_count == 0 {
            return Err(StageError::NoLayers);
        }
        let mut executor = Self::new(metadata);
        executor.model_path = Some(path);
        executor.engine_path = find_stage_engine();
        Ok(executor)
    }

    pub fn with_engine(mut self, engine_path: impl Into<PathBuf>) -> Self {
        self.engine_path = Some(engine_path.into());
        self
    }

    pub fn with_options(mut self, options: StageOptions) -> Self {
        self.options = options;
        self
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Whether this executor can actually run layers.
    pub fn engine_available(&self) -> bool {
        self.model_path.is_some() && self.engine_path.is_some()
    }

    /// Tokenizes text with the model vocabulary. Needs a running stage (any range).
    pub async fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<u32>, ExecutorError> {
        let text = text.to_string();
        self.with_running(move |process| process.tokenize(&text, add_special))
            .await
    }

    async fn blocking<T, F>(&self, work: F) -> Result<T, ExecutorError>
    where
        T: Send + 'static,
        F: FnOnce(&StageExecutor, &mut Engine) -> Result<T, ExecutorError> + Send + 'static,
    {
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let engine = this.engine.clone();
            let mut engine = engine.lock().unwrap_or_else(|poison| poison.into_inner());
            work(&this, &mut engine)
        })
        .await
        .map_err(|err| ExecutorError::Failed(format!("stage task failed: {err}")))?
    }

    async fn with_running<T, F>(&self, work: F) -> Result<T, ExecutorError>
    where
        T: Send + 'static,
        F: FnOnce(&mut StageProcess) -> Result<T, StageProcessError> + Send + 'static,
    {
        self.blocking(move |_, engine| {
            let process = engine
                .process
                .as_mut()
                .ok_or_else(|| ExecutorError::Failed("no stage is loaded yet".into()))?;
            let result = work(process);
            engine.settle(result)
        })
        .await
    }

    /// Starts the engine for `range`, replacing a process that serves another range.
    fn ensure_process(&self, engine: &mut Engine, range: &AssignedLayerRange) -> Result<(), ExecutorError> {
        let matches = engine.process.as_ref().is_some_and(|process| {
            let info = process.info();
            info.start_layer == range.start_layer && info.end_layer_exclusive == range.end_layer_exclusive
        });
        if matches {
            return Ok(());
        }
        let (Some(model), Some(engine_path)) = (&self.model_path, &self.engine_path) else {
            return Err(ExecutorError::Failed(ENGINE_MISSING.into()));
        };
        engine.process = None;
        engine.sessions.clear();
        let process = StageProcess::spawn(&StageProcessConfig {
            engine: engine_path.clone(),
            model: model.clone(),
            start_layer: range.start_layer,
            end_layer_exclusive: range.end_layer_exclusive,
            threads: self.options.threads,
            context: self.options.context,
            sessions: self.options.sessions,
        })
        .map_err(|err| ExecutorError::Failed(err.to_string()))?;
        engine.process = Some(process);
        Ok(())
    }
}

impl Engine {
    /// Maps an engine result; a broken pipe means the process is gone, so forget it.
    fn settle<T>(&mut self, result: Result<T, StageProcessError>) -> Result<T, ExecutorError> {
        result.map_err(|err| {
            if !matches!(err, StageProcessError::Engine(_)) {
                self.process = None;
                self.sessions.clear();
            }
            ExecutorError::Failed(err.to_string())
        })
    }
}

#[async_trait]
impl LayerExecutor for StageExecutor {
    /// Token ids go into a range that starts at layer 0, hidden states into any other range.
    /// `token_position == 0` opens the request on this stage; later calls continue it in order.
    /// The final range returns the logits of the last token as an `F32` activation whose
    /// `target_layer` is the layer count; [`LayerExecutor::final_logits`] unwraps it.
    async fn execute_range(
        &self,
        range: &AssignedLayerRange,
        activation: ActivationTensor,
    ) -> Result<ActivationTensor, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }
        let layer_count = self.metadata.layer_count;
        if range.end_layer_exclusive > layer_count || range.start_layer >= range.end_layer_exclusive {
            return Err(ExecutorError::Failed(format!(
                "layer range {}..{} is outside the model's {} blocks",
                range.start_layer, range.end_layer_exclusive, layer_count
            )));
        }
        let range = range.clone();
        if activation.request_id == KEEPALIVE_REQUEST {
            return self
                .blocking(move |this, engine| {
                    this.ensure_process(engine, &range)?;
                    let result = engine.process.as_mut().expect("process started").ping();
                    engine.settle(result)?;
                    Ok(activation)
                })
                .await;
        }

        let hidden_size = self.metadata.hidden_size as usize;
        self.blocking(move |this, engine| {
            let first = range.start_layer == 0;
            let tokens = if first {
                if activation.payload.is_empty() || activation.payload.len() % 4 != 0 {
                    return Err(ExecutorError::Failed("the first stage expects u32 token ids".into()));
                }
                activation.payload.len() / 4
            } else {
                let row = hidden_size * 4;
                if activation.dtype != ActivationDType::F32
                    || activation.payload.is_empty()
                    || activation.payload.len() % row != 0
                {
                    return Err(ExecutorError::Failed(format!(
                        "stage {}..{} expects f32 hidden states of width {hidden_size}",
                        range.start_layer, range.end_layer_exclusive
                    )));
                }
                if activation.target_layer != range.start_layer {
                    return Err(ExecutorError::Failed(format!(
                        "hidden states come from layer {}, this stage starts at {}",
                        activation.target_layer, range.start_layer
                    )));
                }
                activation.payload.len() / row
            };

            this.ensure_process(engine, &range)?;

            if activation.token_position == 0 {
                if let Some(old) = engine.sessions.remove(&activation.request_id) {
                    let result = engine.process.as_mut().expect("process started").end_session(old.id);
                    engine.settle(result)?;
                }
                engine.next_session_id += 1;
                let id = engine.next_session_id;
                engine
                    .sessions
                    .insert(activation.request_id.clone(), Session { id, next_position: 0 });
            }
            let (session_id, position) = match engine.sessions.get(&activation.request_id) {
                Some(session) => (session.id, session.next_position),
                None => {
                    return Err(ExecutorError::Failed(format!(
                        "request {} has no state on this stage",
                        activation.request_id
                    )))
                }
            };

            let process = engine.process.as_mut().expect("process started");
            let result = if first {
                let ids: Vec<u32> = activation
                    .payload
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                process.decode_tokens(session_id, position, &ids)
            } else {
                process.decode_hidden(session_id, position, tokens as u32, &activation.payload)
            };
            let output = match engine.settle(result) {
                Ok(output) => output,
                Err(err) => {
                    // the engine dropped the session with the failed request
                    engine.sessions.remove(&activation.request_id);
                    return Err(err);
                }
            };
            if let Some(session) = engine.sessions.get_mut(&activation.request_id) {
                session.next_position += tokens as u32;
            }

            let (values, shape) = match output {
                StageOutput::Hidden(values) => (values, vec![tokens as u32, hidden_size as u32]),
                StageOutput::Logits(values) => {
                    let width = values.len() as u32;
                    (values, vec![width])
                }
            };
            let payload: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
            Ok(ActivationTensor::new(
                activation.request_id,
                activation.token_position,
                range.start_layer,
                range.end_layer_exclusive,
                shape,
                ActivationDType::F32,
                payload,
            ))
        })
        .await
    }

    async fn decode_token_text(&self, token_id: u32) -> String {
        match self.with_running(move |process| process.token_text(&[token_id])).await {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(_) => format!("<token:{token_id}>"),
        }
    }

    async fn end_request(&self, request_id: &str) {
        let request_id = request_id.to_string();
        let _ = self
            .blocking(move |_, engine| {
                if let (Some(session), Some(process)) =
                    (engine.sessions.remove(&request_id), engine.process.as_mut())
                {
                    let result = process.end_session(session.id);
                    engine.settle(result)?;
                }
                Ok(())
            })
            .await;
    }

    /// Unwraps the logits the final stage returned from [`LayerExecutor::execute_range`].
    async fn final_logits(&self, activation: ActivationTensor) -> Result<Logits, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }
        if activation.dtype != ActivationDType::F32
            || activation.target_layer != self.metadata.layer_count
            || activation.shape.len() != 1
            || activation.payload.len() != activation.shape[0] as usize * 4
        {
            return Err(ExecutorError::Failed(
                "expected the logits activation produced by the final stage".into(),
            ));
        }
        let logits = activation
            .payload
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(Logits::new(activation.request_id, activation.token_position, logits))
    }
}
