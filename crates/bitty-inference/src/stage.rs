//! Executor for one slice ("stage") of the model: a contiguous range of transformer blocks.
//!
//! The compute engine is a patched llama.cpp running as a child process (`bitty-stage`, see
//! `engine/`). It loads only the tensors of its range, takes token ids (first stage) or hidden
//! states (later stages) and returns hidden states, or logits on the final stage. Without the
//! engine binary every compute call fails with [`ExecutorError::Failed`] instead of returning
//! made-up activations.

use crate::gpu::{
    find_gpu_engine, gpu_is_faster, gpu_lock, gpu_mode, layers_that_fit, probe_gpu_engine,
    time_engine, GpuMode,
};
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
    /// Layers of the served range to keep on the GPU, counted from the top of the range.
    pub gpu_layers: u32,
    /// Keeps the output layer on the GPU even when no block of the range is there. It is one
    /// large matrix, about a tenth of the work of a token, and needs little video memory.
    pub gpu_output: bool,
    /// Lets bitty choose: when the engine for graphics cards finds a card, as many layers as
    /// fit in its free memory go there, and `gpu_layers` and `gpu_output` are not read.
    pub gpu_auto: bool,
    /// Tokens of its last request a stage can take back: the longest guess of a draft model
    /// that can be checked in one pass. Costs memory (about 25 MB each for Qwen3.5-35B).
    pub rollback: u32,
}

impl StageOptions {
    /// Defaults overridden by `BITTY_STAGE_THREADS`, `BITTY_GPU`, `BITTY_GPU_LAYERS`,
    /// `BITTY_GPU_OUTPUT` and `BITTY_STAGE_ROLLBACK`. Without `BITTY_GPU_LAYERS` the layers
    /// for the graphics card are chosen by bitty; `BITTY_GPU=off` keeps everything on the
    /// processor whatever the others say.
    pub fn from_env() -> Self {
        let read = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
        };
        let mut options = Self::default();
        // One thread for each core: the engine alone picks 4 on any machine, which left half
        // of an 8-core leader idle (13% slower in the ring). The second threads of the cores
        // are not counted, they make it slower.
        options.threads = read("BITTY_STAGE_THREADS")
            .or_else(bitty_model::physical_cores)
            .unwrap_or(0);
        if gpu_mode() != GpuMode::Off {
            match read("BITTY_GPU_LAYERS") {
                Some(gpu_layers) => options.gpu_layers = gpu_layers,
                None => options.gpu_auto = true,
            }
            if let Some(gpu_output) = read("BITTY_GPU_OUTPUT") {
                options.gpu_output = gpu_output > 0;
            }
        }
        if let Some(rollback) = read("BITTY_STAGE_ROLLBACK") {
            options.rollback = rollback;
        }
        options
    }
}

/// Translates "layers of this range on the GPU" into llama.cpp's count, which starts from the
/// top of the whole model and counts the output layer first.
fn engine_gpu_layers(
    gpu_layers: u32,
    gpu_output: bool,
    start: u32,
    end: u32,
    layer_count: u32,
) -> u32 {
    let wanted = gpu_layers.min(end.saturating_sub(start));
    if wanted == 0 {
        // the output layer is the first one llama.cpp counts
        return u32::from(gpu_output && end == layer_count);
    }
    layer_count.saturating_sub(end) + wanted + 1
}

impl Default for StageOptions {
    fn default() -> Self {
        Self {
            threads: 0,
            context: 8192,
            sessions: 4,
            gpu_layers: 0,
            gpu_output: false,
            gpu_auto: false,
            rollback: DEFAULT_ROLLBACK,
        }
    }
}

/// Tokens a stage can take back unless `BITTY_STAGE_ROLLBACK` says otherwise.
pub const DEFAULT_ROLLBACK: u32 = 4;

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

/// Asked while a stage file is being fetched; `true` means the range is no longer wanted.
pub type CancelCheck = Arc<dyn Fn() -> bool + Send + Sync>;

/// Returns the model file to open for layers `[start, end)`, fetching it first if needed.
/// The fetch gives up when the check answers `true`.
pub type StageSource = Arc<dyn Fn(u32, u32, CancelCheck) -> Result<PathBuf, String> + Send + Sync>;

fn pack_range(range: &AssignedLayerRange) -> u64 {
    ((range.start_layer as u64) << 32) | range.end_layer_exclusive as u64
}

#[derive(Clone)]
pub struct StageExecutor {
    metadata: Arc<ModelMetadata>,
    model_path: Option<PathBuf>,
    engine_path: Option<PathBuf>,
    /// The engine was named by the caller: it is never exchanged for the one for graphics cards.
    engine_fixed: bool,
    options: StageOptions,
    stage_source: Option<StageSource>,
    /// The range asked for last. A fetch for any other range is stale and stops, so a node
    /// that was given new layers does not finish downloading the old ones first.
    wanted: Arc<std::sync::atomic::AtomicU64>,
    engine: Arc<Mutex<Engine>>,
    /// Stops a guess of the running engine; reachable while `engine` is locked by the
    /// thread that waits for the guess.
    abort: Arc<Mutex<Option<crate::stage_process::StageAbort>>>,
}

/// The name a guess has for the engine: one request never has two with the same position.
fn guess_tag(request_id: &str, token_position: u32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (request_id, token_position).hash(&mut hasher);
    hasher.finish()
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
            engine_fixed: false,
            options: StageOptions::from_env(),
            stage_source: None,
            wanted: Arc::default(),
            abort: Arc::default(),
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
        // said once, so the log of a node tells which build of the engine its processor got
        if let Some(name) = executor
            .engine_path
            .as_ref()
            .and_then(|path| path.file_name())
        {
            eprintln!("bitty: engine {}", name.to_string_lossy());
        }
        Ok(executor)
    }

    pub fn with_engine(mut self, engine_path: impl Into<PathBuf>) -> Self {
        self.engine_path = Some(engine_path.into());
        self.engine_fixed = true;
        self
    }

    /// Opens a file chosen per layer range, for a node that holds stage files and not the whole
    /// model. The file given to [`StageExecutor::load`] then only supplies the metadata.
    pub fn with_stage_source(mut self, source: StageSource) -> Self {
        self.stage_source = Some(source);
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
    fn ensure_process(
        &self,
        engine: &mut Engine,
        range: &AssignedLayerRange,
    ) -> Result<(), ExecutorError> {
        let matches = engine.process.as_ref().is_some_and(|process| {
            let info = process.info();
            info.start_layer == range.start_layer
                && info.end_layer_exclusive == range.end_layer_exclusive
        });
        if matches {
            return Ok(());
        }
        let (Some(model), Some(engine_path)) = (&self.model_path, &self.engine_path) else {
            return Err(ExecutorError::Failed(ENGINE_MISSING.into()));
        };
        engine.process = None;
        engine.sessions.clear();
        let fetched;
        let model = match &self.stage_source {
            Some(source) => {
                let wanted = self.wanted.clone();
                let mine = pack_range(range);
                let stale: CancelCheck =
                    Arc::new(move || wanted.load(std::sync::atomic::Ordering::Relaxed) != mine);
                fetched =
                    source(range.start_layer, range.end_layer_exclusive, stale).map_err(|err| {
                        ExecutorError::Failed(format!("stage file is not available: {err}"))
                    })?;
                &fetched
            }
            None => model,
        };
        let process = self
            .start_engine(engine_path, model, range)
            .map_err(|err| ExecutorError::Failed(err.to_string()))?;
        *self.abort.lock().unwrap_or_else(|lock| lock.into_inner()) = Some(process.abort_handle());
        engine.process = Some(process);
        Ok(())
    }

    /// Starts the engine for `range` on `model`, the file it opens: on a graphics card when
    /// that was asked for, or when bitty may choose and the card proves faster than the
    /// processor.
    fn start_engine(
        &self,
        cpu_engine: &Path,
        model: &Path,
        range: &AssignedLayerRange,
    ) -> Result<StageProcess, StageProcessError> {
        let (start, end) = (range.start_layer, range.end_layer_exclusive);
        let layer_count = self.metadata.layer_count;
        let spawn = |engine: &Path, gpu_layers: u32| {
            StageProcess::spawn(&StageProcessConfig {
                engine: engine.to_path_buf(),
                model: model.to_path_buf(),
                start_layer: start,
                end_layer_exclusive: end,
                threads: self.options.threads,
                gpu_layers,
                context: self.options.context,
                sessions: self.options.sessions,
                top_logits: crate::sampling::top_logits_limit() as u32,
                rollback: self.options.rollback,
            })
        };
        if !self.options.gpu_auto {
            let layers = engine_gpu_layers(
                self.options.gpu_layers,
                self.options.gpu_output,
                start,
                end,
                layer_count,
            );
            // Asked for by hand. Without an engine for cards that works here the usual one
            // gets the count as before: it may be a build of one's own.
            let gpu_engine = (layers > 0 && !self.engine_fixed)
                .then(probe_gpu_engine)
                .flatten()
                .map(|gpu| gpu.path);
            return spawn(gpu_engine.as_deref().unwrap_or(cpu_engine), layers);
        }
        if self.engine_fixed || find_gpu_engine().is_none() {
            return spawn(cpu_engine, 0);
        }
        // a verdict of this process for the same range: the card does not change under it
        let verdict_key = (start, end);
        let known = GPU_VERDICTS
            .lock()
            .unwrap_or_else(|lock| lock.into_inner())
            .get(&verdict_key)
            .copied();
        if known == Some(false) {
            return spawn(cpu_engine, 0);
        }
        // from reading the free memory to the end of the load, one node of a machine at a time
        let _lock = gpu_lock();
        let Some((gpu_engine, candidates)) = self.gpu_candidates(model, range) else {
            return spawn(cpu_engine, 0);
        };
        let check = known.is_none()
            && std::env::var("BITTY_GPU_CHECK").map_or(true, |value| value.trim() != "0");
        // The processor first: its time is the one to beat. Its engine goes away before the
        // card is loaded, a machine has no room for the layers twice.
        let cpu_time = if check {
            let mut process = spawn(cpu_engine, 0)?;
            time_engine(&mut process).ok()
        } else {
            None
        };
        for (layers, what) in candidates {
            let loaded = spawn(&gpu_engine, layers).and_then(|mut process| {
                let time = if check {
                    Some(time_engine(&mut process)?)
                } else {
                    None
                };
                Ok((process, time))
            });
            let (process, gpu_time) = match loaded {
                Ok(loaded) => loaded,
                Err(err) => {
                    // the free memory may have gone to another program, or the driver fails
                    eprintln!("bitty: {what} could not be loaded on the graphics card ({err})");
                    continue;
                }
            };
            let faster = match (cpu_time, gpu_time) {
                (Some(cpu), Some(gpu)) => {
                    let faster = gpu_is_faster(cpu, gpu);
                    eprintln!(
                        "bitty: {what} on the graphics card: {:.1} ms a token, {:.1} on the processor alone: {}",
                        gpu.as_secs_f64() * 1000.0,
                        cpu.as_secs_f64() * 1000.0,
                        if faster { "the card is used" } else { "the card is not used" }
                    );
                    faster
                }
                _ => {
                    eprintln!("bitty: {what} on the graphics card");
                    true
                }
            };
            GPU_VERDICTS
                .lock()
                .unwrap_or_else(|lock| lock.into_inner())
                .insert(verdict_key, faster);
            if faster {
                return Ok(process);
            }
            drop(process);
            return spawn(cpu_engine, 0);
        }
        eprintln!("bitty: layers {start}..{end} are computed on the processor");
        spawn(cpu_engine, 0)
    }

    /// What of `range` to try on a graphics card: the engine for it, and llama.cpp's count of
    /// layers for each attempt with a description for the log, the largest first. `None`
    /// when there is no card to use or nothing fits. `model` is the file the engine will open.
    fn gpu_candidates(
        &self,
        model: &Path,
        range: &AssignedLayerRange,
    ) -> Option<(PathBuf, Vec<(u32, String)>)> {
        let (start, end) = (range.start_layer, range.end_layer_exclusive);
        let layer_count = self.metadata.layer_count;
        let gpu = probe_gpu_engine().filter(|gpu| gpu.usable(gpu_mode()))?;
        // A node that fetches its stages was opened on the vocabulary file, which names no
        // weights: the sizes are in the stage file itself.
        let stage_file;
        let tensors = if self.model_path.as_deref() == Some(model) {
            &self.metadata.tensors
        } else {
            stage_file = ModelMetadata::from_gguf_path(model)
                .map(|metadata| metadata.tensors)
                .unwrap_or_default();
            &stage_file
        };
        // The sizes of the weights are known, what the engine needs on top of them is not:
        // if the load fails with all that should fit, half of it is tried.
        let budget = gpu.weight_budget_bytes();
        let mut candidates: Vec<(u32, String)> = Vec::new();
        for budget in [budget, budget / 2] {
            let fit = layers_that_fit(tensors, start, end, layer_count, budget);
            let layers = engine_gpu_layers(fit.layers, fit.output, start, end, layer_count);
            if layers == 0 || candidates.iter().any(|(tried, _)| *tried == layers) {
                continue;
            }
            candidates.push((
                layers,
                format!(
                    "{} of the {} layers {start}..{end}{} ({} MB of weights)",
                    fit.layers,
                    end.saturating_sub(start),
                    if fit.output {
                        " and the output layer"
                    } else {
                        ""
                    },
                    fit.bytes / 1024 / 1024
                ),
            ));
        }
        if candidates.is_empty() {
            eprintln!(
                "bitty: graphics card found ({}), but no layer of {start}..{end} fits in its free memory",
                gpu.summary()
            );
            return None;
        }
        eprintln!("bitty: graphics card: {}", gpu.summary());
        Some((gpu.path, candidates))
    }
}

/// Whether the card was faster than the processor, for each range this process has timed.
static GPU_VERDICTS: Mutex<std::collections::BTreeMap<(u32, u32), bool>> =
    Mutex::new(std::collections::BTreeMap::new());

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
    async fn prepare_range(&self, range: &AssignedLayerRange) -> Result<(), ExecutorError> {
        let range = range.clone();
        self.wanted
            .store(pack_range(&range), std::sync::atomic::Ordering::Relaxed);
        self.blocking(move |this, engine| this.ensure_process(engine, &range))
            .await
    }

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
        if range.end_layer_exclusive > layer_count || range.start_layer >= range.end_layer_exclusive
        {
            return Err(ExecutorError::Failed(format!(
                "layer range {}..{} is outside the model's {} blocks",
                range.start_layer, range.end_layer_exclusive, layer_count
            )));
        }
        let range = range.clone();
        self.wanted
            .store(pack_range(&range), std::sync::atomic::Ordering::Relaxed);
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
                    return Err(ExecutorError::Failed(
                        "the first stage expects u32 token ids".into(),
                    ));
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
                    let result = engine
                        .process
                        .as_mut()
                        .expect("process started")
                        .end_session(old.id);
                    engine.settle(result)?;
                }
                engine.next_session_id += 1;
                let id = engine.next_session_id;
                engine.sessions.insert(
                    activation.request_id.clone(),
                    Session {
                        id,
                        next_position: 0,
                    },
                );
            }
            let (session_id, position) = match engine.sessions.get(&activation.request_id) {
                // a guess that was wrong is taken back: the engine goes back by itself when
                // a request starts before the end of the session
                Some(session)
                    if activation.rewind_to != 0
                        && activation.rewind_to <= session.next_position =>
                {
                    (session.id, activation.rewind_to)
                }
                Some(session) if activation.rewind_to != 0 => {
                    return Err(ExecutorError::Failed(format!(
                        "request {} is at position {} and was asked to go back to {}",
                        activation.request_id, session.next_position, activation.rewind_to
                    )))
                }
                Some(session) => (session.id, session.next_position),
                None => {
                    return Err(ExecutorError::Failed(format!(
                        "request {} has no state on this stage",
                        activation.request_id
                    )))
                }
            };
            let every_row = activation.every_row;

            let process = engine.process.as_mut().expect("process started");
            let tag = guess_tag(&activation.request_id, activation.token_position);
            let result = if first {
                let ids: Vec<u32> = activation
                    .payload
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                if activation.guess {
                    process.guess_tokens(session_id, position, &ids, tag)
                } else {
                    process
                        .decode_tokens(session_id, position, &ids, every_row)
                        .map(Some)
                }
            } else if activation.guess {
                process.guess_hidden(
                    session_id,
                    position,
                    tokens as u32,
                    &activation.payload,
                    tag,
                )
            } else {
                process
                    .decode_hidden(
                        session_id,
                        position,
                        tokens as u32,
                        &activation.payload,
                        every_row,
                    )
                    .map(Some)
            };
            let output = match engine.settle(result) {
                Ok(Some(output)) => output,
                // the engine put the session back where the guess started
                Ok(None) => return Err(ExecutorError::Stopped),
                Err(err) => {
                    // the engine dropped the session with the failed request
                    engine.sessions.remove(&activation.request_id);
                    return Err(err);
                }
            };
            if let Some(session) = engine.sessions.get_mut(&activation.request_id) {
                session.next_position = position + tokens as u32;
            }

            let (values, shape) = match output {
                StageOutput::TopLogits(pairs) => {
                    return Ok(ActivationTensor::new(
                        activation.request_id,
                        activation.token_position,
                        range.start_layer,
                        range.end_layer_exclusive,
                        vec![(pairs.len() / 8) as u32],
                        ActivationDType::TopLogits,
                        pairs,
                    ));
                }
                StageOutput::TopLogitRows(rows) => {
                    return Ok(ActivationTensor::new(
                        activation.request_id,
                        activation.token_position,
                        range.start_layer,
                        range.end_layer_exclusive,
                        vec![tokens as u32],
                        ActivationDType::TopLogitRows,
                        rows,
                    ));
                }
                StageOutput::Hidden(values) => (values, vec![tokens as u32, hidden_size as u32]),
                StageOutput::Logits(_) if every_row => {
                    return Err(ExecutorError::Failed(
                        "the logits of every token need a limit on them (BITTY_TOP_LOGITS is 0)"
                            .into(),
                    ))
                }
                StageOutput::Logits(values) => {
                    let width = values.len() as u32;
                    (values, vec![width])
                }
            };
            let (rewind_to, every_row, guess) =
                (activation.rewind_to, activation.every_row, activation.guess);
            let mut payload = Vec::with_capacity(values.len() * 4);
            for value in &values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let mut output = ActivationTensor::new(
                activation.request_id,
                activation.token_position,
                range.start_layer,
                range.end_layer_exclusive,
                shape,
                ActivationDType::F32,
                payload,
            );
            // the stages that follow must go back and answer in the same way
            output.rewind_to = rewind_to;
            output.every_row = every_row;
            output.guess = guess;
            Ok(output)
        })
        .await
    }

    async fn decode_token_text(&self, token_id: u32) -> String {
        match self
            .with_running(move |process| process.token_text(&[token_id]))
            .await
        {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(_) => format!("<token:{token_id}>"),
        }
    }

    async fn stop_guess(&self, request_id: &str, token_position: u32) {
        let abort = self
            .abort
            .lock()
            .unwrap_or_else(|lock| lock.into_inner())
            .clone();
        if let Some(abort) = abort {
            abort.stop(guess_tag(request_id, token_position));
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
        if activation.dtype == ActivationDType::TopLogits
            && activation.target_layer == self.metadata.layer_count
        {
            // the logits the stage left out can never be sampled: they stand as minus infinity
            let vocabulary = self
                .blocking(|_, engine| {
                    engine
                        .process
                        .as_ref()
                        .map(|process| process.info().vocab_size as usize)
                        .ok_or_else(|| ExecutorError::Failed("no stage is loaded yet".into()))
                })
                .await?;
            let mut logits = vec![f32::NEG_INFINITY; vocabulary];
            for (token, logit) in
                crate::sampling::decode_top_logits(&activation.payload).unwrap_or_default()
            {
                if let Some(slot) = logits.get_mut(token as usize) {
                    *slot = logit;
                }
            }
            return Ok(Logits::new(
                activation.request_id,
                activation.token_position,
                logits,
            ));
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
        Ok(Logits::new(
            activation.request_id,
            activation.token_position,
            logits,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::engine_gpu_layers;

    #[test]
    fn gpu_layers_count_from_the_top_of_the_model() {
        // nothing asked, nothing offloaded
        assert_eq!(engine_gpu_layers(0, false, 10, 20, 40), 0);
        // the whole middle range: everything from layer 10 up
        assert_eq!(engine_gpu_layers(10, false, 10, 20, 40), 31);
        assert_eq!(engine_gpu_layers(99, false, 10, 20, 40), 31);
        // the top four layers of the range
        assert_eq!(engine_gpu_layers(4, false, 10, 20, 40), 25);
        // the final range: its layers and the output layer
        assert_eq!(engine_gpu_layers(10, false, 30, 40, 40), 11);
        assert_eq!(engine_gpu_layers(3, false, 30, 40, 40), 4);
        // the output layer alone, only where the range holds it
        assert_eq!(engine_gpu_layers(0, true, 30, 40, 40), 1);
        assert_eq!(engine_gpu_layers(0, true, 10, 20, 40), 0);
    }
}
