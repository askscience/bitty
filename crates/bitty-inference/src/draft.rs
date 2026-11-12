//! A small model that guesses the next tokens of a request, so the workers can check several
//! tokens in one pass (speculative decoding).
//!
//! The draft model runs whole, in one `bitty-stage` process, on the machine the request is
//! sampled on. It must have the vocabulary of the model it guesses for. A guess never changes
//! the text: the large model has the last word on every token.

use crate::gpu::{gpu_is_faster, gpu_lock, gpu_mode, probe_gpu_engine, time_engine, GpuMode};
use crate::stage_process::{StageOutput, StageProcess, StageProcessConfig, StageProcessError};
use bitty_model::ModelMetadata;
use std::collections::HashMap;
use std::path::Path;

/// Requests the draft keeps state for at the same time.
const SESSIONS: u32 = 4;

#[derive(Default)]
struct DraftSession {
    id: u64,
    /// Tokens of the text the engine holds.
    held: usize,
    /// Guessed tokens the engine holds after them. They stay if the guesses were right.
    beyond: usize,
}

pub struct DraftModel {
    process: StageProcess,
    context: usize,
    sessions: HashMap<String, DraftSession>,
    next_session_id: u64,
}

impl std::fmt::Debug for DraftModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftModel")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

impl DraftModel {
    /// Loads the whole model. `threads` and `gpu_layers` of 0 leave the choice to the engine
    /// and keep the model on the CPU.
    pub fn open(
        engine: &Path,
        model: &Path,
        threads: u32,
        gpu_layers: u32,
    ) -> Result<Self, StageProcessError> {
        let layers = ModelMetadata::from_gguf_path(model)
            .map_err(|err| StageProcessError::Engine(format!("{}: {err}", model.display())))?
            .layer_count;
        let context = 8192;
        let process = StageProcess::spawn(&StageProcessConfig {
            engine: engine.to_path_buf(),
            model: model.to_path_buf(),
            start_layer: 0,
            end_layer_exclusive: layers,
            threads,
            gpu_layers,
            context,
            sessions: SESSIONS,
            top_logits: 1,
            rollback: 0,
        })?;
        Ok(Self {
            process,
            context: context as usize,
            sessions: HashMap::new(),
            next_session_id: 0,
        })
    }

    /// Loads the whole model where it guesses fastest. With `gpu_layers` the card gets that
    /// many layers and nothing is compared. Without, and unless `BITTY_GPU=off`, the model
    /// is timed on the processor and, when it fits in the free video memory, on the card:
    /// the faster one stays. A card that fails leaves the processor.
    pub fn open_placed(
        engine: &Path,
        model: &Path,
        threads: u32,
        gpu_layers: Option<u32>,
    ) -> Result<Self, StageProcessError> {
        let mode = gpu_mode();
        if mode == GpuMode::Off {
            return Self::open(engine, model, threads, 0);
        }
        if let Some(gpu_layers) = gpu_layers {
            let gpu_engine = (gpu_layers > 0)
                .then(probe_gpu_engine)
                .flatten()
                .map(|gpu| gpu.path);
            return Self::open(
                gpu_engine.as_deref().unwrap_or(engine),
                model,
                threads,
                gpu_layers,
            );
        }
        let mut on_cpu = Self::open(engine, model, threads, 0)?;
        let _lock = gpu_lock();
        let size = std::fs::metadata(model).map_or(u64::MAX, |file| file.len());
        let Some(gpu) =
            probe_gpu_engine().filter(|gpu| gpu.usable(mode) && size <= gpu.weight_budget_bytes())
        else {
            return Ok(on_cpu);
        };
        let Ok(cpu_time) = time_engine(&mut on_cpu.process) else {
            return Ok(on_cpu);
        };
        // more layers than any model has: all of it
        let on_gpu = Self::open(&gpu.path, model, threads, 999).and_then(|mut draft| {
            let time = time_engine(&mut draft.process)?;
            Ok((draft, time))
        });
        match on_gpu {
            Ok((draft, gpu_time)) if gpu_is_faster(cpu_time, gpu_time) => {
                eprintln!(
                    "bitty: draft model on the graphics card: {:.1} ms a token, {:.1} on the processor",
                    gpu_time.as_secs_f64() * 1000.0,
                    cpu_time.as_secs_f64() * 1000.0
                );
                Ok(draft)
            }
            Ok(_) => Ok(on_cpu),
            Err(err) => {
                eprintln!(
                    "bitty: the draft model could not be loaded on the graphics card ({err})"
                );
                Ok(on_cpu)
            }
        }
    }

    pub fn vocab_size(&self) -> u32 {
        self.process.info().vocab_size
    }

    /// Guesses up to `count` tokens that follow `text` (the prompt and every token chosen so
    /// far). Stops after a token for which `is_end` holds. Call [`DraftModel::settle`] with
    /// the number of guesses that were right before the next call for the request.
    pub fn propose(
        &mut self,
        request: &str,
        text: &[u32],
        count: usize,
        is_end: impl Fn(u32) -> bool,
    ) -> Result<Vec<u32>, StageProcessError> {
        if count == 0 || text.is_empty() || text.len() + count >= self.context {
            return Ok(Vec::new());
        }
        if !self.sessions.contains_key(request) {
            if self.sessions.len() >= SESSIONS as usize {
                // the engine has dropped the oldest state by itself; start all over
                self.sessions.clear();
            }
            self.next_session_id += 1;
            let session = DraftSession {
                id: self.next_session_id,
                ..DraftSession::default()
            };
            self.sessions.insert(request.to_string(), session);
        }
        let result = self.guess(request, text, count, is_end);
        if result.is_err() {
            self.sessions.remove(request);
        }
        result
    }

    fn guess(
        &mut self,
        request: &str,
        text: &[u32],
        count: usize,
        is_end: impl Fn(u32) -> bool,
    ) -> Result<Vec<u32>, StageProcessError> {
        let session = self.sessions.get_mut(request).expect("session was opened");
        if session.held >= text.len() {
            session.held = 0;
        }
        session.beyond = 0;
        // catch up with the text; a start before the end of what the engine holds takes the
        // engine back to the mark below
        let output = self.process.decode_tokens(
            session.id,
            session.held as u32,
            &text[session.held..],
            false,
        )?;
        session.held = text.len();
        self.process.mark(session.id)?;
        let mut guesses = vec![best_token(output)?];
        while guesses.len() < count && !is_end(*guesses.last().expect("not empty")) {
            let position = (session.held + guesses.len() - 1) as u32;
            let last = *guesses.last().expect("not empty");
            let output = self
                .process
                .decode_tokens(session.id, position, &[last], false)?;
            session.beyond = guesses.len();
            guesses.push(best_token(output)?);
        }
        Ok(guesses)
    }

    /// Tells the draft how many of its last guesses for `request` became part of the text.
    pub fn settle(&mut self, request: &str, accepted: usize) {
        if let Some(session) = self.sessions.get_mut(request) {
            if accepted >= session.beyond {
                session.held += session.beyond;
            }
            session.beyond = 0;
        }
    }

    /// Drops the state of a request.
    pub fn end(&mut self, request: &str) {
        if let Some(session) = self.sessions.remove(request) {
            let _ = self.process.end_session(session.id);
        }
    }
}

fn best_token(output: StageOutput) -> Result<u32, StageProcessError> {
    match output {
        StageOutput::TopLogits(pairs) if pairs.len() >= 8 => Ok(u32::from_le_bytes(
            pairs[..4].try_into().expect("four bytes"),
        )),
        _ => Err(StageProcessError::BadReply),
    }
}
