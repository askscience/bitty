//! A small model that guesses the next tokens of a request, so the workers can check several
//! tokens in one pass (speculative decoding).
//!
//! The draft model runs whole, in one `bitty-stage` process, on the machine the request is
//! sampled on. It must have the vocabulary of the model it guesses for. A guess never changes
//! the text: the large model has the last word on every token.

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
