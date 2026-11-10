//! Text on the requester's side: prompt to tokens, tokens back to text.
//!
//! This needs only the vocabulary of the model, so it runs on the machine that asks the question,
//! with a `bitty-stage --vocab-only` process that loads no weights.

use crate::stage_process::{find_stage_engine, StageProcess, StageProcessError};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

pub struct TextCodec {
    process: Mutex<StageProcess>,
    end_tokens: HashSet<u32>,
    vocab_size: u32,
}

impl std::fmt::Debug for TextCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextCodec")
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

impl TextCodec {
    /// Opens the vocabulary of `model` (the whole GGUF or any stage file of it).
    pub fn open(model: &Path) -> Result<Self, StageProcessError> {
        let engine = find_stage_engine().ok_or_else(|| StageProcessError::Spawn {
            path: "bitty-stage".into(),
            source: std::io::ErrorKind::NotFound.into(),
        })?;
        Self::open_with_engine(&engine, model)
    }

    pub fn open_with_engine(engine: &Path, model: &Path) -> Result<Self, StageProcessError> {
        let mut process = StageProcess::spawn_vocab(engine, model)?;
        let end_tokens = process.eog_tokens()?.into_iter().collect();
        let vocab_size = process.info().vocab_size;
        Ok(Self {
            process: Mutex::new(process),
            end_tokens,
            vocab_size,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    fn process(&self) -> std::sync::MutexGuard<'_, StageProcess> {
        self.process
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Tokens of a raw prompt, with the prefix the model expects.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, StageProcessError> {
        self.process().tokenize(text, true)
    }

    /// Tokens of a conversation formatted with the model's chat template, ready for the
    /// assistant to answer.
    pub fn chat_tokens(
        &self,
        messages: &[(&str, &str)],
        thinking: bool,
    ) -> Result<Vec<u32>, StageProcessError> {
        let mut process = self.process();
        let prompt = process.chat_prompt(messages, thinking)?;
        process.tokenize(&prompt, true)
    }

    /// Whether the token ends the answer.
    pub fn is_end(&self, token: u32) -> bool {
        self.end_tokens.contains(&token)
    }

    /// Bytes of one token. A character can be split over several tokens: feed the pieces to a
    /// [`TextStream`].
    pub fn piece(&self, token: u32) -> Result<Vec<u8>, StageProcessError> {
        self.process().token_text(&[token])
    }
}

/// Turns token pieces into text, holding back the bytes of a character that is not complete yet.
#[derive(Debug, Default)]
pub struct TextStream {
    pending: Vec<u8>,
}

impl TextStream {
    pub fn push(&mut self, piece: &[u8]) -> String {
        self.pending.extend_from_slice(piece);
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    text.push_str(valid);
                    self.pending.clear();
                    return text;
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    text.push_str(
                        std::str::from_utf8(&self.pending[..valid]).expect("checked prefix"),
                    );
                    match err.error_len() {
                        // bytes that can never be valid: replace them and go on
                        Some(bad) => {
                            text.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                        // the start of a character whose remaining bytes come with the next token
                        None => {
                            self.pending.drain(..valid);
                            return text;
                        }
                    }
                }
            }
        }
    }

    /// Text of whatever was left incomplete when generation stopped.
    pub fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_joins_a_character_split_over_tokens() {
        let mut stream = TextStream::default();
        let bytes = "aè€".as_bytes();
        assert_eq!(stream.push(&bytes[..2]), "a");
        assert_eq!(stream.push(&bytes[2..4]), "è");
        assert_eq!(stream.push(&bytes[4..5]), "");
        assert_eq!(stream.push(&bytes[5..]), "€");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn stream_replaces_bytes_that_are_not_text() {
        let mut stream = TextStream::default();
        assert_eq!(stream.push(&[b'a', 0xff, b'b']), "a\u{fffd}b");
        assert_eq!(stream.push(&[0xe2, 0x82]), "");
        assert_eq!(stream.finish(), "\u{fffd}");
    }
}
