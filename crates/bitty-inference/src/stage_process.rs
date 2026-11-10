//! Client side of the `bitty-stage` process: a patched llama.cpp serving one layer range over
//! stdin/stdout. The frame format is documented at the top of `examples/stage-server/stage-server.cpp` in `engine/llama-cpp-stage.patch`.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use thiserror::Error;

const PROTOCOL_VERSION: u32 = 1;
const MAX_PAYLOAD: u32 = 1 << 30;

const OP_DECODE_TOKENS: u32 = 1;
const OP_DECODE_HIDDEN: u32 = 2;
const OP_END: u32 = 3;
const OP_PING: u32 = 4;
const OP_TOKEN_TEXT: u32 = 5;
const OP_TOKENIZE: u32 = 6;
const OP_CHAT_PROMPT: u32 = 7;
const OP_EOG_TOKENS: u32 = 8;

const KIND_HIDDEN: u32 = 1;
const KIND_LOGITS: u32 = 2;
const KIND_TOP_LOGITS: u32 = 5;

#[derive(Debug, Error)]
pub enum StageProcessError {
    #[error("cannot start the stage engine {path}: {source}")]
    Spawn { path: PathBuf, source: io::Error },
    #[error("the stage engine exited or closed its pipe: {0}")]
    Io(#[from] io::Error),
    #[error("the stage engine sent an invalid handshake")]
    BadHandshake,
    #[error("the stage engine speaks protocol {0}, this build expects {PROTOCOL_VERSION}")]
    Version(u32),
    #[error("stage engine: {0}")]
    Engine(String),
    #[error("the stage engine sent an unexpected reply")]
    BadReply,
}

/// What the engine reported about the model and its range once the weights were loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInfo {
    pub layer_count: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub start_layer: u32,
    pub end_layer_exclusive: u32,
}

impl StageInfo {
    pub fn is_first(&self) -> bool {
        self.start_layer == 0
    }

    pub fn is_final(&self) -> bool {
        self.end_layer_exclusive == self.layer_count
    }
}

#[derive(Clone, Debug)]
pub struct StageProcessConfig {
    pub engine: PathBuf,
    pub model: PathBuf,
    pub start_layer: u32,
    pub end_layer_exclusive: u32,
    /// 0 lets the engine choose.
    pub threads: u32,
    /// llama.cpp's count: layers from the top of the whole model; 0 keeps everything on the CPU.
    pub gpu_layers: u32,
    /// Context length of one session, in tokens.
    pub context: u32,
    /// Sessions the stage keeps state for at the same time.
    pub sessions: u32,
    /// How many logits the final stage answers with, best first; 0 asks for the whole vocabulary.
    pub top_logits: u32,
}

/// Output of one decode call.
#[derive(Clone, Debug, PartialEq)]
pub enum StageOutput {
    /// `tokens * hidden_size` values: the residual stream after the last layer of the range.
    Hidden(Vec<f32>),
    /// `vocab_size` logits of the last token; only the final stage returns them.
    Logits(Vec<f32>),
    /// The best logits of the last token in the wire form of
    /// [`crate::sampling::encode_top_logits`]; the final stage returns them when it was started
    /// with a limit.
    TopLogits(Vec<u8>),
}

/// One running `bitty-stage` child. Calls block; the process dies with this value.
pub struct StageProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    info: StageInfo,
}

impl StageProcess {
    /// Starts the engine and waits until its weights are loaded.
    pub fn spawn(config: &StageProcessConfig) -> Result<Self, StageProcessError> {
        let mut command = Command::new(&config.engine);
        command
            .arg("-m")
            .arg(&config.model)
            .arg("--start")
            .arg(config.start_layer.to_string())
            .arg("--end")
            .arg(config.end_layer_exclusive.to_string())
            .arg("--ctx")
            .arg(config.context.to_string())
            .arg("--sessions")
            .arg(config.sessions.to_string());
        if config.gpu_layers > 0 {
            command.arg("--gpu-layers").arg(config.gpu_layers.to_string());
        }
        if config.threads > 0 {
            command.arg("-t").arg(config.threads.to_string());
        }
        if config.top_logits > 0 {
            command.arg("--top-logits").arg(config.top_logits.to_string());
        }
        Self::start(command, &config.engine)
    }

    /// Starts the engine on the vocabulary alone: no weights are loaded, and only the text calls
    /// (`tokenize`, `token_text`, `chat_prompt`, `eog_tokens`) work.
    pub fn spawn_vocab(engine: &Path, model: &Path) -> Result<Self, StageProcessError> {
        let mut command = Command::new(engine);
        command.arg("-m").arg(model).arg("--vocab-only");
        Self::start(command, engine)
    }

    fn start(mut command: Command, engine: &Path) -> Result<Self, StageProcessError> {
        let stderr = if std::env::var_os("BITTY_STAGE_LOG").is_some() {
            Stdio::inherit()
        } else {
            Stdio::null()
        };
        // An engine that waits for its turn must get out of the way. By default the OpenMP
        // threads of llama.cpp spin for milliseconds after every batch, and on a machine that
        // runs several stages they take the cores from the stage that is working. Sleeping at
        // once (OMP_WAIT_POLICY=passive) is no answer: the threads then also sleep at every
        // barrier inside a batch, which costs about 8 ms per token and stage. A short spin
        // keeps the barriers fast and frees the cores within microseconds.
        if std::env::var_os("OMP_WAIT_POLICY").is_none()
            && std::env::var_os("GOMP_SPINCOUNT").is_none()
        {
            command.env("GOMP_SPINCOUNT", "1000");
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
            .map_err(|source| StageProcessError::Spawn {
                path: engine.to_path_buf(),
                source,
            })?;
        let stdin = child.stdin.take().expect("piped stdin");
        let mut stdout = child.stdout.take().expect("piped stdout");

        let mut hello = [0u8; 28];
        if let Err(err) = stdout.read_exact(&mut hello) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(StageProcessError::Io(err));
        }
        let word = |index: usize| u32::from_le_bytes(hello[4 + index * 4..8 + index * 4].try_into().unwrap());
        if &hello[..4] != b"BSTG" {
            let _ = child.kill();
            let _ = child.wait();
            return Err(StageProcessError::BadHandshake);
        }
        if word(0) != PROTOCOL_VERSION {
            let _ = child.kill();
            let _ = child.wait();
            return Err(StageProcessError::Version(word(0)));
        }
        let info = StageInfo {
            layer_count: word(1),
            hidden_size: word(2),
            vocab_size: word(3),
            start_layer: word(4),
            end_layer_exclusive: word(5),
        };
        Ok(Self {
            child,
            stdin,
            stdout,
            info,
        })
    }

    pub fn info(&self) -> StageInfo {
        self.info
    }

    /// Runs token ids through the first stage. `position` is where they sit in the session;
    /// 0 opens a new session.
    pub fn decode_tokens(
        &mut self,
        session: u64,
        position: u32,
        tokens: &[u32],
    ) -> Result<StageOutput, StageProcessError> {
        let payload: Vec<u8> = tokens.iter().flat_map(|token| token.to_le_bytes()).collect();
        self.decode(OP_DECODE_TOKENS, session, position, tokens.len() as u32, &payload)
    }

    /// Runs hidden states (`tokens * hidden_size` values) through a later stage.
    pub fn decode_hidden(
        &mut self,
        session: u64,
        position: u32,
        tokens: u32,
        hidden: &[u8],
    ) -> Result<StageOutput, StageProcessError> {
        self.decode(OP_DECODE_HIDDEN, session, position, tokens, hidden)
    }

    fn decode(
        &mut self,
        op: u32,
        session: u64,
        position: u32,
        tokens: u32,
        payload: &[u8],
    ) -> Result<StageOutput, StageProcessError> {
        let (kind, _, bytes) = self.call(op, session, position, tokens, payload)?;
        if kind == KIND_TOP_LOGITS {
            return if bytes.len() % 8 == 0 {
                Ok(StageOutput::TopLogits(bytes))
            } else {
                Err(StageProcessError::BadReply)
            };
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        match kind {
            KIND_HIDDEN => Ok(StageOutput::Hidden(values)),
            KIND_LOGITS => Ok(StageOutput::Logits(values)),
            _ => Err(StageProcessError::BadReply),
        }
    }

    /// Drops the state of a session.
    pub fn end_session(&mut self, session: u64) -> Result<(), StageProcessError> {
        self.call(OP_END, session, 0, 0, &[]).map(|_| ())
    }

    pub fn ping(&mut self) -> Result<(), StageProcessError> {
        self.call(OP_PING, 0, 0, 0, &[]).map(|_| ())
    }

    pub fn token_text(&mut self, tokens: &[u32]) -> Result<Vec<u8>, StageProcessError> {
        let payload: Vec<u8> = tokens.iter().flat_map(|token| token.to_le_bytes()).collect();
        self.call(OP_TOKEN_TEXT, 0, 0, 0, &payload).map(|reply| reply.2)
    }

    pub fn tokenize(&mut self, text: &str, add_special: bool) -> Result<Vec<u32>, StageProcessError> {
        let (_, _, bytes) = self.call(OP_TOKENIZE, 0, 0, add_special as u32, text.as_bytes())?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Builds the prompt text for a conversation with the model's chat template, ending with an
    /// open assistant turn. `thinking` lets a reasoning model think before it answers.
    pub fn chat_prompt(
        &mut self,
        messages: &[(&str, &str)],
        thinking: bool,
    ) -> Result<String, StageProcessError> {
        let mut payload = Vec::new();
        for (role, content) in messages {
            for part in [role, content] {
                payload.extend_from_slice(&(part.len() as u32).to_le_bytes());
                payload.extend_from_slice(part.as_bytes());
            }
        }
        let (_, _, bytes) = self.call(OP_CHAT_PROMPT, 0, thinking as u32, 1, &payload)?;
        String::from_utf8(bytes).map_err(|_| StageProcessError::BadReply)
    }

    /// Every token id that ends generation (end of turn, end of text).
    pub fn eog_tokens(&mut self) -> Result<Vec<u32>, StageProcessError> {
        let (_, _, bytes) = self.call(OP_EOG_TOKENS, 0, 0, 0, &[])?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn call(
        &mut self,
        op: u32,
        session: u64,
        position: u32,
        tokens: u32,
        payload: &[u8],
    ) -> Result<(u32, u32, Vec<u8>), StageProcessError> {
        let mut head = [0u8; 24];
        head[0..4].copy_from_slice(&op.to_le_bytes());
        head[4..12].copy_from_slice(&session.to_le_bytes());
        head[12..16].copy_from_slice(&position.to_le_bytes());
        head[16..20].copy_from_slice(&tokens.to_le_bytes());
        head[20..24].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        self.stdin.write_all(&head)?;
        self.stdin.write_all(payload)?;
        self.stdin.flush()?;

        let mut reply = [0u8; 16];
        self.stdout.read_exact(&mut reply)?;
        let word = |index: usize| u32::from_le_bytes(reply[index * 4..index * 4 + 4].try_into().unwrap());
        let (status, kind, rows, len) = (word(0), word(1), word(2), word(3));
        if len > MAX_PAYLOAD {
            return Err(StageProcessError::BadReply);
        }
        let mut bytes = vec![0u8; len as usize];
        self.stdout.read_exact(&mut bytes)?;
        if status != 0 {
            return Err(StageProcessError::Engine(String::from_utf8_lossy(&bytes).into_owned()));
        }
        Ok((kind, rows, bytes))
    }
}

impl Drop for StageProcess {
    fn drop(&mut self) {
        // closing stdin makes the engine exit on its own; kill covers a stuck one
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Finds the `bitty-stage` binary: `BITTY_STAGE_BIN`, then next to this executable, then `PATH`.
pub fn find_stage_engine() -> Option<PathBuf> {
    // "bitty-stage.exe" on Windows
    let name = format!("bitty-stage{}", std::env::consts::EXE_SUFFIX);
    if let Some(path) = std::env::var_os("BITTY_STAGE_BIN") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(candidate) = exe.parent().map(|dir| dir.join(&name)) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| Path::new(&dir).join(&name))
        .find(|candidate| candidate.is_file())
}
