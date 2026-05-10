use std::collections::BTreeMap;
use std::path::Path;
use minijinja::{Environment, ErrorKind};

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Clone, Default)]
pub struct GgufTokenizerOverrides {
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub add_bos_token: Option<bool>,
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: u32,
    eos_id: u32,
    eot_id: Option<u32>,
    im_end_id: Option<u32>,
    token_strings: Vec<String>,
    chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_gguf_path(model_path: &Path, hf_model_id: Option<&str>) -> Result<Self> {
        Self::from_gguf_path_with_overrides(model_path, hf_model_id, GgufTokenizerOverrides::default())
    }

    pub fn from_gguf_path_with_overrides(
        model_path: &Path,
        hf_model_id: Option<&str>,
        overrides: GgufTokenizerOverrides,
    ) -> Result<Self> {
        let dir = model_path
            .parent()
            .unwrap_or_else(|| Path::new("."));

        let json_path = dir.join("tokenizer.json");
        if json_path.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(&json_path)
                .map_err(|e| TokenizerError::Tokenizer(format!("Failed to load tokenizer.json: {e}")))?;
            let chat_template = load_chat_template(dir, hf_model_id);
            return Self::from_hf_tokenizer_with_overrides(tokenizer, overrides, chat_template);
        }

        if let Some(model_id) = hf_model_id {
            let tokenizer = tokenizers::Tokenizer::from_pretrained(model_id, None)
                .map_err(|e| TokenizerError::Tokenizer(format!("Failed to load tokenizer from HF: {e}")))?;
            let _ = tokenizer.save(&json_path, false);
            let chat_template = load_chat_template(dir, hf_model_id);
            return Self::from_hf_tokenizer_with_overrides(tokenizer, overrides, chat_template);
        }

        Err(TokenizerError::Tokenizer(format!(
            "No tokenizer.json found next to {} and no HuggingFace model ID provided",
            model_path.display()
        )))
    }

    fn from_hf_tokenizer_with_overrides(
        tokenizer: tokenizers::Tokenizer,
        overrides: GgufTokenizerOverrides,
        chat_template: Option<String>,
    ) -> Result<Self> {
        let vocab_size = tokenizer.get_vocab_size(true);
        let token_strings: Vec<String> = (0..vocab_size)
            .map(|i| {
                tokenizer
                    .id_to_token(i as u32)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| char::REPLACEMENT_CHARACTER.to_string())
            })
            .collect();

        let bos_id = overrides
            .bos_id
            .or_else(|| {
                tokenizer
                    .token_to_id("<s>")
                    .or_else(|| tokenizer.token_to_id("<|begin_of_text|>"))
                    .or_else(|| tokenizer.token_to_id("<bos>"))
            })
            .unwrap_or(1);

        let eos_id = overrides
            .eos_id
            .or_else(|| {
                tokenizer
                    .token_to_id("</s>")
                    .or_else(|| tokenizer.token_to_id("<|end_of_text|>"))
                    .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
                    .or_else(|| tokenizer.token_to_id("<eos>"))
            })
            .unwrap_or(2);

        Ok(Self {
            eot_id: tokenizer.token_to_id("<|eot_id|>"),
            im_end_id: tokenizer.token_to_id("<|im_end|>"),
            inner: tokenizer,
            bos_id,
            eos_id,
            token_strings,
            chat_template,
        })
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizerError::Tokenizer(format!("Encode failed: {e}")))?;

        let mut ids: Vec<u32> = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| {
                id != self.bos_id && id != self.eos_id && !is_pad_token(id, &self.token_strings)
            })
            .collect();

        self.inner
            .decode(&filtered, true)
            .map_err(|e| TokenizerError::Tokenizer(format!("Decode failed: {e}")))
    }

    pub fn decode_one(&self, id: u32) -> Result<String> {
        if is_pad_token(id, &self.token_strings)
            || id == self.bos_id
            || id == self.eos_id
        {
            return Ok(String::new());
        }
        self.inner
            .decode(&[id], true)
            .map_err(|e| TokenizerError::Tokenizer(format!("Decode failed: {e}")))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_id
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_id
    }

    pub fn eot_token_id(&self) -> Option<u32> {
        self.eot_id
    }

    pub fn im_end_token_id(&self) -> Option<u32> {
        self.im_end_id
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if let Some(ref tmpl) = self.chat_template {
            return self.apply_jinja_template(messages, tmpl);
        }

        // Architecture-specific detection must come BEFORE generic ChatML/Llama3
        // because models like DeepSeek-R1-Qwen share vocabulary with ChatML models
        // but use completely different chat formats.

        // Check for DeepSeek-R1: <｜User｜>content<｜Assistant｜>
        let ds_user = self.inner.token_to_id(" <｜User｜>");
        let ds_assistant = self.inner.token_to_id(" <｜Assistant｜>");
        if let (Some(_), Some(_)) = (ds_user, ds_assistant) {
            return self.apply_deepseek_template(messages);
        }

        // Check for Gemma3: <start_of_turn>role\ncontent<end_of_turn>
        let start_of_turn = self.inner.token_to_id("<start_of_turn>");
        let end_of_turn = self.inner.token_to_id("<end_of_turn>");
        if let (Some(sot), Some(eot_tok)) = (start_of_turn, end_of_turn) {
            return self.apply_gemma_template(messages, sot, eot_tok);
        }

        // Check for TinyLlama: <|user|>\ncontent</s> etc.
        let tl_user = self.inner.token_to_id("<|user|>");
        let tl_assistant = self.inner.token_to_id("<|assistant|>");
        if let (Some(_), Some(_)) = (tl_user, tl_assistant) {
            return self.apply_tinyllama_template(messages);
        }

        // Generic: ChatML (used by Qwen, etc.)
        let im_start = self.inner.token_to_id("<|im_start|>");
        let im_end = self.inner.token_to_id("<|im_end|>");
        if let (Some(im_start), Some(im_end)) = (im_start, im_end) {
            return self.apply_chatml(messages, im_start, im_end);
        }

        // Llama3 format
        let start_header = self.inner.token_to_id("<|start_header_id|>");
        let end_header = self.inner.token_to_id("<|end_header_id|>");
        let eot = self.inner.token_to_id("<|eot_id|>");

        if start_header.is_none() || end_header.is_none() || eot.is_none() {
            // Last resort: raw concatenation (will produce bad output for instruction-tuned models)
            eprintln!("warning: no chat template found for this model; using raw text fallback. Output may be low quality.");
            let text: String = messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n");
            return self.encode(&text, true);
        }

        let start_header = start_header.unwrap();
        let end_header = end_header.unwrap();
        let eot = eot.unwrap();

        let mut tokens = vec![self.bos_id];
        for msg in messages {
            tokens.push(start_header);
            tokens.extend(self.encode(&msg.role, false)?);
            tokens.push(end_header);
            tokens.extend(self.encode(&format!("\n\n{}", msg.content), false)?);
            tokens.push(eot);
        }
        tokens.push(start_header);
        tokens.extend(self.encode("assistant", false)?);
        tokens.push(end_header);
        tokens.extend(self.encode("\n\n", false)?);
        Ok(tokens)
    }

    fn apply_jinja_template(&self, messages: &[ChatMessage], tmpl: &str) -> Result<Vec<u32>> {
        let mut env = Environment::new();
        env.set_keep_trailing_newline(true);
        env.add_filter("trim", |s: String| s.trim().to_string());
        env.add_function("raise_exception", |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::UndefinedError, msg))
        });

        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({"role": &m.role, "content": &m.content})
            })
            .collect();

        let mut ctx_map: BTreeMap<String, minijinja::Value> = BTreeMap::new();
        ctx_map.insert("messages".into(), minijinja::Value::from_serialize(&msgs));
        ctx_map.insert(
            "bos_token".into(),
            minijinja::Value::from(self.token_strings.get(self.bos_id as usize).cloned().unwrap_or_default()),
        );
        ctx_map.insert(
            "eos_token".into(),
            minijinja::Value::from(self.token_strings.get(self.eos_id as usize).cloned().unwrap_or_default()),
        );

        let rendered = env.render_str(tmpl, minijinja::Value::from(ctx_map))
            .map_err(|e| TokenizerError::Tokenizer(format!("Jinja template error: {e}")))?;

        self.encode(&rendered, false)
    }

    fn apply_chatml(&self, messages: &[ChatMessage], im_start: u32, im_end: u32) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_id];
        for msg in messages {
            tokens.push(im_start);
            tokens.extend(self.encode(&format!("{}\n{}", msg.role, msg.content), false)?);
            tokens.push(im_end);
            tokens.extend(self.encode("\n", false)?);
        }
        tokens.push(im_start);
        tokens.extend(self.encode("assistant\n", false)?);
        Ok(tokens)
    }

    /// Gemma3 format: <start_of_turn>role\ncontent<end_of_turn>\n
    fn apply_gemma_template(&self, messages: &[ChatMessage], sot: u32, eot: u32) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_id];
        for msg in messages {
            tokens.push(sot);
            tokens.extend(self.encode(&format!("{}\n{}", msg.role, msg.content), false)?);
            tokens.push(eot);
            tokens.extend(self.encode("\n", false)?);
        }
        tokens.push(sot);
        tokens.extend(self.encode("model\n", false)?);
        Ok(tokens)
    }

    /// TinyLlama format: <|user|>\ncontent</s>\n<|assistant|>\ncontent</s>
    fn apply_tinyllama_template(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_id];
        for msg in messages {
            match msg.role.as_str() {
                "user" => tokens.extend(self.encode(&format!("<|user|>\n{}</s>\n", msg.content), false)?),
                "system" => tokens.extend(self.encode(&format!("<|system|>\n{}</s>\n", msg.content), false)?),
                "assistant" => tokens.extend(self.encode(&format!("<|assistant|>\n{}</s>\n", msg.content), false)?),
                _ => tokens.extend(self.encode(&format!("<|user|>\n{}</s>\n", msg.content), false)?),
            }
        }
        tokens.extend(self.encode("<|assistant|>\n", false)?);
        Ok(tokens)
    }

    /// DeepSeek-R1: uses special tokens encoded as single IDs
    fn apply_deepseek_template(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let ds_user = self.inner.token_to_id("<｜User｜>");
        let ds_assistant = self.inner.token_to_id("<｜Assistant｜>");
        let ds_begin = self.inner.token_to_id("<｜begin▁of▁sentence｜>");
        let ds_think = self.inner.token_to_id("<｜end▁of▁thinking｜>");

        let mut tokens = Vec::new();
        // DeepSeek uses <｜begin▁of▁sentence｜> instead of regular BOS
        if let Some(bos) = ds_begin {
            tokens.push(bos);
        } else {
            tokens.push(self.bos_id);
        }
        for msg in messages {
            if msg.role == "user" {
                if let Some(u) = ds_user { tokens.push(u); }
                tokens.extend(self.encode(&msg.content, false)?);
            } else if msg.role == "system" {
                tokens.extend(self.encode(&msg.content, false)?);
            } else if msg.role == "assistant" {
                if let Some(a) = ds_assistant { tokens.push(a); }
                tokens.extend(self.encode(&msg.content, false)?);
            }
        }
        // Add generation prompt
        if let Some(a) = ds_assistant { tokens.push(a); }
        if let Some(t) = ds_think { tokens.push(t); }
        Ok(tokens)
    }
}

fn is_pad_token(id: u32, token_strings: &[String]) -> bool {
    let Some(name) = token_strings.get(id as usize) else {
        return false;
    };
    if name.len() < 4 {
        return false;
    }
    name.starts_with("[PAD")
        || name.starts_with("<unused")
        || name == "<pad>"
        || name == "<unk>"
}

fn load_chat_template(model_dir: &Path, hf_model_id: Option<&str>) -> Option<String> {
    let cfg_path = model_dir.join("tokenizer_config.json");
    if let Ok(content) = std::fs::read_to_string(&cfg_path) {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(tmpl) = cfg.get("chat_template").and_then(|v| v.as_str()) {
                return Some(tmpl.to_string());
            }
        }
    }

    if let Some(model_id) = hf_model_id {
        let url = format!("https://huggingface.co/{}/raw/main/tokenizer_config.json", model_id);
        if let Ok(response) = ureq::get(&url).call() {
            if let Ok(body) = response.into_body().read_to_string() {
                if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(tmpl) = cfg.get("chat_template").and_then(|v| v.as_str()) {
                        let _ = std::fs::write(
                            model_dir.join("tokenizer_config.json"),
                            &body,
                        );
                        return Some(tmpl.to_string());
                    }
                }
            }
        }
    }

    None
}
