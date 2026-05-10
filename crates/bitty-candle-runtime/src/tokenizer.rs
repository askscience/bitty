use std::path::Path;

#[derive(Debug, Clone, thiserror::Error)]
pub enum TokenizerError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: u32,
    eos_id: u32,
    eot_id: Option<u32>,
    im_end_id: Option<u32>,
    token_strings: Vec<String>,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::Tokenizer(format!("Failed to load tokenizer: {e}")))?;
        Self::from_hf_tokenizer(tokenizer)
    }

    pub fn from_gguf_path(model_path: &Path, hf_model_id: Option<&str>) -> Result<Self> {
        let parent = model_path.parent().unwrap_or(Path::new("."));
        let json_path = parent.join("tokenizer.json");
        if json_path.exists() {
            return Self::from_file(&json_path.to_string_lossy());
        }
        if let Some(model_id) = hf_model_id {
            let tokenizer = tokenizers::Tokenizer::from_pretrained(model_id, None)
                .map_err(|e| TokenizerError::Tokenizer(format!("Failed to load tokenizer from HF: {e}")))?;
            // Save locally so subsequent runs skip the network
            let _ = tokenizer.save(&json_path, false);
            return Self::from_hf_tokenizer(tokenizer);
        }
        Err(TokenizerError::Tokenizer(format!(
            "No tokenizer.json found next to {} and no HuggingFace model ID provided",
            model_path.display()
        )))
    }

    fn from_hf_tokenizer(tokenizer: tokenizers::Tokenizer) -> Result<Self> {
        let vocab_size = tokenizer.get_vocab_size(true);
        let token_strings: Vec<String> = (0..vocab_size)
            .map(|i| {
                tokenizer
                    .id_to_token(i as u32)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| char::REPLACEMENT_CHARACTER.to_string())
            })
            .collect();

        let bos_id = tokenizer.token_to_id("<s>")
            .or_else(|| tokenizer.token_to_id("<|begin_of_text|>"))
            .unwrap_or(1);
        let eos_id = tokenizer.token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_text|>"))
            .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
            .unwrap_or(2);

        Ok(Self {
            eot_id: tokenizer.token_to_id("<|eot_id|>"),
            im_end_id: tokenizer.token_to_id("<|im_end|>"),
            inner: tokenizer,
            bos_id,
            eos_id,
            token_strings,
        })
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
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
        let im_start = self.inner.token_to_id("<|im_start|>");
        let im_end = self.inner.token_to_id("<|im_end|>");
        if let (Some(im_start), Some(im_end)) = (im_start, im_end) {
            return self.apply_chatml(messages, im_start, im_end);
        }

        let start_header = self.inner.token_to_id("<|start_header_id|>");
        let end_header = self.inner.token_to_id("<|end_header_id|>");
        let eot = self.inner.token_to_id("<|eot_id|>");

        if start_header.is_none() || end_header.is_none() || eot.is_none() {
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
