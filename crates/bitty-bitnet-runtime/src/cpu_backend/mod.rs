//! CPU inference backend for GGUF quantized models.
//!
//! Supports all quantized GGUF formats (Q4_K, Q5_K, Q6_K, Q8_0, etc.)
//! with on-the-fly dequantization during CPU matmul. Handles both
//! standard transformer attention layers and Mamba SSM layers.
//!
//! Architecture:
//! - `types.rs`     — Core data structures
//! - `loader/`      — GGUF parsing, tensor name classification, metadata extraction
//! - `layers/`      — Layer implementations (attention, SSM, MLP)
//! - `matmul/`      — Quantized matrix-vector multiply by GGML type
//! - `dequant.rs`   — Dequant block readers (Q4K, Q6K, Q8_0)
//! - `ops.rs`       — Shared operations (RMSNorm, SiLU, softmax, RoPE, softplus)

pub mod dequant;
pub mod layers;
pub mod loader;
pub mod matmul;
pub mod ops;
pub mod types;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use types::*;

/// CPU-loaded GGUF model ready for inference.
pub struct CpuModel {
    pub metadata: CpuModelMetadata,
    tokenizer: bitty_candle_runtime::Tokenizer,
    embed_tokens: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<CpuLayer>,
    lm_head: Option<LmHead>,
    kv_cache: KvCache,
    rope_cache: RopeCache,
    mmap: Option<memmap2::Mmap>,
}

impl CpuModel {
    /// Load a GGUF model from disk via memory-mapping for zero-copy weight access.
    pub fn load(path: &Path, hf_model_id: Option<&str>) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open model file: {e}"))?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| format!("Cannot mmap model: {e}"))?
        };
        let (gguf_metadata, weights) = loader::load_gguf(&mmap)?;
        let overrides = loader::metadata::extract_tokenizer_overrides(&gguf_metadata);
        let tokenizer = bitty_candle_runtime::Tokenizer::from_gguf_path_with_overrides(
            path,
            hf_model_id,
            overrides,
        )
        .map_err(|e| format!("tokenizer error: {e}"))?;
        let num_layers = weights.layers.len().max(1);
        let meta = loader::metadata::extract_config(&gguf_metadata, num_layers);

        let max_kv_dim = meta.num_kv_heads * meta.head_dim;
        let mut kv_cache = KvCache::new();
        kv_cache.reserve(meta.num_layers, max_kv_dim, meta.max_seq_len);

        let rope_d = meta.rope_dim.max(2);
        let rope_d = if rope_d % 2 == 1 { rope_d + 1 } else { rope_d };
        let rope_cache = RopeCache::new(meta.max_seq_len, rope_d, meta.rope_theta);

        Ok(Self {
            metadata: meta,
            tokenizer,
            embed_tokens: weights.embed_tokens,
            final_norm: weights.final_norm,
            layers: weights.layers,
            lm_head: weights.lm_head,
            kv_cache,
            rope_cache,
            mmap: Some(mmap),
        })
    }

    pub fn tokenizer(&self) -> &bitty_candle_runtime::Tokenizer {
        &self.tokenizer
    }

    pub fn generate_chat_stream<F>(
        &self,
        messages: &[bitty_candle_runtime::tokenizer::ChatMessage],
        _reset_cache: bool,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
        seed: Option<u64>,
        on_delta: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let prompt_ids = self
            .tokenizer
            .apply_chat_template(messages)
            .map_err(|e| format!("chat template error: {e}"))?;
        self.generate_from_ids(
            &prompt_ids,
            max_tokens,
            temperature,
            top_k,
            repeat_penalty,
            seed,
            on_delta,
        )
    }

    /// Generates tokens directly from pre-tokenized IDs — no decode→encode round-trip.
    pub fn generate_from_ids<F>(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
        seed: Option<u64>,
        mut on_delta: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let tokens = prompt_ids.to_vec();
        let eos = self.tokenizer.eos_token_id();
        let eot = self.tokenizer.eot_token_id();
        let im_end = self.tokenizer.im_end_token_id();
        let m = &self.metadata;
        let d = m.hidden_size;

        let mut kv_cache = KvCache::new();
        let mut recurrent = init_recurrent_states(&self.layers, m);

        let vocab_size = m.vocab_size;

        // ---- Process prompt tokens
        let prompt_len = tokens.len();
        if prompt_len == 0 {
            return Ok(String::new());
        }
        let mut last_hidden = vec![0f32; d];
        for (pos, &tid) in tokens.iter().enumerate() {
            let tid = tid as usize;
            let mut h = embed_token(tid, &self.embed_tokens, d, vocab_size, m.embedding_scale);
            for layer in &self.layers {
                h = layer.forward(&h, pos, &mut kv_cache, &mut recurrent, m, &self.rope_cache)?;
            }
            last_hidden = h;
            kv_cache.seq_len += 1;
        }

        // ---- RNG setup: seed from SystemTime + prompt_pos by default, overridable
        let rng_base = seed.unwrap_or_else(|| {
            let ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            ns ^ (prompt_len as u64).wrapping_mul(0x9E3779B97F4A7C15)
        });
        let mut rng_state = rng_base;

        let mut recent: Vec<u32> = tokens.clone();
        let mut generated: Vec<u32> = Vec::new();
        let mut emitted: String = String::new();

        let eps = m.rms_norm_eps;
        let softcap = m.final_logit_softcapping;

        // ---- Generation loop
        let mut pos = prompt_len;
        for _step in 0..max_tokens {
            // Compute logits from the current hidden state (norm + lm_head)
            let normed = ops::rms_norm(&last_hidden, &self.final_norm, eps);
            let mut logits =
                compute_logits(&normed, &self.lm_head, &self.embed_tokens, d, vocab_size)?;

            // Gemma logit softcapping
            if let Some(attn_logit_softcapping) = softcap {
                for v in logits.iter_mut() {
                    *v = attn_logit_softcapping * (*v / attn_logit_softcapping).tanh();
                }
            }

            // Repeat penalty
            if repeat_penalty > 1.0 {
                let window = recent.len().saturating_sub(64).max(0);
                for &tok in &recent[window..] {
                    let ti = tok as usize;
                    if ti < logits.len() {
                        if logits[ti] > 0.0 {
                            logits[ti] /= repeat_penalty;
                        } else {
                            logits[ti] *= repeat_penalty;
                        }
                    }
                }
            }

            let next_token = sample(&logits, temperature, top_k, &mut rng_state);
            if next_token == eos || Some(next_token) == eot || Some(next_token) == im_end {
                break;
            }

            generated.push(next_token);
            recent.push(next_token);

            // Decode accumulated generated tokens for streaming
            if let Ok(full) = self.tokenizer.decode(&generated) {
                if full.len() > emitted.len() && full.starts_with(&emitted) {
                    let tail = &full[emitted.len()..];
                    if !tail.ends_with('\u{FFFD}') {
                        on_delta(tail);
                        emitted = full;
                    }
                }
            }

            // Embed the sampled token and run layers for the next position
            last_hidden = embed_token(
                next_token as usize,
                &self.embed_tokens,
                d,
                vocab_size,
                m.embedding_scale,
            );
            for layer in &self.layers {
                last_hidden =
                    layer.forward(&last_hidden, pos, &mut kv_cache, &mut recurrent, m, &self.rope_cache)?;
            }
            kv_cache.seq_len += 1;
            pos += 1;
        }

        // Flush any remaining decoded text
        if let Ok(full) = self.tokenizer.decode(&generated) {
            if full.len() > emitted.len() && full.starts_with(&emitted) {
                on_delta(&full[emitted.len()..]);
                emitted = full;
            }
        }
        Ok(emitted)
    }

    /// Full end-to-end generation on CPU, streaming decoded text deltas to
    /// `on_delta` as they become available. Returns the complete final text.
    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
        on_delta: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenize error: {e}"))?;
        self.generate_from_ids(&tokens, max_tokens, temperature, top_k, repeat_penalty, None, on_delta)
    }

    /// Full end-to-end generation on CPU.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
    ) -> Result<String, String> {
        self.generate_stream(prompt, max_tokens, temperature, top_k, repeat_penalty, |_| {})
    }
}

fn embed_token(
    tid: usize,
    embed_tokens: &[f32],
    d: usize,
    vocab_size: usize,
    embedding_scale: Option<f32>,
) -> Vec<f32> {
    let mut h = vec![0f32; d];
    if tid < vocab_size && embed_tokens.len() >= d * vocab_size {
        let off = tid * d;
        for i in 0..d {
            h[i] = embed_tokens[off + i];
        }
    }
    if let Some(scale) = embedding_scale {
        for v in h.iter_mut() {
            *v *= scale;
        }
    }
    h
}

fn init_recurrent_states(layers: &[CpuLayer], meta: &CpuModelMetadata) -> Vec<RecurrentState> {
    let max_idx = layers.iter().map(|l| l.layer_idx).max().unwrap_or(0);
    let mut recurrent = vec![RecurrentState::None; max_idx + 1];
    for layer in layers {
        recurrent[layer.layer_idx] = match &layer.kind {
            LayerKind::Ssm(w) => {
                RecurrentState::new_mamba(w.d_inner, w.d_state, w.kernel_size)
            }
            LayerKind::LinearAttn(_) => {
                let num_v = meta.ssm_dt_rank.max(1);
                let head_v = meta.ssm_d_inner / num_v;
                let head_k = meta.ssm_d_state;
                let num_k = meta.ssm_n_group;
                let key_dim = head_k * num_k;
                let value_dim = head_v * num_v;
                let conv_dim = key_dim * 2 + value_dim;
                let d_conv = meta.ssm_d_conv.max(1);
                RecurrentState::new_qwen_linear(d_conv, conv_dim, head_v, num_v)
            }
            _ => RecurrentState::None,
        };
    }
    recurrent
}

/// Argmax over logits (used at temperature 0 or as a safe fallback).
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Xorshift-style sampler over logits with temperature and top-k.
fn sample(logits: &[f32], temperature: f32, top_k: usize, rng_state: &mut u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return argmax(logits);
    }

    // Collect (index, logit) and apply temperature.
    let inv_t = 1.0f32 / temperature.max(1e-6);
    let mut scored: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v * inv_t))
        .collect();

    // Top-k truncation (0 = disabled).
    if top_k > 0 && top_k < scored.len() {
        scored.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
    }

    // Numerically stable softmax.
    let max = scored.iter().fold(
        f32::NEG_INFINITY,
        |acc, &(_, v)| if v > acc { v } else { acc },
    );
    let mut sum = 0f32;
    for e in scored.iter_mut() {
        let p = (e.1 - max).exp();
        e.1 = p;
        sum += p;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(logits);
    }

    // xorshift64 -> uniform in [0, 1)
    let mut s = *rng_state;
    if s == 0 {
        s = 0x9E3779B97F4A7C15;
    }
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *rng_state = s;
    let u = ((s >> 11) as f32) / ((1u64 << 53) as f32);
    let target = u * sum;

    let mut acc = 0f32;
    for &(i, p) in &scored {
        acc += p;
        if acc >= target {
            return i as u32;
        }
    }
    scored.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

fn compute_logits(
    hidden: &[f32],
    lm_head: &Option<LmHead>,
    embed_tokens: &[f32],
    d: usize,
    vocab: usize,
) -> std::result::Result<Vec<f32>, String> {
    use rayon::prelude::*;
    match lm_head {
        Some(LmHead::Packed(lm)) => matmul::matmul(hidden, lm, d, vocab),
        _ => {
            let mut logits = vec![0f32; vocab];
            if embed_tokens.len() >= d * vocab {
                logits
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(v, out)| {
                        let offset = v * d;
                        let mut sum = 0f32;
                        for i in 0..d {
                            sum += hidden[i] * embed_tokens[offset + i];
                        }
                        *out = sum;
                    });
            }
            Ok(logits)
        }
    }
}
