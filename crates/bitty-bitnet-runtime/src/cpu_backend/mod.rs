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

pub mod types;
pub mod dequant;
pub mod ops;
pub mod matmul;
pub mod loader;
pub mod layers;

use std::path::Path;
use types::*;

/// CPU-loaded GGUF model ready for inference.
pub struct CpuModel {
    pub metadata: CpuModelMetadata,
    tokenizer: oxbitnet::Tokenizer,
    embed_tokens: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<CpuLayer>,
    lm_head: Option<LmHead>,
    kv_cache: KvCache,
    ssm_states: Vec<SsmState>,
}

impl CpuModel {
    /// Load a GGUF model from disk.
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("Cannot read model: {e}"))?;
        let (metadata, tokenizer, weights) = loader::load_gguf(&data)?;
        let num_layers = weights.layers.len().max(1);
        let meta = loader::metadata::extract_config(&metadata, num_layers);

        // Initialize KV cache
        let max_kv_dim = meta.num_kv_heads * meta.head_dim;
        let max_seq = meta.max_seq_len;
        let mut kv_cache = KvCache::new();
        kv_cache.reserve(meta.num_layers, max_kv_dim, max_seq);

        // Initialize SSM states
        let ssm_states: Vec<SsmState> = weights.layers.iter().map(|layer| {
            match &layer.kind {
                LayerKind::Ssm(w) => SsmState::new(w.d_inner, w.d_state, w.kernel_size),
                _ => SsmState::new(1, 1, 1), // dummy for non-SSM layers
            }
        }).collect();

        Ok(Self {
            metadata: meta,
            tokenizer,
            embed_tokens: weights.embed_tokens,
            final_norm: weights.final_norm,
            layers: weights.layers,
            lm_head: weights.lm_head,
            kv_cache,
            ssm_states,
        })
    }

    pub fn tokenizer(&self) -> &oxbitnet::Tokenizer {
        &self.tokenizer
    }

    /// Full end-to-end generation on CPU.
    pub fn generate(
        &self, prompt: &str, max_tokens: usize,
        temperature: f32, top_k: usize, repeat_penalty: f32,
    ) -> Result<String, String> {
        let tokens = self.tokenizer.encode(prompt, true)
            .map_err(|e| format!("Tokenize error: {e}"))?;
        let mut output = String::new();
        let eos = self.tokenizer.eos_token_id();
        let eot = self.tokenizer.eot_token_id();
        let im_end = self.tokenizer.im_end_token_id();
        let m = &self.metadata;
        let d = m.hidden_size;

        let mut kv_cache = KvCache::new();
        let mut ssm_states: Vec<SsmState> = self.layers.iter().map(|layer| {
            match &layer.kind {
                LayerKind::Ssm(w) => SsmState::new(w.d_inner, w.d_state, w.kernel_size),
                _ => SsmState::new(1, 1, 1),
            }
        }).collect();

        let vocab_size = m.vocab_size;
        let mut hidden = vec![0f32; d];
        for (pos, &tid) in tokens.iter().enumerate() {
            let tid = tid as usize;
            let mut h = vec![0f32; d];
            if tid < vocab_size && self.embed_tokens.len() >= d * vocab_size {
                for i in 0..d { h[i] = self.embed_tokens[tid * d + i]; }
            }
            for layer in &self.layers {
                h = layer.forward(&h, pos, &mut kv_cache, &mut ssm_states, m)?;
            }
            hidden = h;
            kv_cache.seq_len += 1;
        }

        // Track recently generated tokens for repeat penalty.
        let mut recent: Vec<u32> = tokens.clone();

        let mut pos = tokens.len();
        let mut rng_state: u64 = 0xC0FFEE_u64
            ^ (pos as u64).wrapping_mul(0x9E3779B97F4A7C15);
        for _ in 0..max_tokens {
            for layer in &self.layers {
                hidden = layer.forward(&hidden, pos, &mut kv_cache, &mut ssm_states, m)?;
            }
            kv_cache.seq_len += 1;
            let normed = ops::rms_norm(&hidden, &self.final_norm, m.rms_norm_eps);
            let mut logits = compute_logits(&normed, &self.lm_head, &self.embed_tokens, d, m.vocab_size)?;

            // Apply simple repeat penalty: divide positive logits, multiply negatives.
            if repeat_penalty > 1.0 {
                let window = recent.len().saturating_sub(64).max(0);
                for &tok in &recent[window..] {
                    let ti = tok as usize;
                    if ti < logits.len() {
                        if logits[ti] > 0.0 { logits[ti] /= repeat_penalty; }
                        else                 { logits[ti] *= repeat_penalty; }
                    }
                }
            }

            let next_token = sample(&logits, temperature, top_k, &mut rng_state);
            if next_token == eos || Some(next_token) == eot || Some(next_token) == im_end { break; }
            let text = self.tokenizer.decode_one(next_token)
                .unwrap_or_else(|_| char::REPLACEMENT_CHARACTER.to_string());
            output.push_str(&text);
            recent.push(next_token);
            if (next_token as usize) < vocab_size && self.embed_tokens.len() >= d * vocab_size {
                for i in 0..d { hidden[i] = self.embed_tokens[next_token as usize * d + i]; }
            }
            pos += 1;
        }
        Ok(output)
    }
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
    if logits.is_empty() { return 0; }
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
        scored.select_nth_unstable_by(top_k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
    }

    // Numerically stable softmax.
    let max = scored.iter().fold(f32::NEG_INFINITY, |acc, &(_, v)| if v > acc { v } else { acc });
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
    if s == 0 { s = 0x9E3779B97F4A7C15; }
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *rng_state = s;
    let u = ((s >> 11) as f32) / ((1u64 << 53) as f32);
    let target = u * sum;

    let mut acc = 0f32;
    for &(i, p) in &scored {
        acc += p;
        if acc >= target { return i as u32; }
    }
    scored.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

fn compute_logits(
    hidden: &[f32], lm_head: &Option<LmHead>,
    embed_tokens: &[f32], d: usize, vocab: usize,
) -> std::result::Result<Vec<f32>, String> {
    match lm_head {
        Some(LmHead::Packed(lm)) => {
            matmul::matmul(hidden, lm, d, vocab)
        }
        _ => {
            // Tied embeddings: embedding is stored as [vocab, hidden] (GGUF row-major)
            let mut logits = vec![0f32; vocab];
            if embed_tokens.len() >= d * vocab {
                for v in 0..vocab {
                    let mut sum = 0f32;
                    let offset = v * d;
                    for i in 0..d {
                        sum += hidden[i] * embed_tokens[offset + i];
                    }
                    logits[v] = sum;
                }
            }
            Ok(logits)
        }
    }
}


