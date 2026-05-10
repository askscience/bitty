use candle_core::{Device, Result, Tensor, D};
use candle_nn::ops::rms_norm;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeStyle {
    Neox,
    Interleaved,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_style: RopeStyle,
    pub tie_word_embeddings: bool,
    pub lm_head_f16: bool,
    pub is_qwen: bool,
    pub embedding_scale: Option<f32>,
    pub final_logit_softcap: Option<f32>,
    pub is_gemma3: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

fn rotate_half_neox(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, last - half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_emb_neox(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok((x.broadcast_mul(cos)? + rotate_half_neox(x)?.broadcast_mul(sin)?)?)
}

fn apply_rotary_emb_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let last_d = x.dim(D::Minus1)?;
    let half = last_d / 2;
    let cos_half = cos.narrow(D::Minus1, 0, half)?;
    let sin_half = sin.narrow(D::Minus1, 0, half)?;

    let (batch, n_heads, n_tokens, _) = x.dims4()?;
    let xr = x.reshape((batch, n_heads, n_tokens, half, 2))?;
    let x0 = xr.narrow(4, 0, 1)?.reshape((batch, n_heads, n_tokens, half))?;
    let x1 = xr.narrow(4, 1, 1)?.reshape((batch, n_heads, n_tokens, half))?;

    let cos_b = cos_half.reshape((1, 1, n_tokens, half))?;
    let sin_b = sin_half.reshape((1, 1, n_tokens, half))?;

    let r0 = (x0.broadcast_mul(&cos_b)? - x1.broadcast_mul(&sin_b)?)?;
    let r1 = (x0.broadcast_mul(&sin_b)? + x1.broadcast_mul(&cos_b)?)?;

    let r0u = r0.unsqueeze(4)?;
    let r1u = r1.unsqueeze(4)?;
    let ri = Tensor::cat(&[&r0u, &r1u], 4)?;
    ri.reshape((batch, n_heads, n_tokens, last_d))
}

fn precompute_freqs_cis(
    head_dim: usize,
    seq_len: usize,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let freqs: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
        .collect();

    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let freqs_t = Tensor::from_vec(freqs, head_dim / 2, device)?;
    let positions_t = Tensor::from_vec(positions, seq_len, device)?;

    let freqs = positions_t.unsqueeze(1)?.broadcast_mul(&freqs_t.unsqueeze(0)?)?;
    let freqs = freqs.reshape((seq_len, head_dim / 2))?;

    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
    let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
    Ok((cos, sin))
}

pub struct Attention {
    q_proj_w: Tensor,
    k_proj_w: Tensor,
    v_proj_w: Tensor,
    o_proj_w: Tensor,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_group_size: usize,
    is_qwen: bool,
    rope_theta: f32,
    rope_style: RopeStyle,
    rope_cos: Option<Tensor>,
    rope_sin: Option<Tensor>,
    max_seq_len: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj_w: Tensor,
        k_proj_w: Tensor,
        v_proj_w: Tensor,
        o_proj_w: Tensor,
        q_bias: Option<Tensor>,
        k_bias: Option<Tensor>,
        v_bias: Option<Tensor>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        is_qwen: bool,
        rope_theta: f32,
        rope_style: RopeStyle,
        max_seq_len: usize,
    ) -> Self {
        Self {
            q_proj_w,
            k_proj_w,
            v_proj_w,
            o_proj_w,
            q_bias,
            k_bias,
            v_bias,
            num_heads,
            num_kv_heads,
            head_dim,
            gqa_group_size: num_heads / num_kv_heads,
            is_qwen,
            rope_theta,
            rope_style,
            rope_cos: None,
            rope_sin: None,
            max_seq_len,
        }
    }

    fn ensure_rope_cache(&mut self, total_seq_len: usize, device: &Device) -> Result<()> {
        let needed = total_seq_len.next_multiple_of(32).max(32);
        if self.rope_cos.is_some() {
            let current = self.rope_cos.as_ref().unwrap().dim(0)?;
            if current >= needed {
                return Ok(());
            }
        }
        let (cos, sin) = precompute_freqs_cis(self.head_dim, needed, self.rope_theta, device)?;
        self.rope_cos = Some(cos);
        self.rope_sin = Some(sin);
        Ok(())
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        n_tokens: usize,
        cache: &mut crate::kv_cache::KvCache,
    ) -> Result<Tensor> {
        let (_batch, _seq_len, hidden_size) = x.dims3()?;
        let hidden = x.reshape((n_tokens, hidden_size))?;
        let device = x.device();

        let mut q = hidden.matmul(&self.q_proj_w.t()?)?;
        if let Some(ref bias) = self.q_bias {
            q = q.broadcast_add(bias)?;
        }
        let mut k = hidden.matmul(&self.k_proj_w.t()?)?;
        if let Some(ref bias) = self.k_bias {
            k = k.broadcast_add(bias)?;
        }
        let mut v = hidden.matmul(&self.v_proj_w.t()?)?;
        if let Some(ref bias) = self.v_bias {
            v = v.broadcast_add(bias)?;
        }

        if self.is_qwen {
            q = q.reshape((n_tokens, self.num_heads, self.head_dim))?;
            k = k.reshape((n_tokens, self.num_kv_heads, self.head_dim))?;
            v = v.reshape((n_tokens, self.num_kv_heads, self.head_dim))?;
        }

        let q = Self::reshape_head(q.reshape((n_tokens, self.num_heads, self.head_dim))?)?;
        let k = Self::reshape_head(k.reshape((n_tokens, self.num_kv_heads, self.head_dim))?)?;
        let v = Self::reshape_head(v.reshape((n_tokens, self.num_kv_heads, self.head_dim))?)?;

        let total_seq_len = cache.seq_len + n_tokens;
        self.ensure_rope_cache(total_seq_len, device)?;

        let cos = self.rope_cos.as_ref().unwrap();
        let sin = self.rope_sin.as_ref().unwrap();

        let q_seq_start = cache.seq_len;
        let cos_q = cos.narrow(0, q_seq_start, n_tokens)?;
        let sin_q = sin.narrow(0, q_seq_start, n_tokens)?;
        let q = match self.rope_style {
            RopeStyle::Neox => apply_rotary_emb_neox(&q, &cos_q, &sin_q)?,
            RopeStyle::Interleaved => apply_rotary_emb_interleaved(&q, &cos_q, &sin_q)?,
        };

        let k_seq_start = cache.seq_len;
        let cos_k = cos.narrow(0, k_seq_start, n_tokens)?;
        let sin_k = sin.narrow(0, k_seq_start, n_tokens)?;
        let k = match self.rope_style {
            RopeStyle::Neox => apply_rotary_emb_neox(&k, &cos_k, &sin_k)?,
            RopeStyle::Interleaved => apply_rotary_emb_interleaved(&k, &cos_k, &sin_k)?,
        };

        cache.append(&k, &v)?;

        let k_full = cache.cache_k.as_ref().unwrap();
        let v_full = cache.cache_v.as_ref().unwrap();

        // GQA: repeat K/V to match number of query heads
        let (k_expanded, v_expanded) = if self.gqa_group_size > 1 {
            let k = k_full.repeat((1, self.gqa_group_size, 1, 1))?;
            let v = v_full.repeat((1, self.gqa_group_size, 1, 1))?;
            (k, v)
        } else {
            (k_full.clone(), v_full.clone())
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_weights = (q.matmul(&k_expanded.t()?)? * scale as f64)?;

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v_expanded)?;

        let attn_output = attn_output.reshape((n_tokens, self.num_heads * self.head_dim))?;

        let output = attn_output.matmul(&self.o_proj_w.t()?)?;

        Ok(output)
    }

    fn reshape_head(x: Tensor) -> Result<Tensor> {
        let (n_tokens, n_heads, head_dim) = x.dims3()?;
        x.reshape((1, n_heads, n_tokens, head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_seq_len_increment() -> candle_core::Result<()> {
        let device = Device::Cpu;

        let mut attention = Attention::new(
            Tensor::zeros((16, 16), candle_core::DType::F32, &device)?,
            Tensor::zeros((8, 16), candle_core::DType::F32, &device)?,
            Tensor::zeros((8, 16), candle_core::DType::F32, &device)?,
            Tensor::zeros((16, 16), candle_core::DType::F32, &device)?,
            None, None, None,
            2, 1, 8, false, 10000.0, RopeStyle::Neox, 128,
        );

        let mut cache = crate::kv_cache::KvCache::new(128);
        let input = Tensor::zeros((1, 1, 16), candle_core::DType::F32, &device)?;

        // First token
        attention.forward(&input, 1, &mut cache)?;
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.cache_k.as_ref().unwrap().dim(2)?, 1);

        // Second token
        attention.forward(&input, 1, &mut cache)?;
        assert_eq!(cache.seq_len, 2);
        assert_eq!(cache.cache_k.as_ref().unwrap().dim(2)?, 2);

        Ok(())
    }
}

pub struct FFN {
    up_proj_w: Tensor,
    gate_proj_w: Option<Tensor>,
    down_proj_w: Tensor,
}

impl FFN {
    pub fn new(up_proj_w: Tensor, gate_proj_w: Option<Tensor>, down_proj_w: Tensor) -> Self {
        Self {
            up_proj_w,
            gate_proj_w,
            down_proj_w,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up = x.matmul(&self.up_proj_w.t()?)?;
        let activated = if let Some(ref gate_w) = self.gate_proj_w {
            let gate = x.matmul(&gate_w.t()?)?;
            let gate = candle_nn::ops::silu(&gate)?;
            gate.broadcast_mul(&up)?
        } else {
            candle_nn::ops::silu(&up)?
        };
        activated.matmul(&self.down_proj_w.t()?)
    }
}

pub struct TransformerBlock {
    input_ln_w: Tensor,
    post_attn_ln_w: Tensor,
    post_attention_norm: Option<Tensor>,
    pre_ffn_norm: Option<Tensor>,
    post_ffn_norm: Option<Tensor>,
    attention: Attention,
    ffn: FFN,
    rms_norm_eps: f32,
}

impl TransformerBlock {
    pub fn new(
        input_ln_w: Tensor,
        post_attn_ln_w: Tensor,
        post_attention_norm: Option<Tensor>,
        pre_ffn_norm: Option<Tensor>,
        post_ffn_norm: Option<Tensor>,
        attention: Attention,
        ffn: FFN,
        rms_norm_eps: f32,
    ) -> Self {
        Self {
            input_ln_w,
            post_attn_ln_w,
            post_attention_norm,
            pre_ffn_norm,
            post_ffn_norm,
            attention,
            ffn,
            rms_norm_eps,
        }
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        n_tokens: usize,
        cache: &mut crate::kv_cache::KvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let normed = rms_norm(x, &self.input_ln_w, self.rms_norm_eps)?;
        let attn_out = self.attention.forward(&normed, n_tokens, cache)?;
        let mut x = (residual + attn_out)?;

        if let Some(ref post_attn_norm) = self.post_attention_norm {
            x = rms_norm(&x, post_attn_norm, self.rms_norm_eps)?;
        }

        let ffn_input = if let Some(ref pre_ffn) = self.pre_ffn_norm {
            rms_norm(&x, pre_ffn, self.rms_norm_eps)?
        } else {
            rms_norm(&x, &self.post_attn_ln_w, self.rms_norm_eps)?
        };

        let ffn_out = self.ffn.forward(&ffn_input)?;

        let ffn_out = if let Some(ref post_ffn) = self.post_ffn_norm {
            rms_norm(&ffn_out, post_ffn, self.rms_norm_eps)?
        } else {
            ffn_out
        };

        (x + ffn_out)
    }
}
