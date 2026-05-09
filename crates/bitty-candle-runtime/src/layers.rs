use candle_core::{Device, Result, Tensor, D};
use candle_nn::ops::rms_norm;

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
    pub tie_word_embeddings: bool,
    pub lm_head_f16: bool,
    pub is_qwen: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, last - half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok((x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?)?)
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
        let q = apply_rotary_emb(&q, &cos_q, &sin_q)?;

        let k_seq_start = cache.seq_len;
        let cos_k = cos.narrow(0, k_seq_start, n_tokens)?;
        let sin_k = sin.narrow(0, k_seq_start, n_tokens)?;
        let k = apply_rotary_emb(&k, &cos_k, &sin_k)?;

        let k_for_full = if let Some(ref ck) = cache.cache_k {
            Tensor::cat(&[ck, &k], 2)?
        } else {
            k.clone()
        };
        let v_for_full = if let Some(ref cv) = cache.cache_v {
            Tensor::cat(&[cv, &v], 2)?
        } else {
            v.clone()
        };

        // GQA: repeat K/V to match number of query heads
        let (k_expanded, v_expanded) = if self.gqa_group_size > 1 {
            let k = k_for_full.repeat((1, self.gqa_group_size, 1, 1))?;
            let v = v_for_full.repeat((1, self.gqa_group_size, 1, 1))?;
            (k, v)
        } else {
            (k_for_full, v_for_full)
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_weights = (q.matmul(&k_expanded.t()?)? * scale as f64)?;

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v_expanded)?;

        let attn_output = attn_output.reshape((n_tokens, self.num_heads * self.head_dim))?;

        let output = attn_output.matmul(&self.o_proj_w.t()?)?;

        cache.cache_k = Some(k_expanded);
        cache.cache_v = Some(v_expanded);
        cache.seq_len = total_seq_len;

        Ok(output)
    }

    fn reshape_head(x: Tensor) -> Result<Tensor> {
        let (n_tokens, n_heads, head_dim) = x.dims3()?;
        x.reshape((1, n_heads, n_tokens, head_dim))
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
    attention: Attention,
    ffn: FFN,
    rms_norm_eps: f32,
}

impl TransformerBlock {
    pub fn new(
        input_ln_w: Tensor,
        post_attn_ln_w: Tensor,
        attention: Attention,
        ffn: FFN,
        rms_norm_eps: f32,
    ) -> Self {
        Self {
            input_ln_w,
            post_attn_ln_w,
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
        let x = (residual + attn_out)?;

        let residual = &x;
        let normed = rms_norm(&x, &self.post_attn_ln_w, self.rms_norm_eps)?;
        let ffn_out = self.ffn.forward(&normed)?;
        residual + ffn_out
    }
}
