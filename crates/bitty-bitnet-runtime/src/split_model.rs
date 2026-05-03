use crate::{BitNetRuntimeError, Result};
use oxbitnet::gpu::buffer_pool::{BufferPool, GpuBuf};
use oxbitnet::gpu::{init_gpu, PipelineManager};
use oxbitnet::model::config::ModelConfig;
use oxbitnet::model::loader::{load_model, LoadOptions};
use oxbitnet::model::weights::WeightStore;
use oxbitnet::nn::attention::{create_kv_cache, Attention, KvCache};
use oxbitnet::nn::bitlinear::BitLinear;
use oxbitnet::nn::ffn::FFN;
use oxbitnet::nn::transformer::TransformerBlock;
use oxbitnet::tokenizer::Tokenizer;
use std::ops::Range;
use std::sync::Arc;
use wgpu::BufferUsages;

const EMBEDDING_WGSL: &str = include_str!("shaders/embedding.wgsl");
const RMSNORM_WGSL: &str = include_str!("shaders/rmsnorm.wgsl");
const F32_MATMUL_WGSL: &str = include_str!("shaders/f32_matmul.wgsl");

pub struct SplitBitNetModel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: PipelineManager,
    pool: BufferPool,
    pub config: ModelConfig,
    tokenizer: Tokenizer,
    embed_tokens: GpuBuf,
    layers: Vec<TransformerBlock>,
    final_norm: GpuBuf,
    lm_head: LmHead,
    kv_caches: Vec<KvCache>,
}

enum LmHead {
    Tied,
    F16(GpuBuf),
    Separate(BitLinear),
}

pub struct GpuActivation {
    pub buffer: GpuBuf,
    pub tokens: usize,
}

impl SplitBitNetModel {
    pub async fn load(source: &str, max_seq_len: usize) -> Result<Self> {
        let gpu = init_gpu()
            .await
            .map_err(|err| BitNetRuntimeError::Backend(err.to_string()))?;
        let result = load_model(
            source,
            Arc::clone(&gpu.device),
            Arc::clone(&gpu.queue),
            LoadOptions::default(),
        )
        .await
        .map_err(|err| BitNetRuntimeError::Backend(err.to_string()))?;
        let tokenizer = result
            .metadata
            .as_ref()
            .ok_or_else(|| BitNetRuntimeError::Backend("missing GGUF tokenizer metadata".into()))
            .and_then(|metadata| {
                Tokenizer::from_gguf_metadata(metadata)
                    .map_err(|err| BitNetRuntimeError::Backend(err.to_string()))
            })?;
        let model = Self::build(
            Arc::clone(&gpu.device),
            Arc::clone(&gpu.queue),
            result.config,
            &result.weights,
            tokenizer,
            max_seq_len,
        )?;
        Ok(model)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn reset_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.seq_len = 0;
        }
        for layer in &mut self.layers {
            layer.clear_bg_cache();
        }
        if let LmHead::Separate(ref mut lm_head) = self.lm_head {
            lm_head.clear_bg_cache();
        }
    }

    pub fn embed_tokens(&mut self, token_ids: &[u32]) -> GpuActivation {
        let n = token_ids.len();
        let token_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bitty_token_ids"),
            size: (token_ids.len() * 4).max(4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = token_buffer.slice(..).get_mapped_range_mut();
            let bytes: &[u8] = bytemuck::cast_slice(token_ids);
            view[..bytes.len()].copy_from_slice(bytes);
        }
        token_buffer.unmap();

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let hidden = self.dispatch_embedding(&mut encoder, &token_buffer, n);
        self.queue.submit(std::iter::once(encoder.finish()));
        GpuActivation {
            buffer: hidden,
            tokens: n,
        }
    }

    pub fn upload_activation(&self, payload: &[u8], tokens: usize) -> GpuActivation {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bitty_remote_activation"),
            size: payload.len().max(4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            view[..payload.len()].copy_from_slice(payload);
        }
        buffer.unmap();
        GpuActivation {
            buffer: Arc::new(buffer),
            tokens,
        }
    }

    pub fn forward_layers(
        &mut self,
        activation: GpuActivation,
        range: Range<usize>,
    ) -> GpuActivation {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut hidden = activation.buffer;
        for layer_id in range {
            let new_hidden = {
                let kv = &mut self.kv_caches[layer_id];
                self.layers[layer_id].forward(
                    &hidden,
                    activation.tokens,
                    kv,
                    &mut encoder,
                    &mut self.pipelines,
                    &self.pool,
                )
            };
            hidden = new_hidden;
            self.kv_caches[layer_id].seq_len += activation.tokens;
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        GpuActivation {
            buffer: hidden,
            tokens: activation.tokens,
        }
    }

    pub fn final_logits(&mut self, activation: GpuActivation) -> GpuBuf {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let normed = self.dispatch_final_norm(&mut encoder, &activation.buffer, activation.tokens);
        let lm_input = if activation.tokens > 1 {
            let lm_buf = self.pool.acquire(
                (self.config.hidden_size * 4) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            );
            encoder.copy_buffer_to_buffer(
                &normed,
                ((activation.tokens - 1) * self.config.hidden_size * 4) as u64,
                &lm_buf,
                0,
                (self.config.hidden_size * 4) as u64,
            );
            lm_buf
        } else {
            normed
        };
        let logits = match &self.lm_head {
            LmHead::F16(weight) => {
                self.dispatch_lm_head_with(&mut encoder, &lm_input, 1, weight.clone())
            }
            LmHead::Tied => {
                self.dispatch_lm_head_with(&mut encoder, &lm_input, 1, self.embed_tokens.clone())
            }
            LmHead::Separate(_) => {
                if let LmHead::Separate(ref mut head) = self.lm_head {
                    head.forward(&lm_input, 1, &mut encoder, &mut self.pipelines, &self.pool)
                } else {
                    unreachable!()
                }
            }
        };
        self.queue.submit(std::iter::once(encoder.finish()));
        logits
    }

    pub async fn read_activation(&self, activation: &GpuActivation) -> Result<Vec<u8>> {
        self.read_buffer_bytes(
            &activation.buffer,
            activation.tokens * self.config.hidden_size * std::mem::size_of::<f32>(),
        )
        .await
    }

    pub async fn read_logits(&self, logits: &GpuBuf) -> Result<Vec<f32>> {
        let bytes = self
            .read_buffer_bytes(logits, self.config.vocab_size * std::mem::size_of::<f32>())
            .await?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    async fn read_buffer_bytes(&self, source: &GpuBuf, size: usize) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bitty_readback"),
            size: size.max(4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        let (tx, rx) = tokio::sync::oneshot::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.await
            .map_err(|_| BitNetRuntimeError::Backend("GPU readback channel closed".into()))?
            .map_err(|_| BitNetRuntimeError::Backend("GPU readback failed".into()))?;
        let data = staging.slice(..).get_mapped_range();
        let result = data[..size].to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }

    fn build(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: ModelConfig,
        weights: &WeightStore,
        tokenizer: Tokenizer,
        max_seq_len: usize,
    ) -> Result<Self> {
        let pipelines = PipelineManager::new(Arc::clone(&device));
        let pool = BufferPool::new(Arc::clone(&device), 256);
        let require = |name: &str| -> Result<GpuBuf> {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| BitNetRuntimeError::MissingWeight(name.to_string()))
        };

        let embed_tokens = require("model.embed_tokens.weight")?;
        let final_norm = require("model.norm.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut kv_caches = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let head_dim = config.head_dim();
            let attention = Attention::new(
                Arc::clone(&device),
                config.clone(),
                BitLinear::new(
                    Arc::clone(&device),
                    require(&format!("{prefix}.self_attn.q_proj.weight"))?,
                    require(&format!("{prefix}.self_attn.q_proj.weight_scale"))?,
                    None,
                    config.hidden_size,
                    config.num_attention_heads * head_dim,
                ),
                BitLinear::new(
                    Arc::clone(&device),
                    require(&format!("{prefix}.self_attn.k_proj.weight"))?,
                    require(&format!("{prefix}.self_attn.k_proj.weight_scale"))?,
                    None,
                    config.hidden_size,
                    config.num_key_value_heads * head_dim,
                ),
                BitLinear::new(
                    Arc::clone(&device),
                    require(&format!("{prefix}.self_attn.v_proj.weight"))?,
                    require(&format!("{prefix}.self_attn.v_proj.weight_scale"))?,
                    None,
                    config.hidden_size,
                    config.num_key_value_heads * head_dim,
                ),
                BitLinear::new(
                    Arc::clone(&device),
                    require(&format!("{prefix}.self_attn.o_proj.weight"))?,
                    require(&format!("{prefix}.self_attn.o_proj.weight_scale"))?,
                    weights
                        .get(&format!("{prefix}.self_attn.sub_norm.weight"))
                        .cloned(),
                    config.num_attention_heads * head_dim,
                    config.hidden_size,
                ),
            );
            let ffn = FFN::new(
                Arc::clone(&device),
                config.clone(),
                BitLinear::new(
                    Arc::clone(&device),
                    require(&format!("{prefix}.mlp.up_proj.weight"))?,
                    require(&format!("{prefix}.mlp.up_proj.weight_scale"))?,
                    None,
                    config.hidden_size,
                    config.intermediate_size,
                ),
                BitLinear::new(
                    Arc::clone(&device),
                    require(&format!("{prefix}.mlp.down_proj.weight"))?,
                    require(&format!("{prefix}.mlp.down_proj.weight_scale"))?,
                    weights
                        .get(&format!("{prefix}.mlp.sub_norm.weight"))
                        .cloned(),
                    config.intermediate_size,
                    config.hidden_size,
                ),
                if weights.has(&format!("{prefix}.mlp.gate_proj.weight")) {
                    Some(BitLinear::new(
                        Arc::clone(&device),
                        require(&format!("{prefix}.mlp.gate_proj.weight"))?,
                        require(&format!("{prefix}.mlp.gate_proj.weight_scale"))?,
                        None,
                        config.hidden_size,
                        config.intermediate_size,
                    ))
                } else {
                    None
                },
            );
            layers.push(TransformerBlock::new(
                Arc::clone(&device),
                config.clone(),
                require(&format!("{prefix}.input_layernorm.weight"))?,
                require(&format!("{prefix}.post_attention_layernorm.weight"))?,
                attention,
                ffn,
            ));
            kv_caches.push(create_kv_cache(&device, &config, max_seq_len));
        }

        let lm_head = if config.tie_word_embeddings || !weights.has("lm_head.weight") {
            LmHead::Tied
        } else if config.lm_head_f16 {
            LmHead::F16(require("lm_head.weight")?)
        } else {
            LmHead::Separate(BitLinear::new(
                Arc::clone(&device),
                require("lm_head.weight")?,
                require("lm_head.weight_scale")?,
                weights
                    .get("lm_head.input_norm.weight")
                    .cloned()
                    .or_else(|| Some(final_norm.clone())),
                config.hidden_size,
                config.vocab_size,
            ))
        };

        Ok(Self {
            device,
            queue,
            pipelines,
            pool,
            config,
            tokenizer,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            kv_caches,
        })
    }

    fn dispatch_embedding(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        token_buffer: &wgpu::Buffer,
        n: usize,
    ) -> GpuBuf {
        let entry = self
            .pipelines
            .get_or_create_default("embedding", EMBEDDING_WGSL);
        let output = self.pool.acquire(
            (n * self.config.hidden_size * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let params = create_uniform_raw(
            &self.device,
            &[
                (n as u32).to_le_bytes(),
                (self.config.hidden_size as u32).to_le_bytes(),
                (self.config.vocab_size as u32).to_le_bytes(),
            ]
            .concat(),
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitty_embedding"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, token_buffer),
                buf_entry(1, &self.embed_tokens),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });
        let total = (n * self.config.hidden_size) as u32;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        output
    }

    fn dispatch_final_norm(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        n: usize,
    ) -> GpuBuf {
        let entry = self
            .pipelines
            .get_or_create_default("rmsnorm", RMSNORM_WGSL);
        let output = self.pool.acquire(
            (n * self.config.hidden_size * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let params = create_uniform_raw(
            &self.device,
            &[
                (n as u32).to_le_bytes(),
                (self.config.hidden_size as u32).to_le_bytes(),
                self.config.rms_norm_eps.to_le_bytes(),
            ]
            .concat(),
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitty_final_norm"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, &self.final_norm),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
        output
    }

    fn dispatch_lm_head_with(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        n: usize,
        weight: GpuBuf,
    ) -> GpuBuf {
        let entry = self
            .pipelines
            .get_or_create_default("f32_matmul", F32_MATMUL_WGSL);
        let output = self.pool.acquire(
            (n * self.config.vocab_size * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let params = create_uniform_raw(
            &self.device,
            &[
                (n as u32).to_le_bytes(),
                (self.config.vocab_size as u32).to_le_bytes(),
                (self.config.hidden_size as u32).to_le_bytes(),
            ]
            .concat(),
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitty_lm_head"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, &weight),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });
        let total = (n * self.config.vocab_size) as u32;
        let wg_x = total.min(65535);
        let wg_y = total.div_ceil(65535);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        output
    }
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_uniform_raw(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(4).div_ceil(4) * 4) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = buffer.slice(..).get_mapped_range_mut();
        view[..data.len()].copy_from_slice(data);
    }
    buffer.unmap();
    buffer
}
