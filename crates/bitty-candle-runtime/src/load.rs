use bitty_gguf_loader::mmap_store::{self, WeightStore as SharedWeightStore};
use bitty_model::gguf::{GgufTensorInfo, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32};
use candle_core::{Device, DType, Tensor};
use std::path::Path;

pub use bitty_gguf_loader::config::ModelConfig;
pub use bitty_gguf_loader::mmap_store::LoadError;

pub type Result<T> = std::result::Result<T, LoadError>;

pub struct LoadedModel {
    pub config: ModelConfig,
    pub weights: DeviceWeightStore,
}

fn candle_err(e: candle_core::Error) -> LoadError {
    LoadError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("candle: {e}")))
}

/// Device-aware wrapper around the shared mmap'd weight store.
pub struct DeviceWeightStore {
    inner: SharedWeightStore,
    device: Device,
}

impl DeviceWeightStore {
    pub fn get_raw(&self, name: &str) -> Option<&[u8]> {
        self.inner.get_raw(name)
    }

    pub fn get_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.inner.get_info(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }

    /// Zero-copy tensor upload from mmap to device (no intermediate Vec allocation).
    pub fn get_raw_tensor(&self, name: &str, shape: &[usize], ggml_type: u32) -> Result<Tensor> {
        let raw = self.get_raw(name).ok_or_else(|| LoadError::MissingWeight(name.to_string()))?;
        let dtype = match ggml_type {
            GGML_TYPE_F32 => DType::F32,
            GGML_TYPE_F16 => DType::F16,
            GGML_TYPE_BF16 => DType::BF16,
            _ => return Err(LoadError::UnsupportedGgmlType(ggml_type)),
        };
        Tensor::from_raw_buffer(raw, dtype, shape, &self.device).map_err(candle_err)
    }

    pub fn get_f32(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let raw = self.get_raw(name).ok_or_else(|| LoadError::MissingWeight(name.to_string()))?;
        Tensor::from_raw_buffer(raw, DType::F32, shape, &self.device).map_err(candle_err)
    }

    pub fn get_f16_to_f32(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let raw = self.get_raw(name).ok_or_else(|| LoadError::MissingWeight(name.to_string()))?;
        let halfs: &[half::f16] = bytemuck::cast_slice(raw);
        let data: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
        Tensor::from_vec(data, shape, &self.device).map_err(candle_err)
    }

    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.inner.tensors
    }

    pub fn build_offset_map(&self) -> std::collections::HashMap<String, (usize, usize, u32)> {
        self.inner.build_offset_map()
    }
}

pub fn load_gguf(source: &str, device: &Device) -> Result<LoadedModel> {
    let path = Path::new(source);
    let (mmap, gguf, data_offset) = mmap_store::load_gguf(path)?;

    let config = bitty_gguf_loader::extract_model_config(&gguf, &gguf.tensors);

    let inner = SharedWeightStore::new(mmap, gguf.tensors.clone(), data_offset);

    Ok(LoadedModel {
        config,
        weights: DeviceWeightStore {
            inner,
            device: device.clone(),
        },
    })
}
