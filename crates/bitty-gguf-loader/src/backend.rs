use std::path::Path;

/// Backend hardware type for reporting and dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Wgpu,
}

/// Common inference interface implemented by all Bitty backends.
///
/// ```ignore
/// use bitty_gguf_loader::InferenceBackend;
///
/// let mut model = MyBackend::load(path, None)?;
/// let logits = model.forward(&[1, 2, 3])?;
/// model.reset_kv_cache();
/// ```
pub trait InferenceBackend: Sized {
    type Error: std::fmt::Debug;

    /// Load model weights from a GGUF file path.
    fn load(path: &Path, hf_source: Option<&str>) -> Result<Self, Self::Error>;

    /// Run a full forward pass: token IDs → logits over vocabulary.
    fn forward(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, Self::Error>;

    /// Reset the KV cache for a new sequence.
    fn reset_kv_cache(&mut self);

    /// Hardware backend this model runs on.
    fn backend_kind(&self) -> BackendKind;

    /// Model hidden dimension.
    fn hidden_size(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}
