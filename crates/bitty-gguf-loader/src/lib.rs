//! Shared GGUF loading infrastructure for all Bitty backends.
//!
//! Centralizes mmap handling, GGUF parsing, tensor offset calculations,
//! and model configuration extraction. Consumed by candle-runtime,
//! bitnet-runtime (CPU), and wgpu-runtime backends.

pub mod mmap_store;
pub mod config;
pub mod backend;

pub use backend::{BackendKind, InferenceBackend};
pub use config::{extract_model_config, ModelConfig, RopeStyle};
pub use mmap_store::{WeightStore, LoadedModel, LoadError, load_gguf};
pub use mmap_store::compute_data_offset;
pub use bitty_model::gguf;
