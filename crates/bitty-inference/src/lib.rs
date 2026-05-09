pub mod backend;
pub mod executor;
pub mod lifecycle;
pub mod rust_bitnet;
pub mod sampling;
pub mod tiny_lm;

pub use backend::{BackendCapability, BackendDecision, ModelBackendKind};
pub use executor::{
    DraftExecutor, ExecutorError, FakeLayerExecutor, LayerExecutor, LowBitReferenceExecutor,
    StubLayerExecutor,
};
pub use lifecycle::{
    BatchPolicy, DecodePipeline, InferencePhase, PrefixCacheKey, RequestLifecycle,
    SpeculativeDecision,
};
pub use rust_bitnet::{BitNetBackendProbe, BitNetLayerExecutor, RustBitNetError};
pub use tiny_lm::TinyLanguageModel;
