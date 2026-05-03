use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelBackendKind {
    ExternalBitnetCpp,
    DistributedReference,
    RustBitNetExperimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendCapability {
    LocalBitNetInference,
    ShardAwareLayerExecution,
    DeterministicReference,
    ProductionKernel,
    RustOnlyCorrectnessGates,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendDecision {
    pub kind: ModelBackendKind,
    pub capabilities: Vec<BackendCapability>,
    pub notes: String,
}

impl BackendDecision {
    pub fn local_bitnet_cpp() -> Self {
        Self {
            kind: ModelBackendKind::ExternalBitnetCpp,
            capabilities: vec![BackendCapability::LocalBitNetInference],
            notes: "Uses Microsoft's BitNet runtime for local single-machine inference.".into(),
        }
    }

    pub fn distributed_reference() -> Self {
        Self {
            kind: ModelBackendKind::DistributedReference,
            capabilities: vec![
                BackendCapability::ShardAwareLayerExecution,
                BackendCapability::DeterministicReference,
            ],
            notes: "Exercises distributed scheduling, sharding, and activation flow without production kernels."
                .into(),
        }
    }

    pub fn rust_bitnet_experimental() -> Self {
        Self {
            kind: ModelBackendKind::RustBitNetExperimental,
            capabilities: vec![
                BackendCapability::ShardAwareLayerExecution,
                BackendCapability::ProductionKernel,
                BackendCapability::RustOnlyCorrectnessGates,
            ],
            notes: "Rust-only BitNet path gated by GGUF parsing, I2_S decode, full local, split local, and gRPC deterministic checks."
                .into(),
        }
    }

    pub fn recommended_for_distributed_bitnet(has_rust_bitnet: bool) -> Self {
        if has_rust_bitnet {
            Self::rust_bitnet_experimental()
        } else {
            Self::distributed_reference()
        }
    }
}
