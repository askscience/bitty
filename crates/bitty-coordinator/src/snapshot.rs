use bitty_protocol::{HardwareProfile, LayerAssignment};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorSnapshot {
    pub topology_epoch: String,
    pub profiles: Vec<HardwareProfile>,
    pub assignments: Vec<LayerAssignment>,
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("snapshot serialization failed: {0}")]
    Serialize(#[from] bincode::Error),
}

impl CoordinatorSnapshot {
    pub fn encode(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(bincode::serialize(self)?)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, SnapshotError> {
        Ok(bincode::deserialize(bytes)?)
    }
}
