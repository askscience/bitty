use bitty_protocol::{AssignedLayerRange, NodeId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightShardManifest {
    pub shard_id: String,
    pub node_id: NodeId,
    pub range: AssignedLayerRange,
    pub byte_len: u64,
    pub sha256_hex: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightShard {
    pub manifest: WeightShardManifest,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum ShardError {
    #[error("weight shard length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: u64, actual: u64 },
    #[error("weight shard checksum mismatch")]
    ChecksumMismatch,
}

impl WeightShard {
    pub fn verify(&self) -> Result<(), ShardError> {
        if self.manifest.byte_len != self.bytes.len() as u64 {
            return Err(ShardError::LengthMismatch {
                expected: self.manifest.byte_len,
                actual: self.bytes.len() as u64,
            });
        }

        let digest = Sha256::digest(&self.bytes);
        let digest_hex = format!("{digest:x}");
        if digest_hex != self.manifest.sha256_hex {
            return Err(ShardError::ChecksumMismatch);
        }

        Ok(())
    }
}
