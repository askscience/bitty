use bitty_protocol::{AssignedLayerRange, NodeId};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
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

#[derive(Debug)]
pub struct MappedWeightShard {
    pub manifest: WeightShardManifest,
    mmap: Mmap,
}

impl MappedWeightShard {
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    pub fn verify(&self) -> Result<(), ShardError> {
        verify_bytes(&self.manifest, self.bytes())
    }
}

#[derive(Debug, Error)]
pub enum ShardError {
    #[error("weight shard I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("weight shard length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: u64, actual: u64 },
    #[error("weight shard checksum mismatch")]
    ChecksumMismatch,
}

impl WeightShardManifest {
    pub fn for_file(
        shard_id: impl Into<String>,
        node_id: NodeId,
        range: AssignedLayerRange,
        path: impl AsRef<Path>,
    ) -> Result<Self, ShardError> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0_u8; 64 * 1024];
        let mut byte_len = 0_u64;
        loop {
            let read = file.read(&mut buf)?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
            byte_len += read as u64;
        }

        Ok(Self {
            shard_id: shard_id.into(),
            node_id,
            range,
            byte_len,
            sha256_hex: format!("{:x}", hasher.finalize()),
        })
    }
}

impl WeightShard {
    pub fn verify(&self) -> Result<(), ShardError> {
        verify_bytes(&self.manifest, &self.bytes)
    }

    pub fn from_file(
        manifest: WeightShardManifest,
        path: impl AsRef<Path>,
    ) -> Result<Self, ShardError> {
        let bytes = std::fs::read(path)?;
        let shard = Self { manifest, bytes };
        shard.verify()?;
        Ok(shard)
    }

    pub fn mmap(
        manifest: WeightShardManifest,
        path: impl AsRef<Path>,
    ) -> Result<MappedWeightShard, ShardError> {
        let file = File::open(path)?;
        // SAFETY: the file is mapped read-only and the returned Mmap owns the mapping.
        let mmap = unsafe { Mmap::map(&file)? };
        let shard = MappedWeightShard { manifest, mmap };
        shard.verify()?;
        Ok(shard)
    }
}

fn verify_bytes(manifest: &WeightShardManifest, bytes: &[u8]) -> Result<(), ShardError> {
    if manifest.byte_len != bytes.len() as u64 {
        return Err(ShardError::LengthMismatch {
            expected: manifest.byte_len,
            actual: bytes.len() as u64,
        });
    }

    let digest = Sha256::digest(bytes);
    let digest_hex = format!("{digest:x}");
    if digest_hex != manifest.sha256_hex {
        return Err(ShardError::ChecksumMismatch);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_protocol::Quantization;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn manifest_verifies_mmap_file() {
        let path = std::env::temp_dir().join(format!(
            "bitty-shard-{}.bin",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, [1_u8, 2, 3, 4]).unwrap();

        let range = AssignedLayerRange {
            start_layer: 0,
            end_layer_exclusive: 1,
            quantization: Quantization::Bit1,
        };
        let manifest =
            WeightShardManifest::for_file("shard-0", NodeId::new("node-0"), range, &path).unwrap();
        let mapped = WeightShard::mmap(manifest, &path).unwrap();

        assert_eq!(mapped.bytes(), &[1, 2, 3, 4]);
        std::fs::remove_file(path).unwrap();
    }
}
