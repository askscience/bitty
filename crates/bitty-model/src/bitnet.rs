use crate::gguf::{
    parse_gguf_file, GgufError, GgufFileMetadata, GgufMetadataValue, GgufTensorInfo,
};
use bitty_protocol::{AssignedLayerRange, LayerAssignment, LayerMetadata, NodeId, Quantization};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitNetModelFamily {
    BitNetB158,
    OneBit,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BitNetModelMetadata {
    pub family: BitNetModelFamily,
    pub layer_count: u32,
    pub hidden_size: u32,
    pub activation_bytes: u64,
    pub quantization: Quantization,
    pub tokenizer_path: Option<PathBuf>,
    pub tensors: Vec<BitNetTensorMetadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BitNetTensorMetadata {
    pub name: String,
    pub layer_id: Option<u32>,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub byte_len: u64,
    pub offset: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardPlan {
    pub node_id: NodeId,
    pub range: AssignedLayerRange,
    pub tensors: Vec<BitNetTensorMetadata>,
    pub byte_len: u64,
}

#[derive(Debug, Error)]
pub enum BitNetMetadataError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("model has no layer tensors")]
    NoLayerTensors,
}

impl BitNetModelMetadata {
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, BitNetMetadataError> {
        let path = path.as_ref();
        let gguf = parse_gguf_file(path)?;
        let tokenizer_path = path
            .parent()
            .map(|parent| parent.join("tokenizer.json"))
            .filter(|path| path.exists());
        Self::from_gguf(gguf, tokenizer_path)
    }

    pub fn from_gguf(
        gguf: GgufFileMetadata,
        tokenizer_path: Option<PathBuf>,
    ) -> Result<Self, BitNetMetadataError> {
        let tensors = gguf
            .tensors
            .iter()
            .map(BitNetTensorMetadata::from)
            .collect::<Vec<_>>();
        let layer_count = tensors
            .iter()
            .filter_map(|tensor| tensor.layer_id)
            .max()
            .map(|layer_id| layer_id + 1)
            .ok_or(BitNetMetadataError::NoLayerTensors)?;
        let hidden_size = metadata_u64(
            &gguf,
            &["bitnet.embedding_length", "llama.embedding_length"],
        )
        .or_else(|| infer_hidden_size(&gguf.tensors))
        .unwrap_or(0) as u32;
        let architecture =
            metadata_string(&gguf, &["general.architecture"]).unwrap_or_else(|| "unknown".into());

        Ok(Self {
            family: if architecture.contains("bitnet") {
                BitNetModelFamily::BitNetB158
            } else if architecture.contains("onebit") {
                BitNetModelFamily::OneBit
            } else {
                BitNetModelFamily::Unknown(architecture)
            },
            layer_count,
            hidden_size,
            activation_bytes: hidden_size as u64 * 2,
            quantization: Quantization::Bit1,
            tokenizer_path,
            tensors,
        })
    }

    pub fn layer_metadata(&self) -> Vec<LayerMetadata> {
        let mut bytes_by_layer = BTreeMap::<u32, u64>::new();
        for tensor in &self.tensors {
            if let Some(layer_id) = tensor.layer_id {
                *bytes_by_layer.entry(layer_id).or_default() += tensor.byte_len;
            }
        }

        (0..self.layer_count)
            .map(|layer_id| LayerMetadata {
                layer_id,
                weight_bytes: bytes_by_layer.get(&layer_id).copied().unwrap_or_default(),
                activation_bytes: self.activation_bytes,
                estimated_flops: estimate_layer_flops(self.hidden_size),
                precision_critical: layer_id == 0 || layer_id + 1 == self.layer_count,
            })
            .collect()
    }

    pub fn shard_plan(&self, assignment: &LayerAssignment) -> ShardPlan {
        let tensors = self
            .tensors
            .iter()
            .filter(|tensor| {
                tensor
                    .layer_id
                    .is_some_and(|layer_id| assignment.range.contains(layer_id))
            })
            .cloned()
            .collect::<Vec<_>>();
        let byte_len = tensors.iter().map(|tensor| tensor.byte_len).sum();

        ShardPlan {
            node_id: assignment.node_id.clone(),
            range: assignment.range.clone(),
            tensors,
            byte_len,
        }
    }
}

impl From<&GgufTensorInfo> for BitNetTensorMetadata {
    fn from(tensor: &GgufTensorInfo) -> Self {
        Self {
            name: tensor.name.clone(),
            layer_id: tensor.layer_id(),
            dimensions: tensor.dimensions.clone(),
            ggml_type: tensor.ggml_type,
            byte_len: tensor.byte_len,
            offset: tensor.offset,
        }
    }
}

fn metadata_u64(gguf: &GgufFileMetadata, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match gguf.metadata.get(*key)? {
        GgufMetadataValue::U64(value) => Some(*value),
        GgufMetadataValue::I64(value) => Some((*value).try_into().ok()?),
        _ => None,
    })
}

fn metadata_string(gguf: &GgufFileMetadata, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match gguf.metadata.get(*key)? {
        GgufMetadataValue::String(value) => Some(value.clone()),
        _ => None,
    })
}

fn infer_hidden_size(tensors: &[GgufTensorInfo]) -> Option<u64> {
    tensors
        .iter()
        .find(|tensor| tensor.name.contains("token_embd") || tensor.name.contains("embed_tokens"))
        .and_then(|tensor| tensor.dimensions.first().copied())
}

fn estimate_layer_flops(hidden_size: u32) -> f64 {
    let hidden = hidden_size.max(1) as f64;
    hidden * hidden * 12.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFileMetadata, GgufTensorInfo};
    use std::collections::HashMap;

    #[test]
    fn metadata_builds_layer_metadata_and_shards() {
        let gguf = GgufFileMetadata {
            version: 3,
            alignment: 32,
            metadata: HashMap::from([
                (
                    "general.architecture".into(),
                    GgufMetadataValue::String("bitnet-25".into()),
                ),
                (
                    "bitnet.embedding_length".into(),
                    GgufMetadataValue::U64(4096),
                ),
            ]),
            tensors: vec![
                GgufTensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![4096, 4096],
                    ggml_type: 36,
                    offset: 0,
                    byte_len: 128,
                },
                GgufTensorInfo {
                    name: "blk.1.ffn_down.weight".into(),
                    dimensions: vec![4096, 4096],
                    ggml_type: 36,
                    offset: 128,
                    byte_len: 256,
                },
            ],
        };
        let metadata = BitNetModelMetadata::from_gguf(gguf, None).unwrap();

        assert_eq!(metadata.layer_count, 2);
        assert_eq!(metadata.hidden_size, 4096);
        assert_eq!(metadata.layer_metadata()[0].weight_bytes, 128);
    }
}
