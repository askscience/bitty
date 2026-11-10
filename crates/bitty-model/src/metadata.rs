use crate::gguf::{
    parse_gguf_file, GgufError, GgufFileMetadata, GgufMetadataValue, GgufTensorInfo,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K,
};
use bitty_protocol::{AssignedLayerRange, LayerAssignment, LayerMetadata, NodeId, Quantization};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// What a peer needs to know about a GGUF model to plan and serve layer ranges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// `general.architecture` from the GGUF header, e.g. `qwen35moe`.
    pub architecture: String,
    pub layer_count: u32,
    pub hidden_size: u32,
    /// Bytes of one token's hidden state on the wire (fp16).
    pub activation_bytes: u64,
    pub quantization: Quantization,
    pub tokenizer_path: Option<PathBuf>,
    pub tensors: Vec<TensorMetadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorMetadata {
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
    pub tensors: Vec<TensorMetadata>,
    pub byte_len: u64,
}

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("model has no layer tensors")]
    NoLayerTensors,
}

impl ModelMetadata {
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, MetadataError> {
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
    ) -> Result<Self, MetadataError> {
        let tensors = gguf
            .tensors
            .iter()
            .map(TensorMetadata::from)
            .collect::<Vec<_>>();
        let architecture =
            metadata_string(&gguf, &["general.architecture"]).unwrap_or_else(|| "unknown".into());
        // The declared block count comes first: a stage file holds only some of the blocks and
        // a vocabulary file none.
        let block_key = format!("{architecture}.block_count");
        let layer_count = metadata_u64(&gguf, &[block_key.as_str()])
            .map(|count| count as u32)
            .filter(|count| *count > 0)
            .or_else(|| {
                tensors
                    .iter()
                    .filter_map(|tensor| tensor.layer_id)
                    .max()
                    .map(|layer_id| layer_id + 1)
            })
            .ok_or(MetadataError::NoLayerTensors)?;
        let hidden_key = format!("{architecture}.embedding_length");
        let hidden_size = metadata_u64(&gguf, &[hidden_key.as_str(), "llama.embedding_length"])
            .or_else(|| infer_hidden_size(&gguf.tensors))
            .unwrap_or(0) as u32;
        let quantization = dominant_quantization(&tensors);

        Ok(Self {
            architecture,
            layer_count,
            hidden_size,
            activation_bytes: hidden_size as u64 * 2,
            quantization,
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

impl From<&GgufTensorInfo> for TensorMetadata {
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

/// Coarse weight precision of the layer tensors, picked by which GGML type holds most bytes.
/// Types without a matching [`Quantization`] (Q5 and wider) count as fp16, the safe upper bound
/// when the scheduler estimates memory.
fn dominant_quantization(tensors: &[TensorMetadata]) -> Quantization {
    let mut bytes_by_type = BTreeMap::<u32, u64>::new();
    for tensor in tensors.iter().filter(|tensor| tensor.layer_id.is_some()) {
        *bytes_by_type.entry(tensor.ggml_type).or_default() += tensor.byte_len;
    }
    let dominant = bytes_by_type
        .into_iter()
        .max_by_key(|(_, bytes)| *bytes)
        .map(|(ggml_type, _)| ggml_type);
    match dominant {
        Some(GGML_TYPE_Q2_K) => Quantization::Q2,
        Some(GGML_TYPE_Q3_K) => Quantization::Q3,
        Some(GGML_TYPE_Q4_0 | GGML_TYPE_Q4_1 | GGML_TYPE_Q4_K) => Quantization::Q4,
        _ => Quantization::Fp16,
    }
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
                    GgufMetadataValue::String("qwen35moe".into()),
                ),
                (
                    "qwen35moe.embedding_length".into(),
                    GgufMetadataValue::U64(2048),
                ),
            ]),
            tensors: vec![
                GgufTensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![2048, 248320],
                    ggml_type: 8,
                    offset: 0,
                    byte_len: 1024,
                },
                GgufTensorInfo {
                    name: "blk.0.attn_qkv.weight".into(),
                    dimensions: vec![2048, 8192],
                    ggml_type: 8,
                    offset: 1024,
                    byte_len: 128,
                },
                GgufTensorInfo {
                    name: "blk.0.ffn_gate_exps.weight".into(),
                    dimensions: vec![2048, 512, 256],
                    ggml_type: GGML_TYPE_Q4_K,
                    offset: 1152,
                    byte_len: 512,
                },
                GgufTensorInfo {
                    name: "blk.1.ffn_down_exps.weight".into(),
                    dimensions: vec![512, 2048, 256],
                    ggml_type: GGML_TYPE_Q4_K,
                    offset: 1664,
                    byte_len: 256,
                },
            ],
        };
        let metadata = ModelMetadata::from_gguf(gguf, None).unwrap();

        assert_eq!(metadata.architecture, "qwen35moe");
        assert_eq!(metadata.layer_count, 2);
        assert_eq!(metadata.hidden_size, 2048);
        assert_eq!(metadata.quantization, Quantization::Q4);
        assert_eq!(metadata.layer_metadata()[0].weight_bytes, 640);
        assert_eq!(metadata.layer_metadata()[1].weight_bytes, 256);
    }
}
