use crate::gguf::{
    parse_gguf_file, quantization_from_ggml_type, GgufError, GgufFileMetadata, GgufMetadataValue,
    GgufTensorInfo,
};
use bitty_protocol::{AssignedLayerRange, LayerAssignment, LayerMetadata, NodeId, Quantization};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    BitNetB158,
    OneBit,
    Llama,
    Mistral,
    Phi,
    Qwen2,
    Gemma,
    Gemma2,
    Falcon,
    StableLM,
    DeepSeek,
    Mamba,
    Unknown(String),
}

impl ModelArchitecture {
    pub fn as_str(&self) -> &str {
        match self {
            Self::BitNetB158 => "bitnet",
            Self::OneBit => "onebit",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Phi => "phi",
            Self::Qwen2 => "qwen2",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Falcon => "falcon",
            Self::StableLM => "stablelm",
            Self::DeepSeek => "deepseek",
            Self::Mamba => "mamba",
            Self::Unknown(s) => s.as_str(),
        }
    }

    pub fn hidden_size_key(&self) -> &[&str] {
        match self {
            Self::Llama => &["llama.embedding_length"],
            Self::Mistral => &["mistral.dim"],
            Self::Phi => &["phi.hidden_size"],
            Self::Qwen2 => &["qwen2.embedding_length"],
            Self::Gemma | Self::Gemma2 => &["gemma.embedding_length"],
            Self::Falcon => &["falcon.hidden_size"],
            Self::StableLM => &["stablelm.embedding_length"],
            Self::DeepSeek => &["deepseek.embedding_length"],
            Self::Mamba => &["mamba.embedding_length"],
            _ => &["bitnet.embedding_length", "llama.embedding_length"],
        }
    }

    pub fn num_attention_heads_key(&self) -> Option<&str> {
        match self {
            Self::Llama => Some("llama.attention.head_count"),
            Self::Mistral => Some("mistral.attention.head_count"),
            Self::Phi => Some("phi.num_attention_heads"),
            Self::Qwen2 => Some("qwen2.attention.head_count"),
            Self::Gemma | Self::Gemma2 => Some("gemma.attention.head_count"),
            Self::Falcon => Some("falcon.num_attention_heads"),
            Self::StableLM => Some("stablelm.attention.head_count"),
            Self::DeepSeek => Some("deepseek.attention.head_count"),
            _ => None,
        }
    }

    pub fn rope_dimension_key(&self) -> Option<&str> {
        match self {
            Self::Llama => Some("llama.rope.dimension_count"),
            Self::Mistral => Some("mistral.rope.dimension_count"),
            Self::Qwen2 => Some("qwen2.rope.dimension_count"),
            Self::Gemma | Self::Gemma2 => Some("gemma.rope.dimension_count"),
            _ => None,
        }
    }
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

pub fn classify_architecture(arch_str: &str) -> ModelArchitecture {
    let lower = arch_str.to_lowercase();
    if lower.contains("bitnet") || lower.contains("bitlinear") {
        ModelArchitecture::BitNetB158
    } else if lower.contains("onebit") {
        ModelArchitecture::OneBit
    } else if lower.contains("llama") || lower.contains("llama-") {
        ModelArchitecture::Llama
    } else if lower.contains("mistral") {
        ModelArchitecture::Mistral
    } else if lower.contains("phi-3") || lower.contains("phi3") || lower.contains("phi-") {
        ModelArchitecture::Phi
    } else if lower.contains("qwen2")
        || lower.contains("qwen-2")
        || lower.contains("qwen3")
        || lower.contains("qwen-3")
    {
        ModelArchitecture::Qwen2
    } else if lower.contains("gemma-2") || lower.contains("gemma2") {
        ModelArchitecture::Gemma2
    } else if lower.contains("gemma") {
        ModelArchitecture::Gemma
    } else if lower.contains("falcon") {
        ModelArchitecture::Falcon
    } else if lower.contains("stablelm") || lower.contains("stable-lm") {
        ModelArchitecture::StableLM
    } else if lower.contains("deepseek") {
        ModelArchitecture::DeepSeek
    } else if lower.contains("mamba") {
        ModelArchitecture::Mamba
    } else {
        ModelArchitecture::Unknown(arch_str.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture: ModelArchitecture,
    pub layer_count: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub activation_bytes: u64,
    pub quantization: Quantization,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_dimension_count: u32,
    pub tokenizer_path: Option<PathBuf>,
    pub tensors: Vec<ModelTensorMetadata>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelTensorMetadata {
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
    pub tensors: Vec<ModelTensorMetadata>,
    pub byte_len: u64,
}

#[derive(Debug, Error)]
pub enum ModelMetadataError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("model has no layer tensors")]
    NoLayerTensors,
}

impl ModelMetadata {
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, ModelMetadataError> {
        let path = path.as_ref();
        let gguf = parse_gguf_file(path)?;
        let tokenizer_path = path
            .parent()
            .map(|parent| parent.join("tokenizer.json"))
            .filter(|p| p.exists());
        Self::from_gguf(gguf, tokenizer_path)
    }

    pub fn from_gguf(
        gguf: GgufFileMetadata,
        tokenizer_path: Option<PathBuf>,
    ) -> Result<Self, ModelMetadataError> {
        let tensors = gguf
            .tensors
            .iter()
            .map(ModelTensorMetadata::from)
            .collect::<Vec<_>>();

        let layer_count = tensors
            .iter()
            .filter_map(|tensor| tensor.layer_id)
            .max()
            .map(|layer_id| layer_id + 1)
            .unwrap_or(1);

        if layer_count == 1 && tensors.iter().all(|t| t.layer_id.is_none()) {
            return Err(ModelMetadataError::NoLayerTensors);
        }

        let arch_str =
            metadata_string(&gguf, &["general.architecture"]).unwrap_or_else(|| "unknown".into());
        let architecture = classify_architecture(&arch_str);

        let hidden_size = metadata_u64(&gguf, architecture.hidden_size_key())
            .or_else(|| infer_hidden_size(&gguf.tensors))
            .unwrap_or(0) as u32;

        let intermediate_size = metadata_u64(
            &gguf,
            &[
                "llama.feed_forward_length",
                "mistral.intermediate_size",
                "phi.intermediate_size",
            ],
        )
        .unwrap_or(hidden_size as u64 * 8 / 3) as u32;

        let num_attention_heads = architecture
            .num_attention_heads_key()
            .and_then(|key| metadata_u64(&gguf, &[key]))
            .unwrap_or(1) as u32;

        let num_key_value_heads = metadata_u64(
            &gguf,
            &[
                "llama.attention.key_value_head_count",
                "mistral.attention.key_value_head_count",
                "phi.num_key_value_heads",
            ],
        )
        .unwrap_or(num_attention_heads as u64) as u32;

        let rope_dimension_count = architecture
            .rope_dimension_key()
            .and_then(|key| metadata_u64(&gguf, &[key]))
            .unwrap_or(hidden_size as u64 / num_attention_heads as u64)
            as u32;

        let vocab_size = metadata_u64(&gguf, &["llama.vocab_size", "tokenizer.ggml.tokens"])
            .unwrap_or(32_000) as u32;

        let context_length = metadata_u64(
            &gguf,
            &[
                "llama.context_length",
                "mistral.context_length",
                "phi.context_length",
            ],
        )
        .unwrap_or(2048) as u32;

        let quantization = tensors
            .iter()
            .map(|tensor| quantization_from_ggml_type(tensor.ggml_type))
            .max_by(|a, b| a.bytes_per_weight().total_cmp(&b.bytes_per_weight()))
            .unwrap_or(Quantization::Fp16);

        let activation_bytes = hidden_size as u64 * quantization.bytes_per_weight() as u64;

        Ok(Self {
            architecture,
            layer_count,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            activation_bytes,
            quantization,
            vocab_size,
            max_seq_len: context_length,
            rope_dimension_count,
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

impl From<&GgufTensorInfo> for ModelTensorMetadata {
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
    use crate::gguf::GgufFileMetadata;
    use std::collections::HashMap;

    fn make_gguf(arch: &str) -> GgufFileMetadata {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String(arch.into()),
        );
        GgufFileMetadata {
            version: 3,
            alignment: 32,
            metadata,
            tensors: vec![
                GgufTensorInfo {
                    name: "token_embd.weight".into(),
                    dimensions: vec![4096, 32000],
                    ggml_type: 1,
                    offset: 0,
                    byte_len: 1024,
                },
                GgufTensorInfo {
                    name: "blk.0.attn_q.weight".into(),
                    dimensions: vec![4096, 4096],
                    ggml_type: 10,
                    offset: 1024,
                    byte_len: 256,
                },
                GgufTensorInfo {
                    name: "blk.1.ffn_down.weight".into(),
                    dimensions: vec![4096, 4096],
                    ggml_type: 10,
                    offset: 1280,
                    byte_len: 256,
                },
            ],
        }
    }

    #[test]
    fn classifies_bitnet_architecture() {
        let m = ModelMetadata::from_gguf(make_gguf("bitnet-25"), None).unwrap();
        assert_eq!(m.architecture, ModelArchitecture::BitNetB158);
        assert_eq!(m.layer_count, 2);
        assert_eq!(m.hidden_size, 4096);
    }

    #[test]
    fn classifies_llama_architecture() {
        let mut gguf = make_gguf("llama");
        gguf.metadata.insert(
            "llama.embedding_length".into(),
            GgufMetadataValue::U64(4096),
        );
        let m = ModelMetadata::from_gguf(gguf, None).unwrap();
        assert_eq!(m.architecture, ModelArchitecture::Llama);
    }

    #[test]
    fn classifies_mistral_architecture() {
        let mut gguf = make_gguf("mistral");
        gguf.metadata
            .insert("mistral.dim".into(), GgufMetadataValue::U64(4096));
        let m = ModelMetadata::from_gguf(gguf, None).unwrap();
        assert_eq!(m.architecture, ModelArchitecture::Mistral);
    }

    #[test]
    fn unknown_architecture_is_preserved() {
        let m = ModelMetadata::from_gguf(make_gguf("custom-transformer"), None).unwrap();
        match m.architecture {
            ModelArchitecture::Unknown(ref s) => assert_eq!(s, "custom-transformer"),
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn quantization_derived_from_tensors() {
        let mut gguf = make_gguf("llama");
        gguf.metadata.insert(
            "llama.embedding_length".into(),
            GgufMetadataValue::U64(4096),
        );
        gguf.tensors[0].ggml_type = 1;
        gguf.tensors[1].ggml_type = 10;
        let m = ModelMetadata::from_gguf(gguf, None).unwrap();
        assert_eq!(m.quantization, Quantization::Fp16);
    }

    #[test]
    fn quantization_all_q2_tensors_yields_q2() {
        let mut gguf = make_gguf("llama");
        gguf.metadata.insert(
            "llama.embedding_length".into(),
            GgufMetadataValue::U64(4096),
        );
        gguf.tensors[0].ggml_type = 10;
        gguf.tensors[1].ggml_type = 10;
        gguf.tensors[2].ggml_type = 10;
        let m = ModelMetadata::from_gguf(gguf, None).unwrap();
        assert_eq!(m.quantization, Quantization::Q2);
    }

    #[test]
    fn quantization_mixed_tensors_yields_highest_precision() {
        let mut gguf = make_gguf("llama");
        gguf.metadata.insert(
            "llama.embedding_length".into(),
            GgufMetadataValue::U64(4096),
        );
        gguf.tensors[0].ggml_type = 10;
        gguf.tensors[1].ggml_type = 10;
        gguf.tensors[2].ggml_type = 12;
        let m = ModelMetadata::from_gguf(gguf, None).unwrap();
        assert_eq!(m.quantization, Quantization::Q4);
    }
}
