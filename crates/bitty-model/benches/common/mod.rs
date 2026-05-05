#![allow(dead_code, unused_imports)]

use bitty_model::gguf::{GgufFileMetadata, GgufMetadataValue, GgufTensorInfo};
use std::collections::HashMap;

pub fn make_base_gguf(architecture: &str, version: u32, alignment: u64) -> GgufFileMetadata {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".into(),
        GgufMetadataValue::String(architecture.into()),
    );
    GgufFileMetadata {
        version,
        alignment,
        metadata,
        tensors: Vec::new(),
    }
}

pub fn make_small_gguf(architecture: &str, num_layers: u32, hidden_size: u64) -> GgufFileMetadata {
    let mut gguf = make_base_gguf(architecture, 3, 32);
    gguf.metadata.insert(
        "llama.embedding_length".into(),
        GgufMetadataValue::U64(hidden_size),
    );
    gguf.metadata.insert(
        "llama.vocab_size".into(),
        GgufMetadataValue::U64(32000),
    );
    gguf.metadata.insert(
        "llama.attention.head_count".into(),
        GgufMetadataValue::U64(32),
    );
    gguf.metadata.insert(
        "llama.context_length".into(),
        GgufMetadataValue::U64(4096),
    );

    let vocab_size = 32000;

    let mut offset = 0u64;
    gguf.tensors.push(GgufTensorInfo {
        name: "token_embd.weight".into(),
        dimensions: vec![hidden_size, vocab_size],
        ggml_type: 1,
        offset,
        byte_len: hidden_size * vocab_size * 2,
    });
    offset += gguf.tensors.last().unwrap().byte_len;

    for layer_id in 0..num_layers {
        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.attn_q.weight"),
            dimensions: vec![hidden_size, hidden_size],
            ggml_type: 10,
            offset,
            byte_len: hidden_size * hidden_size / 4,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.attn_k.weight"),
            dimensions: vec![hidden_size, hidden_size],
            ggml_type: 10,
            offset,
            byte_len: hidden_size * hidden_size / 4,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.attn_v.weight"),
            dimensions: vec![hidden_size, hidden_size],
            ggml_type: 10,
            offset,
            byte_len: hidden_size * hidden_size / 4,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.attn_output.weight"),
            dimensions: vec![hidden_size, hidden_size],
            ggml_type: 10,
            offset,
            byte_len: hidden_size * hidden_size / 4,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.attn_norm.weight"),
            dimensions: vec![hidden_size],
            ggml_type: 1,
            offset,
            byte_len: hidden_size * 2,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.ffn_gate.weight"),
            dimensions: vec![hidden_size * 8 / 3, hidden_size],
            ggml_type: 10,
            offset,
            byte_len: hidden_size * hidden_size / 4,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.ffn_up.weight"),
            dimensions: vec![hidden_size * 8 / 3, hidden_size],
            ggml_type: 10,
            offset,
            byte_len: hidden_size * hidden_size / 4,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.ffn_down.weight"),
            dimensions: vec![hidden_size, hidden_size * 8 / 3],
            ggml_type: 10,
            offset,
            byte_len: hidden_size * hidden_size / 4,
        });
        offset += gguf.tensors.last().unwrap().byte_len;

        gguf.tensors.push(GgufTensorInfo {
            name: format!("blk.{layer_id}.ffn_norm.weight"),
            dimensions: vec![hidden_size],
            ggml_type: 1,
            offset,
            byte_len: hidden_size * 2,
        });
        offset += gguf.tensors.last().unwrap().byte_len;
    }

    gguf.tensors.push(GgufTensorInfo {
        name: "output_norm.weight".into(),
        dimensions: vec![hidden_size],
        ggml_type: 1,
        offset,
        byte_len: hidden_size * 2,
    });
    offset += gguf.tensors.last().unwrap().byte_len;

    gguf.tensors.push(GgufTensorInfo {
        name: "output.weight".into(),
        dimensions: vec![vocab_size, hidden_size],
        ggml_type: 1,
        offset,
        byte_len: vocab_size * hidden_size * 2,
    });

    gguf
}

pub fn make_many_tensors(num_layers: u32, tensors_per_layer: u32) -> GgufFileMetadata {
    let mut gguf = make_small_gguf("llama", num_layers, 4096);
    gguf.tensors.clear();
    let mut offset = 0u64;
    for layer_id in 0..num_layers {
        for t in 0..tensors_per_layer {
            gguf.tensors.push(GgufTensorInfo {
                name: format!("blk.{layer_id}.tensor_{t}.weight"),
                dimensions: vec![1024, 1024],
                ggml_type: 10,
                offset,
                byte_len: 1024 * 1024 / 4,
            });
            offset += 1024 * 1024 / 4;
        }
    }
    gguf
}

pub fn serialize_gguf(gguf: &GgufFileMetadata) -> Vec<u8> {
    use std::io::Write;
    let mut out = Vec::new();

    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&gguf.version.to_le_bytes());
    out.extend_from_slice(&(gguf.tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(gguf.metadata.len() as u64).to_le_bytes());

    for (key, value) in &gguf.metadata {
        let key_bytes = key.as_bytes();
        out.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(key_bytes);

        match value {
            GgufMetadataValue::U64(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetadataValue::I64(v) => {
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufMetadataValue::F64(v) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&f64::to_le_bytes(*v));
            }
            GgufMetadataValue::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(if *v { 1 } else { 0 });
            }
            GgufMetadataValue::String(s) => {
                let s_bytes = s.as_bytes();
                out.extend_from_slice(&8u32.to_le_bytes());
                out.extend_from_slice(&(s_bytes.len() as u64).to_le_bytes());
                out.extend_from_slice(s_bytes);
            }
            GgufMetadataValue::ArrayLen(_) => {}
        }
    }

    for tensor in &gguf.tensors {
        let name_bytes = tensor.name.as_bytes();
        out.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dim in &tensor.dimensions {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&tensor.offset.to_le_bytes());
    }

    let data_start = ((out.len() as u64 + gguf.alignment - 1) / gguf.alignment) * gguf.alignment;
    while out.len() < data_start as usize {
        out.push(0);
    }

    for tensor in &gguf.tensors {
        out.write_all(&vec![0u8; tensor.byte_len as usize]).unwrap();
    }

    out
}

pub const ARCHITECTURES: &[&str] = &[
    "llama",
    "mistral",
    "phi-3",
    "qwen2",
    "gemma",
    "gemma2",
    "falcon",
    "stablelm-3b",
    "deepseek",
    "mamba",
    "bitnet-25",
    "onebit-7b",
    "custom-transformer",
    "unknown-arch",
];
