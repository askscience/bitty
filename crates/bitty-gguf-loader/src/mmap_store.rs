use bitty_model::gguf::{self, GgufFileMetadata, GgufTensorInfo, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("GGUF parse error: {0}")]
    Gguf(#[from] bitty_model::gguf::GgufError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Unsupported GGML type: {0}")]
    UnsupportedGgmlType(u32),
}

pub type Result<T> = std::result::Result<T, LoadError>;

pub struct LoadedModel {
    pub metadata: GgufFileMetadata,
    pub weights: WeightStore,
    pub data_offset: u64,
}

/// Memory-mapped weight store with tensor index.
/// Provides zero-copy access to raw bytes and typed tensor upload.
pub struct WeightStore {
    mmap: memmap2::Mmap,
    pub tensors: Vec<GgufTensorInfo>,
    data_offset: u64,
}

impl WeightStore {
    pub fn new(mmap: memmap2::Mmap, tensors: Vec<GgufTensorInfo>, data_offset: u64) -> Self {
        Self { mmap, tensors, data_offset }
    }

    pub fn mmap_data(&self) -> &[u8] {
        &self.mmap
    }

    pub fn get_raw(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.iter().find(|t| t.name == name)?;
        let start = (self.data_offset + info.offset) as usize;
        let end = (start + info.byte_len as usize).min(self.mmap.len());
        Some(&self.mmap[start..end])
    }

    pub fn get_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    /// Build a raw byte→offset map for direct buffer access (used by WGPU backend).
    pub fn build_offset_map(&self) -> std::collections::HashMap<String, (usize, usize, u32)> {
        self.tensors.iter().map(|info| {
            let start = (self.data_offset + info.offset) as usize;
            let byte_len = info.byte_len as usize;
            (info.name.clone(), (start, byte_len, info.ggml_type))
        }).collect()
    }
}

/// Parse a GGUF file and return the mmap'd data with metadata.
pub fn load_gguf(path: &Path) -> Result<(memmap2::Mmap, GgufFileMetadata, u64)> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gguf = gguf::parse_gguf_bytes(&mmap)?;
    let data_offset = compute_data_offset(&mmap, gguf.alignment);
    Ok((mmap, gguf, data_offset))
}

/// Compute the byte offset where tensor data begins in a GGUF file.
pub fn compute_data_offset(data: &[u8], alignment: u64) -> u64 {
    let mut pos: u64 = 0;
    if data.len() < 24 { return 0; }
    pos += 4; // magic
    pos += 4; // version
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    pos += 16;
    for _ in 0..metadata_count {
        if (pos as usize) + 8 > data.len() { return 0; }
        let key_len = u64::from_le_bytes(data[pos as usize..(pos as usize + 8)].try_into().unwrap());
        pos += 8 + key_len;
        if (pos as usize) + 4 > data.len() { return 0; }
        let val_type = u32::from_le_bytes(data[pos as usize..(pos as usize + 4)].try_into().unwrap());
        pos += 4;
        pos = skip_metadata_value(data, pos, val_type);
    }
    for _ in 0..tensor_count {
        if (pos as usize) + 8 > data.len() { return 0; }
        let name_len = u64::from_le_bytes(data[pos as usize..(pos as usize + 8)].try_into().unwrap());
        pos += 8 + name_len;
        if (pos as usize) + 4 > data.len() { return 0; }
        let dim_count = u32::from_le_bytes(data[pos as usize..(pos as usize + 4)].try_into().unwrap());
        pos += 4;
        pos += dim_count as u64 * 8;
        pos += 4;
        pos += 8;
    }
    let alignment = alignment.max(1);
    ((pos + alignment - 1) / alignment) * alignment
}

fn skip_metadata_value(data: &[u8], mut pos: u64, val_type: u32) -> u64 {
    match val_type {
        0 | 1 => pos + 1,
        2 | 3 => pos + 2,
        4 | 5 => pos + 4,
        6 => pos + 4,
        7 => pos + 1,
        8 => {
            let len = u64::from_le_bytes(data[pos as usize..(pos as usize + 8)].try_into().unwrap());
            pos + 8 + len
        }
        9 => {
            let item_type = u32::from_le_bytes(data[pos as usize..(pos as usize + 4)].try_into().unwrap());
            let len = u64::from_le_bytes(data[(pos as usize + 4)..(pos as usize + 12)].try_into().unwrap());
            pos += 12;
            for _ in 0..len {
                pos = skip_metadata_value(data, pos, item_type);
            }
            pos
        }
        10 | 11 => pos + 8,
        12 => pos + 8,
        _ => pos,
    }
}
