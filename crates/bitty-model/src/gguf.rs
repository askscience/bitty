use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use thiserror::Error;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGML_TYPE_I2_S: u32 = 36;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GgufFileMetadata {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub byte_len: u64,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn layer_id(&self) -> Option<u32> {
        layer_id_from_tensor_name(&self.name)
    }

    pub fn is_i2_s(&self) -> bool {
        self.ggml_type == GGML_TYPE_I2_S
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GgufMetadataValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    ArrayLen(u64),
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("GGUF I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("invalid GGUF magic")]
    InvalidMagic,
    #[error("unexpected end of GGUF data")]
    UnexpectedEof,
    #[error("unsupported GGUF metadata type {0}")]
    UnsupportedMetadataType(u32),
    #[error("invalid GGUF string")]
    InvalidString,
}

pub fn parse_gguf_file(path: impl AsRef<Path>) -> Result<GgufFileMetadata, GgufError> {
    let bytes = fs::read(path)?;
    parse_gguf_bytes(&bytes)
}

pub fn parse_gguf_bytes(bytes: &[u8]) -> Result<GgufFileMetadata, GgufError> {
    let mut reader = Reader::new(bytes);
    if reader.read_bytes(4)? != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = reader.read_u32()?;
    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;
    let mut metadata = HashMap::new();
    let mut alignment = 32_u64;

    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value_type = reader.read_u32()?;
        let value = reader.read_metadata_value(value_type)?;
        if key == "general.alignment" {
            alignment = match value {
                GgufMetadataValue::U64(value) => value,
                GgufMetadataValue::I64(value) => value.max(1) as u64,
                _ => alignment,
            };
        }
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let dimension_count = reader.read_u32()?;
        let dimensions = (0..dimension_count)
            .map(|_| reader.read_u64())
            .collect::<Result<Vec<_>, _>>()?;
        let ggml_type = reader.read_u32()?;
        let offset = reader.read_u64()?;
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type,
            offset,
            byte_len: 0,
        });
    }

    let data_start = align_to(reader.offset() as u64, alignment);
    for index in 0..tensors.len() {
        let start = data_start + tensors[index].offset;
        let end = tensors
            .get(index + 1)
            .map(|next| data_start + next.offset)
            .unwrap_or(bytes.len() as u64);
        tensors[index].byte_len = end.saturating_sub(start);
    }

    Ok(GgufFileMetadata {
        version,
        alignment,
        metadata,
        tensors,
    })
}

pub fn decode_i2_s_block(block: &[u8; 32]) -> [i8; 128] {
    let mut decoded = [0_i8; 128];
    for gp in 0..32 {
        let byte = block[gp];
        for group in 0..4 {
            let shift = 6 - 2 * group;
            let code = (byte >> shift) & 0b11;
            decoded[group * 32 + gp] = match code {
                0b00 => 0,
                0b01 => 1,
                0b10 => -1,
                _ => 0,
            };
        }
    }
    decoded
}

pub fn layer_id_from_tensor_name(name: &str) -> Option<u32> {
    ["blk.", "layers.", "model.layers."]
        .iter()
        .find_map(|prefix| {
            let start = name.find(prefix)? + prefix.len();
            let digits = name[start..]
                .chars()
                .take_while(|ch| ch.is_ascii_digit())
                .collect::<String>();
            (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
        })
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(GgufError::UnexpectedEof)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(GgufError::UnexpectedEof)?;
        self.offset = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| GgufError::InvalidString)
    }

    fn read_metadata_value(&mut self, value_type: u32) -> Result<GgufMetadataValue, GgufError> {
        match value_type {
            0 => Ok(GgufMetadataValue::U64(self.read_u8()? as u64)),
            1 => Ok(GgufMetadataValue::I64(self.read_u8()? as i8 as i64)),
            2 => Ok(GgufMetadataValue::U64(self.read_bytes(2).map(|bytes| {
                u16::from_le_bytes(bytes.try_into().unwrap()) as u64
            })?)),
            3 => Ok(GgufMetadataValue::I64(self.read_bytes(2).map(|bytes| {
                i16::from_le_bytes(bytes.try_into().unwrap()) as i64
            })?)),
            4 => Ok(GgufMetadataValue::U64(self.read_u32()? as u64)),
            5 => Ok(GgufMetadataValue::I64(self.read_i32()? as i64)),
            6 => Ok(GgufMetadataValue::F64(self.read_f32()? as f64)),
            7 => Ok(GgufMetadataValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufMetadataValue::String(self.read_string()?)),
            9 => {
                let item_type = self.read_u32()?;
                let len = self.read_u64()?;
                for _ in 0..len {
                    self.skip_metadata_value(item_type)?;
                }
                Ok(GgufMetadataValue::ArrayLen(len))
            }
            10 => Ok(GgufMetadataValue::U64(self.read_u64()?)),
            11 => Ok(GgufMetadataValue::I64(self.read_i64()?)),
            12 => Ok(GgufMetadataValue::F64(self.read_f64()?)),
            other => Err(GgufError::UnsupportedMetadataType(other)),
        }
    }

    fn skip_metadata_value(&mut self, value_type: u32) -> Result<(), GgufError> {
        let _ = self.read_metadata_value(value_type)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i2_s_decodes_interleaved_groups() {
        let mut block = [0_u8; 32];
        block[0] = 0b01_10_00_01;
        let decoded = decode_i2_s_block(&block);

        assert_eq!(decoded[0], 1);
        assert_eq!(decoded[32], -1);
        assert_eq!(decoded[64], 0);
        assert_eq!(decoded[96], 1);
    }

    #[test]
    fn layer_id_parses_common_tensor_names() {
        assert_eq!(layer_id_from_tensor_name("blk.12.attn_q.weight"), Some(12));
        assert_eq!(
            layer_id_from_tensor_name("model.layers.7.mlp.down_proj.weight"),
            Some(7)
        );
        assert_eq!(layer_id_from_tensor_name("token_embd.weight"), None);
    }
}
