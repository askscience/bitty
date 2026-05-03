use bitty_protocol::{ActivationDType, ActivationTensor, CompressionKind};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecKind {
    Raw,
    Fp8Linear,
    SparseTopK,
    Delta,
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("activation checksum failed")]
    ChecksumFailed,
    #[error("payload length must be even for fp16 data")]
    InvalidFp16Length,
    #[error("sparse top-k payload is malformed")]
    InvalidSparseTopK,
}

#[derive(Clone, Debug)]
pub struct ActivationCodec {
    kind: CodecKind,
}

impl ActivationCodec {
    pub fn new(kind: CodecKind) -> Self {
        Self { kind }
    }

    pub fn encode(&self, activation: &ActivationTensor) -> Result<ActivationTensor, CodecError> {
        if !activation.verify_checksum() {
            return Err(CodecError::ChecksumFailed);
        }

        match self.kind {
            CodecKind::Raw => Ok(activation.clone()),
            CodecKind::Fp8Linear => encode_fp8_linear(activation),
            CodecKind::SparseTopK => encode_sparse_topk(activation),
            CodecKind::Delta => Ok(activation.clone().with_compression(CompressionKind::Delta)),
        }
    }

    pub fn decode(&self, activation: &ActivationTensor) -> Result<ActivationTensor, CodecError> {
        if !activation.verify_checksum() {
            return Err(CodecError::ChecksumFailed);
        }

        match self.kind {
            CodecKind::Raw => Ok(activation.clone()),
            CodecKind::Fp8Linear => decode_fp8_linear(activation),
            CodecKind::SparseTopK => decode_sparse_topk(activation),
            CodecKind::Delta => Ok(activation.clone()),
        }
    }
}

fn encode_fp8_linear(activation: &ActivationTensor) -> Result<ActivationTensor, CodecError> {
    if activation.payload.len() % 2 != 0 {
        return Err(CodecError::InvalidFp16Length);
    }

    let payload = activation
        .payload
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            ((sample / 256) + 128).clamp(0, 255) as u8
        })
        .collect();

    Ok(ActivationTensor::new(
        activation.request_id.clone(),
        activation.token_position,
        activation.source_layer,
        activation.target_layer,
        activation.shape.clone(),
        ActivationDType::Fp8,
        payload,
    )
    .with_compression(CompressionKind::Fp8))
}

fn decode_fp8_linear(activation: &ActivationTensor) -> Result<ActivationTensor, CodecError> {
    let payload = activation
        .payload
        .iter()
        .flat_map(|byte| {
            let sample = ((*byte as i16) - 128) * 256;
            sample.to_le_bytes()
        })
        .collect();

    Ok(ActivationTensor::new(
        activation.request_id.clone(),
        activation.token_position,
        activation.source_layer,
        activation.target_layer,
        activation.shape.clone(),
        ActivationDType::Fp16,
        payload,
    ))
}

fn encode_sparse_topk(activation: &ActivationTensor) -> Result<ActivationTensor, CodecError> {
    let keep = ((activation.payload.len() as f64) * 0.30).ceil().max(1.0) as usize;
    let mut ranked = activation
        .payload
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranked.sort_by_key(|(_, byte)| std::cmp::Reverse(byte.abs_diff(128)));
    ranked.truncate(keep.min(ranked.len()));
    ranked.sort_by_key(|(index, _)| *index);

    let mut payload = Vec::with_capacity(8 + ranked.len() * 5);
    payload.extend_from_slice(&(activation.payload.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(ranked.len() as u32).to_le_bytes());
    for (index, byte) in ranked {
        payload.extend_from_slice(&(index as u32).to_le_bytes());
        payload.push(byte);
    }

    Ok(ActivationTensor::new(
        activation.request_id.clone(),
        activation.token_position,
        activation.source_layer,
        activation.target_layer,
        activation.shape.clone(),
        activation.dtype,
        payload,
    )
    .with_compression(CompressionKind::TopK))
}

fn decode_sparse_topk(activation: &ActivationTensor) -> Result<ActivationTensor, CodecError> {
    if activation.payload.len() < 8 {
        return Err(CodecError::InvalidSparseTopK);
    }
    let original_len = u32::from_le_bytes(activation.payload[0..4].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(activation.payload[4..8].try_into().unwrap()) as usize;
    if activation.payload.len() != 8 + count * 5 {
        return Err(CodecError::InvalidSparseTopK);
    }

    let mut payload = vec![0_u8; original_len];
    for chunk in activation.payload[8..].chunks_exact(5) {
        let index = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as usize;
        let Some(slot) = payload.get_mut(index) else {
            return Err(CodecError::InvalidSparseTopK);
        };
        *slot = chunk[4];
    }

    Ok(ActivationTensor::new(
        activation.request_id.clone(),
        activation.token_position,
        activation.source_layer,
        activation.target_layer,
        activation.shape.clone(),
        activation.dtype,
        payload,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_codec_reduces_payload_size() {
        let activation = ActivationTensor::new(
            "req",
            0,
            0,
            1,
            vec![2],
            ActivationDType::Fp16,
            vec![0, 0, 255, 127],
        );

        let encoded = ActivationCodec::new(CodecKind::Fp8Linear)
            .encode(&activation)
            .unwrap();

        assert_eq!(encoded.dtype, ActivationDType::Fp8);
        assert_eq!(encoded.payload.len(), 2);
        assert!(encoded.verify_checksum());
    }

    #[test]
    fn sparse_topk_round_trip_keeps_largest_entries() {
        let activation = ActivationTensor::new(
            "req",
            0,
            0,
            1,
            vec![10],
            ActivationDType::Fp16,
            vec![128, 1, 127, 255, 128, 2, 3, 4, 5, 6],
        );

        let codec = ActivationCodec::new(CodecKind::SparseTopK);
        let encoded = codec.encode(&activation).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(encoded.compression, CompressionKind::TopK);
        assert_eq!(decoded.payload.len(), activation.payload.len());
        assert!(decoded.payload.contains(&255));
    }
}
