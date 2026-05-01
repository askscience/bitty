use bitty_protocol::{ActivationDType, ActivationTensor};
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
            CodecKind::SparseTopK | CodecKind::Delta => Ok(activation.clone()),
        }
    }

    pub fn decode(&self, activation: &ActivationTensor) -> Result<ActivationTensor, CodecError> {
        if !activation.verify_checksum() {
            return Err(CodecError::ChecksumFailed);
        }

        match self.kind {
            CodecKind::Raw => Ok(activation.clone()),
            CodecKind::Fp8Linear => decode_fp8_linear(activation),
            CodecKind::SparseTopK | CodecKind::Delta => Ok(activation.clone()),
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
    ))
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
}
