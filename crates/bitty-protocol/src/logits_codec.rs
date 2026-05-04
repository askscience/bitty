use crate::ProtocolConversionError;

pub fn logits_f32_le_bytes(logits: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(logits.len() * 4);
    for value in logits {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

pub fn logits_from_f32_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, ProtocolConversionError> {
    if bytes.len() % 4 != 0 {
        return Err(ProtocolConversionError::Validation(
            "logits_f32_le length is not a multiple of 4".into(),
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}
