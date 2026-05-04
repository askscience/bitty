use crate::{ActivationTensor, BitNetLogits, GenerateRequest, ProtocolConversionError};

pub const MAX_ACTIVATION_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_ACTIVATION_RANK: usize = 8;
pub const MAX_PROMPT_TOKENS: usize = 32 * 1024;
pub const MAX_PROMPT_BYTES: usize = 256 * 1024;
pub const MAX_LOGITS: usize = 512 * 1024;
pub const MAX_MODEL_PATH_BYTES: usize = 4096;

pub fn validate_activation_tensor(
    activation: &ActivationTensor,
) -> Result<(), ProtocolConversionError> {
    if activation.payload.len() > MAX_ACTIVATION_PAYLOAD_BYTES {
        return Err(ProtocolConversionError::Validation(format!(
            "activation payload exceeds {MAX_ACTIVATION_PAYLOAD_BYTES} bytes"
        )));
    }
    if activation.shape.len() > MAX_ACTIVATION_RANK {
        return Err(ProtocolConversionError::Validation(format!(
            "activation rank exceeds {MAX_ACTIVATION_RANK}"
        )));
    }
    Ok(())
}

pub fn validate_generate_request(request: &GenerateRequest) -> Result<(), ProtocolConversionError> {
    if request.prompt_tokens.len() > MAX_PROMPT_TOKENS {
        return Err(ProtocolConversionError::Validation(format!(
            "prompt token count exceeds {MAX_PROMPT_TOKENS}"
        )));
    }
    if request.prompt.len() > MAX_PROMPT_BYTES {
        return Err(ProtocolConversionError::Validation(format!(
            "prompt exceeds {MAX_PROMPT_BYTES} bytes"
        )));
    }
    Ok(())
}

pub fn validate_logits(logits: &BitNetLogits) -> Result<(), ProtocolConversionError> {
    if logits.logits.len() > MAX_LOGITS {
        return Err(ProtocolConversionError::Validation(format!(
            "logits length exceeds {MAX_LOGITS}"
        )));
    }
    Ok(())
}

pub fn validate_model_path(path: &str) -> Result<(), ProtocolConversionError> {
    if path.len() > MAX_MODEL_PATH_BYTES {
        return Err(ProtocolConversionError::Validation(format!(
            "model path exceeds {MAX_MODEL_PATH_BYTES} bytes"
        )));
    }
    Ok(())
}
