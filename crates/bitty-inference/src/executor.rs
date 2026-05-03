use async_trait::async_trait;
use bitty_protocol::{ActivationDType, ActivationTensor, AssignedLayerRange, BitNetLogits};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("activation checksum failed")]
    ChecksumFailed,
    #[error("execution failed: {0}")]
    Failed(String),
}

#[async_trait]
pub trait LayerExecutor: Send + Sync {
    async fn execute_range(
        &self,
        range: &AssignedLayerRange,
        activation: ActivationTensor,
    ) -> Result<ActivationTensor, ExecutorError>;

    async fn final_logits(
        &self,
        activation: ActivationTensor,
    ) -> Result<BitNetLogits, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }
        let token = activation
            .payload
            .chunks_exact(4)
            .last()
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .unwrap_or_default();
        let mut logits = vec![f32::NEG_INFINITY; 32_000];
        let index = (token as usize + 1) % logits.len();
        logits[index] = 0.0;
        Ok(BitNetLogits::new(
            activation.request_id,
            activation.token_position,
            logits,
        ))
    }
}

#[async_trait]
pub trait DraftExecutor: Send + Sync {
    async fn propose_tokens(
        &self,
        request_id: &str,
        prefix: &[u32],
        max_tokens: usize,
    ) -> Result<Vec<u32>, ExecutorError>;
}

#[derive(Clone, Debug, Default)]
pub struct FakeLayerExecutor;

#[derive(Clone, Debug, Default)]
pub struct LowBitReferenceExecutor;

#[async_trait]
impl LayerExecutor for FakeLayerExecutor {
    async fn execute_range(
        &self,
        range: &AssignedLayerRange,
        activation: ActivationTensor,
    ) -> Result<ActivationTensor, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }

        let mut payload = activation.payload;
        let delta = range.len() as u8;
        for byte in &mut payload {
            *byte = byte.wrapping_add(delta);
        }

        Ok(ActivationTensor::new(
            activation.request_id,
            activation.token_position,
            range.start_layer,
            range.end_layer_exclusive,
            activation.shape,
            ActivationDType::Fp16,
            payload,
        ))
    }
}

#[async_trait]
impl LayerExecutor for LowBitReferenceExecutor {
    async fn execute_range(
        &self,
        range: &AssignedLayerRange,
        activation: ActivationTensor,
    ) -> Result<ActivationTensor, ExecutorError> {
        if !activation.verify_checksum() {
            return Err(ExecutorError::ChecksumFailed);
        }

        let scale = range.range_scale();
        let payload = activation
            .payload
            .into_iter()
            .enumerate()
            .map(|(index, byte)| byte.wrapping_add(scale.wrapping_mul(index as u8 + 1)))
            .collect();

        Ok(ActivationTensor::new(
            activation.request_id,
            activation.token_position,
            range.start_layer,
            range.end_layer_exclusive,
            activation.shape,
            activation.dtype,
            payload,
        ))
    }
}

#[async_trait]
impl DraftExecutor for FakeLayerExecutor {
    async fn propose_tokens(
        &self,
        _request_id: &str,
        prefix: &[u32],
        max_tokens: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        let seed = prefix.last().copied().unwrap_or(0);
        Ok((0..max_tokens)
            .map(|offset| seed + offset as u32 + 1)
            .collect())
    }
}

trait RangeScale {
    fn range_scale(&self) -> u8;
}

impl RangeScale for AssignedLayerRange {
    fn range_scale(&self) -> u8 {
        let quant_scale = self.quantization.bytes_per_weight().max(0.125) * 8.0;
        (self.len() as f64 * quant_scale).round().max(1.0) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_protocol::Quantization;

    #[tokio::test]
    async fn fake_executor_advances_activation() {
        let range = AssignedLayerRange {
            start_layer: 0,
            end_layer_exclusive: 2,
            quantization: Quantization::Bit1,
        };
        let activation =
            ActivationTensor::new("req", 0, 0, 0, vec![2], ActivationDType::Fp16, vec![1, 2]);
        let output = FakeLayerExecutor
            .execute_range(&range, activation)
            .await
            .unwrap();

        assert_eq!(output.target_layer, 2);
        assert_eq!(output.payload, vec![3, 4]);
    }
}
