use bitty_inference::{ExecutorError, LayerExecutor};
use bitty_protocol::{ActivationTensor, LayerAssignment, NodeId};
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone)]
pub struct RingWorker<E> {
    pub node_id: NodeId,
    pub assignment: LayerAssignment,
    executor: Arc<E>,
}

impl<E> RingWorker<E>
where
    E: LayerExecutor,
{
    pub fn new(node_id: NodeId, assignment: LayerAssignment, executor: Arc<E>) -> Self {
        Self {
            node_id,
            assignment,
            executor,
        }
    }

    pub async fn forward(
        &self,
        activation: ActivationTensor,
    ) -> Result<ActivationTensor, RingWorkerError> {
        if !activation.verify_checksum() {
            return Err(RingWorkerError::ChecksumFailed);
        }
        Ok(self
            .executor
            .execute_range(&self.assignment.range, activation)
            .await?)
    }
}

#[derive(Debug, Error)]
pub enum RingWorkerError {
    #[error("activation checksum failed")]
    ChecksumFailed,
    #[error(transparent)]
    Executor(#[from] ExecutorError),
}
