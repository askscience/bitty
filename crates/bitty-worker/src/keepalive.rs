use bitty_inference::LayerExecutor;
use bitty_protocol::{ActivationDType, ActivationTensor, LayerAssignment};
use std::time::Duration;

pub async fn touch_weights<E: LayerExecutor>(
    executor: &E,
    assignment: &LayerAssignment,
) -> Result<(), bitty_inference::ExecutorError> {
    let dummy = ActivationTensor::new(
        "keepalive",
        0,
        assignment.range.start_layer,
        assignment.range.start_layer,
        vec![1],
        ActivationDType::Fp16,
        vec![0, 0],
    );
    executor.execute_range(&assignment.range, dummy).await?;
    Ok(())
}

pub fn default_keepalive_interval() -> Duration {
    Duration::from_secs(30)
}
