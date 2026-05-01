use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use thiserror::Error;

pub const LAYER_LATENCY_US: &str = "dlm_layer_latency_us";
pub const ACTIVATION_BYTES_TOTAL: &str = "dlm_activation_bytes_total";
pub const CHECKSUM_FAILURES_TOTAL: &str = "dlm_checksum_failures_total";
pub const TOKENS_GENERATED_TOTAL: &str = "dlm_tokens_generated_total";
pub const HALDA_RUNS_TOTAL: &str = "dlm_halda_runs_total";
pub const HALDA_DURATION_MS: &str = "dlm_halda_duration_ms";

#[derive(Debug, Error)]
pub enum ObservabilityError {
    #[error("failed to install prometheus recorder: {0}")]
    Prometheus(#[from] metrics_exporter_prometheus::BuildError),
}

pub fn install_prometheus_recorder() -> Result<PrometheusHandle, ObservabilityError> {
    Ok(PrometheusBuilder::new().install_recorder()?)
}

pub fn record_halda_run(duration_ms: f64) {
    metrics::counter!(HALDA_RUNS_TOTAL).increment(1);
    metrics::histogram!(HALDA_DURATION_MS).record(duration_ms);
}

pub fn init_tracing() {
    tracing::debug!("bitty tracing initialized by host application");
}
