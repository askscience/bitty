use bitty_protocol::NodeId;

pub fn record_layer_latency(node_id: &NodeId, layer_start: u32, micros: u64) {
    metrics::histogram!(
        "dlm_layer_latency_us",
        "node_id" => node_id.0.clone(),
        "layer_start" => layer_start.to_string()
    )
    .record(micros as f64);
}

pub fn record_activation_bytes(direction: &'static str, bytes: u64) {
    metrics::counter!("dlm_activation_bytes_total", "direction" => direction).increment(bytes);
}

pub fn record_checksum_failure(node_id: &NodeId) {
    metrics::counter!("dlm_checksum_failures_total", "node_id" => node_id.0.clone()).increment(1);
}

pub fn record_generated_token(node_id: &NodeId) {
    metrics::counter!("dlm_tokens_generated_total", "node_id" => node_id.0.clone()).increment(1);
}
