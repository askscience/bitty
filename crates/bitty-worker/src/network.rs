use crate::{metrics, RingWorker};
use bitty_inference::{FakeLayerExecutor, LayerExecutor};
use bitty_protocol::pb::worker_service_server::{WorkerService, WorkerServiceServer};
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, BitNetLogits as ProtoBitNetLogits, CleanupRequest,
    CleanupResponse, HeartbeatResponse, LoadShardRequest, LoadShardResponse, SampleTokenRequest,
    TokenOutput as ProtoTokenOutput, TopologyUpdate,
};
use bitty_protocol::security::{AuthMode, BITTY_TOKEN_HEADER};
use bitty_protocol::validation::MAX_ACTIVATION_PAYLOAD_BYTES;
use bitty_protocol::{ActivationTensor, LayerAssignment, ModelStage, NodeId, ShardManifestMessage};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub struct NetworkWorker<E = FakeLayerExecutor> {
    node_id: NodeId,
    assignment: Arc<Mutex<Option<LayerAssignment>>>,
    executor: Arc<E>,
    topology_epoch: Arc<Mutex<String>>,
    loaded_shard: Arc<Mutex<Option<ShardManifestMessage>>>,
    stats: RuntimeStats,
    auth_mode: AuthMode,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeStats {
    inner: Arc<Mutex<RuntimeStatsInner>>,
}

#[derive(Clone, Debug, Default)]
struct RuntimeStatsInner {
    forward_count: u64,
    generated_tokens: u64,
    activation_bytes: u64,
    forward_latency_us: u128,
    started: Option<Instant>,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeStatsSnapshot {
    pub observed_tokens_per_second: f64,
    pub avg_forward_latency_ms: f64,
    pub activation_bytes_per_second: u64,
}

impl RuntimeStats {
    async fn record_forward(&self, latency_us: u128, rx_bytes: u64, tx_bytes: u64) {
        let mut inner = self.inner.lock().await;
        if inner.started.is_none() {
            inner.started = Some(Instant::now());
        }
        inner.forward_count += 1;
        inner.activation_bytes = inner.activation_bytes.saturating_add(rx_bytes + tx_bytes);
        inner.forward_latency_us = inner.forward_latency_us.saturating_add(latency_us);
    }

    async fn record_token(&self) {
        let mut inner = self.inner.lock().await;
        if inner.started.is_none() {
            inner.started = Some(Instant::now());
        }
        inner.generated_tokens += 1;
    }

    pub async fn snapshot(&self) -> RuntimeStatsSnapshot {
        let inner = self.inner.lock().await;
        let elapsed = inner
            .started
            .map(|started| started.elapsed().as_secs_f64().max(0.001))
            .unwrap_or(0.001);
        RuntimeStatsSnapshot {
            observed_tokens_per_second: inner.generated_tokens as f64 / elapsed,
            avg_forward_latency_ms: if inner.forward_count == 0 {
                0.0
            } else {
                inner.forward_latency_us as f64 / inner.forward_count as f64 / 1000.0
            },
            activation_bytes_per_second: (inner.activation_bytes as f64 / elapsed) as u64,
        }
    }
}

impl NetworkWorker<FakeLayerExecutor> {
    pub fn with_fake_executor(node_id: impl Into<String>) -> Self {
        Self::new(NodeId::new(node_id), Arc::new(FakeLayerExecutor))
    }
}

impl<E> NetworkWorker<E>
where
    E: LayerExecutor + 'static,
{
    pub fn new(node_id: NodeId, executor: Arc<E>) -> Self {
        Self {
            node_id,
            assignment: Arc::new(Mutex::new(None)),
            executor,
            topology_epoch: Arc::new(Mutex::new("worker-epoch-0".into())),
            loaded_shard: Arc::new(Mutex::new(None)),
            stats: RuntimeStats::default(),
            auth_mode: AuthMode::InsecureLocal,
        }
    }

    pub fn with_auth_mode(mut self, auth_mode: AuthMode) -> Self {
        self.auth_mode = auth_mode;
        self
    }

    pub fn runtime_stats(&self) -> RuntimeStats {
        self.stats.clone()
    }

    pub async fn serve(self, listen_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let addr = listen_addr.parse()?;
        println!("bitty-worker: serving activation RPCs on {listen_addr}");
        let service = WorkerServiceServer::new(self)
            .max_decoding_message_size(MAX_ACTIVATION_PAYLOAD_BYTES)
            .max_encoding_message_size(MAX_ACTIVATION_PAYLOAD_BYTES);
        Server::builder().add_service(service).serve(addr).await?;
        Ok(())
    }
}

impl<E> NetworkWorker<E> {
    fn authorize_status<T>(&self, request: &Request<T>) -> Option<Status> {
        let token = request
            .metadata()
            .get(BITTY_TOKEN_HEADER)
            .and_then(|value| value.to_str().ok());
        if self.auth_mode.accepts(token, request.remote_addr()) {
            None
        } else {
            Some(Status::unauthenticated("missing or invalid x-bitty-token"))
        }
    }
}

#[tonic::async_trait]
impl<E> WorkerService for NetworkWorker<E>
where
    E: LayerExecutor + 'static,
{
    async fn forward_activation(
        &self,
        request: Request<ProtoActivationTensor>,
    ) -> Result<Response<ProtoActivationTensor>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let activation = ActivationTensor::try_from(request.into_inner())
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let assignment = self
            .assignment
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("worker has no layer assignment"))?;
        let loaded_shard = self
            .loaded_shard
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("worker shard is not loaded"))?;
        if loaded_shard.range != assignment.range {
            return Err(Status::failed_precondition("loaded shard is stale"));
        }

        let worker = RingWorker::new(self.node_id.clone(), assignment, self.executor.clone());
        let started = Instant::now();
        let payload_len = activation.payload.len() as u64;
        let output = worker
            .forward(activation)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        metrics::record_activation_bytes("rx", payload_len);
        metrics::record_activation_bytes("tx", output.payload.len() as u64);
        self.stats
            .record_forward(
                started.elapsed().as_micros(),
                payload_len,
                output.payload.len() as u64,
            )
            .await;
        Ok(Response::new((&output).into()))
    }

    async fn final_logits(
        &self,
        request: Request<ProtoActivationTensor>,
    ) -> Result<Response<ProtoBitNetLogits>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let activation = ActivationTensor::try_from(request.into_inner())
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        if !activation.verify_checksum() {
            return Err(Status::invalid_argument("activation checksum failed"));
        }
        self.loaded_shard
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("worker shard is not loaded"))?;
        let logits = self
            .executor
            .final_logits(activation)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new((&logits).into()))
    }

    async fn sample_token(
        &self,
        request: Request<SampleTokenRequest>,
    ) -> Result<Response<ProtoTokenOutput>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let request = request.into_inner();
        let activation = request
            .activation
            .ok_or_else(|| Status::invalid_argument("missing activation"))?;
        let activation = ActivationTensor::try_from(activation)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        if !activation.verify_checksum() {
            return Err(Status::invalid_argument("activation checksum failed"));
        }
        self.loaded_shard
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("worker shard is not loaded"))?;
        let logits = self
            .executor
            .final_logits(activation.clone())
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        if !logits.verify_checksum() {
            return Err(Status::internal("worker returned invalid logits checksum"));
        }
        let token_id = logits
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .unwrap_or_default();
        self.stats.record_token().await;
        metrics::record_generated_token(&self.node_id);
        Ok(Response::new(ProtoTokenOutput {
            request_id: activation.request_id,
            token_position: activation.token_position,
            token_id,
            text: format!("<bitnet-rs:{token_id}>"),
            finished: request.finished,
            log_prob: 0.0,
            gen_latency_us: 0,
        }))
    }

    async fn apply_topology(
        &self,
        request: Request<TopologyUpdate>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let update = request.into_inner();
        let assignment = update
            .assignments
            .into_iter()
            .find(|assignment| assignment.node_id == self.node_id.0)
            .map(LayerAssignment::try_from)
            .transpose()
            .map_err(|err| Status::invalid_argument(err.to_string()))?;

        *self.assignment.lock().await = assignment;
        *self.topology_epoch.lock().await = update.topology_epoch.clone();

        Ok(Response::new(HeartbeatResponse {
            accepted: true,
            topology_epoch: update.topology_epoch,
        }))
    }

    async fn load_shard(
        &self,
        request: Request<LoadShardRequest>,
    ) -> Result<Response<LoadShardResponse>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let manifest = request
            .into_inner()
            .manifest
            .ok_or_else(|| Status::invalid_argument("missing shard manifest"))?;
        let manifest = ShardManifestMessage::try_from(manifest)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;

        if manifest.node_id != self.node_id {
            return Err(Status::permission_denied(
                "shard assigned to a different node",
            ));
        }
        if !manifest.path.is_empty() && !std::path::Path::new(&manifest.path).exists() {
            return Err(Status::not_found(format!(
                "shard path does not exist: {}",
                manifest.path
            )));
        }
        if !manifest.sha256_hex.is_empty() && !manifest.path.is_empty() {
            verify_file_sha256(&manifest.path, &manifest.sha256_hex)
                .map_err(ShardHashError::into_status)?;
        }
        if self.loaded_shard.lock().await.as_ref() == Some(&manifest) {
            return Ok(Response::new(LoadShardResponse {
                loaded: true,
                message: "shard manifest already loaded".into(),
            }));
        }

        let assignment = LayerAssignment {
            node_id: manifest.node_id.clone(),
            range: manifest.range.clone(),
            assigned_weight_bytes: manifest.byte_len,
            expected_latency_ms: 0.0,
            next_node_id: None,
            disk_offload_fraction: 0.0,
            model_stage: ModelStage::LayerRange,
        };

        *self.assignment.lock().await = Some(assignment);
        *self.loaded_shard.lock().await = Some(manifest);
        Ok(Response::new(LoadShardResponse {
            loaded: true,
            message: "shard manifest accepted; worker is ready for this topology".into(),
        }))
    }

    async fn cleanup(
        &self,
        request: Request<CleanupRequest>,
    ) -> Result<Response<CleanupResponse>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let request_id = request.into_inner().request_id;
        if request_id.is_empty() {
            return Err(Status::invalid_argument("missing request_id"));
        }
        Ok(Response::new(CleanupResponse { cleaned: true }))
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ShardHashError {
    InvalidExpected,
    Read(String),
    Mismatch,
}

impl ShardHashError {
    fn into_status(self) -> Status {
        match self {
            Self::InvalidExpected => Status::invalid_argument("invalid shard sha256_hex"),
            Self::Read(err) => Status::not_found(err),
            Self::Mismatch => Status::failed_precondition("shard sha256 mismatch"),
        }
    }
}

fn verify_file_sha256(path: &str, expected: &str) -> Result<(), ShardHashError> {
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ShardHashError::InvalidExpected);
    }
    let bytes = std::fs::read(path).map_err(|err| ShardHashError::Read(err.to_string()))?;
    let actual = Sha256::digest(&bytes);
    let actual_hex = actual
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if actual_hex != expected.to_ascii_lowercase() {
        return Err(ShardHashError::Mismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitty_protocol::security::BITTY_TOKEN_HEADER;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn worker_pre_shared_token_rejects_missing_metadata() {
        let worker = NetworkWorker::with_fake_executor("node-a")
            .with_auth_mode(AuthMode::PreSharedToken("secret".into()));
        let request = Request::new(CleanupRequest {
            request_id: "req".into(),
        });

        let status = worker.authorize_status(&request).unwrap();

        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn worker_pre_shared_token_accepts_matching_metadata() {
        let worker = NetworkWorker::with_fake_executor("node-a")
            .with_auth_mode(AuthMode::PreSharedToken("secret".into()));
        let mut request = Request::new(CleanupRequest {
            request_id: "req".into(),
        });
        request
            .metadata_mut()
            .insert(BITTY_TOKEN_HEADER, "secret".parse().unwrap());

        assert!(worker.authorize_status(&request).is_none());
    }

    #[test]
    fn verifies_matching_shard_sha256() {
        let path = unique_temp_path();
        std::fs::write(&path, b"weights").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"weights"));

        let result = verify_file_sha256(path.to_str().unwrap(), &expected);

        let _ = std::fs::remove_file(path);
        assert!(result.is_ok());
    }

    #[test]
    fn rejects_mismatched_shard_sha256() {
        let path = unique_temp_path();
        std::fs::write(&path, b"weights").unwrap();

        let result = verify_file_sha256(
            path.to_str().unwrap(),
            "0000000000000000000000000000000000000000000000000000000000000000",
        );

        let _ = std::fs::remove_file(path);
        assert_eq!(result.unwrap_err(), ShardHashError::Mismatch);
    }

    fn unique_temp_path() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("bitty-worker-test-{nanos}"))
    }
}
