use crate::{metrics, RingWorker};
use bitty_inference::{FakeLayerExecutor, LayerExecutor};
use bitty_protocol::pb::worker_service_server::{WorkerService, WorkerServiceServer};
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, ChainAck, ChainResult, ChainResultRequest,
    ChainStep, CleanupRequest, CleanupResponse, HeartbeatResponse, LoadShardRequest,
    LoadShardResponse, Logits as ProtoLogits, SampleTokenRequest, TokenOutput as ProtoTokenOutput,
    TopologyUpdate,
};
use bitty_protocol::security::{AuthMode, BITTY_TOKEN_HEADER};
use bitty_protocol::validation::MAX_ACTIVATION_PAYLOAD_BYTES;
use bitty_protocol::worker_client::{connect_worker, status_text, WorkerRpcClient};
use bitty_protocol::{ActivationTensor, LayerAssignment, ModelStage, NodeId, ShardManifestMessage};
use iroh::Endpoint;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{oneshot, Mutex};
use tonic::transport::Server;
use tonic::{Request, Response, Status};

pub struct NetworkWorker<E = FakeLayerExecutor> {
    node_id: NodeId,
    assignment: Arc<Mutex<Option<LayerAssignment>>>,
    executor: Arc<E>,
    topology_epoch: Arc<Mutex<String>>,
    loaded_shard: Arc<Mutex<Option<ShardManifestMessage>>>,
    stats: RuntimeStats,
    auth_mode: AuthMode,
    /// Needed to reach the next worker of a route when the cluster runs over iroh.
    iroh_endpoint: Option<Endpoint>,
    /// Connections to the workers this one hands its results to, by endpoint.
    peers: Arc<Mutex<HashMap<String, Arc<Mutex<WorkerRpcClient>>>>>,
    chain_results: ChainMailbox,
}

// written out because the derive would ask the executor itself to be `Clone`
impl<E> Clone for NetworkWorker<E> {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id.clone(),
            assignment: self.assignment.clone(),
            executor: self.executor.clone(),
            topology_epoch: self.topology_epoch.clone(),
            loaded_shard: self.loaded_shard.clone(),
            stats: self.stats.clone(),
            auth_mode: self.auth_mode.clone(),
            iroh_endpoint: self.iroh_endpoint.clone(),
            peers: self.peers.clone(),
            chain_results: self.chain_results.clone(),
        }
    }
}

/// Where the last worker of a route keeps a finished token until the coordinator takes it.
/// The coordinator asks while the token is still on its way, so most of the time the
/// request is here first and waits.
#[derive(Clone, Default)]
struct ChainMailbox {
    slots: Arc<std::sync::Mutex<HashMap<(String, u32), ChainSlot>>>,
}

enum ChainSlot {
    Ready(ChainResult),
    Waiting(oneshot::Sender<ChainResult>),
}

impl ChainMailbox {
    fn deliver(&self, key: (String, u32), result: ChainResult) {
        let mut slots = self.slots.lock().expect("chain mailbox");
        match slots.remove(&key) {
            // a coordinator that gave up leaves a closed channel: the result is dropped
            Some(ChainSlot::Waiting(sender)) => {
                let _ = sender.send(result);
            }
            _ => {
                slots.insert(key, ChainSlot::Ready(result));
            }
        }
    }

    fn take_or_wait(
        &self,
        key: (String, u32),
    ) -> Result<ChainResult, oneshot::Receiver<ChainResult>> {
        let mut slots = self.slots.lock().expect("chain mailbox");
        if let Some(ChainSlot::Ready(result)) = slots.remove(&key) {
            return Ok(result);
        }
        let (sender, receiver) = oneshot::channel();
        slots.insert(key, ChainSlot::Waiting(sender));
        Err(receiver)
    }

    fn forget(&self, request_id: &str) {
        self.slots
            .lock()
            .expect("chain mailbox")
            .retain(|key, _| key.0 != request_id);
    }
}

/// An endpoint as it may appear in a log or an error: the cluster token is left out.
fn endpoint_name(endpoint: &str) -> &str {
    endpoint.split('?').next().unwrap_or(endpoint)
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
            iroh_endpoint: None,
            peers: Arc::new(Mutex::new(HashMap::new())),
            chain_results: ChainMailbox::default(),
        }
    }

    pub fn with_iroh_endpoint(mut self, endpoint: Endpoint) -> Self {
        self.iroh_endpoint = Some(endpoint);
        self
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

impl<E> NetworkWorker<E>
where
    E: LayerExecutor + 'static,
{
    /// Runs the layers of this worker on one activation.
    async fn run_stage(&self, activation: ActivationTensor) -> Result<ActivationTensor, Status> {
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
        let output = trim_logits(output, bitty_inference::sampling::top_logits_limit());
        metrics::record_activation_bytes("rx", payload_len);
        metrics::record_activation_bytes("tx", output.payload.len() as u64);
        self.stats
            .record_forward(
                started.elapsed().as_micros(),
                payload_len,
                output.payload.len() as u64,
            )
            .await;
        Ok(output)
    }

    /// The part of a chain step that runs after the sender got its answer: the layers, then
    /// the hand-over to the next worker, or to the coordinator when this is the last one.
    async fn run_chain_step(
        &self,
        key: (String, u32),
        activation: ActivationTensor,
        route: Vec<String>,
        mut hop_micros: Vec<u64>,
    ) {
        let started = Instant::now();
        let result = self.run_stage(activation).await;
        hop_micros.push(started.elapsed().as_micros() as u64);
        let output = match result {
            Ok(output) => output,
            Err(status) => {
                let error = format!("worker {}: {}", self.node_id.0, status_text(&status));
                return self
                    .report_chain_failure(key, route.last(), error, hop_micros)
                    .await;
            }
        };
        let Some((next, rest)) = route.split_first() else {
            self.chain_results.deliver(
                key,
                ChainResult {
                    activation: Some((&output).into()),
                    error: String::new(),
                    hop_micros,
                },
            );
            return;
        };
        let step = ChainStep {
            activation: Some((&output).into()),
            route: rest.to_vec(),
            error: String::new(),
            hop_micros: hop_micros.clone(),
        };
        if let Err(status) = self.send_chain_step(next, step).await {
            let error = format!(
                "worker {} could not hand its result to {}: {}",
                self.node_id.0,
                endpoint_name(next),
                status_text(&status)
            );
            // the last worker is the only one the coordinator listens to; when it is the
            // one that cannot be reached, the coordinator finds out by its own timeout
            self.report_chain_failure(key, rest.last(), error, hop_micros)
                .await;
        }
    }

    /// Tells the coordinator that a token was lost on the way, through the last worker of
    /// the route (`None` when this worker is the last).
    async fn report_chain_failure(
        &self,
        key: (String, u32),
        last: Option<&String>,
        error: String,
        hop_micros: Vec<u64>,
    ) {
        eprintln!("bitty-worker: {error}");
        let Some(last) = last else {
            self.chain_results.deliver(
                key,
                ChainResult {
                    activation: None,
                    error,
                    hop_micros,
                },
            );
            return;
        };
        let step = ChainStep {
            activation: Some(ProtoActivationTensor {
                request_id: key.0,
                token_position: key.1,
                ..Default::default()
            }),
            route: Vec::new(),
            error,
            hop_micros,
        };
        if let Err(status) = self.send_chain_step(last, step).await {
            eprintln!(
                "bitty-worker: could not report the failure to {}: {}",
                endpoint_name(last),
                status_text(&status)
            );
        }
    }

    async fn send_chain_step(&self, endpoint: &str, step: ChainStep) -> Result<(), Status> {
        let known = self.peers.lock().await.get(endpoint).cloned();
        let client = match known {
            Some(client) => client,
            None => {
                let token = match &self.auth_mode {
                    AuthMode::PreSharedToken(token) => Some(token.clone()),
                    _ => None,
                };
                let client = connect_worker(endpoint, self.iroh_endpoint.clone(), token).await?;
                let client = Arc::new(Mutex::new(client));
                self.peers
                    .lock()
                    .await
                    .insert(endpoint.to_string(), client.clone());
                client
            }
        };
        let result = client.lock().await.forward_chain(step).await;
        if result.is_err() {
            // a new connection is made for the next token
            self.peers.lock().await.remove(endpoint);
        }
        result
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

/// The final stage answers with one logit per vocabulary entry, about 1 MB per token for this
/// model. Only the best ones matter to the sampler, so only those travel.
fn trim_logits(output: ActivationTensor, limit: usize) -> ActivationTensor {
    let is_logits = output.dtype == bitty_protocol::ActivationDType::F32 && output.shape.len() == 1;
    if !is_logits || limit == 0 || output.payload.len() / 4 <= limit {
        return output;
    }
    let logits: Vec<f32> = output
        .payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let best = bitty_inference::sampling::top_logits(&logits, limit);
    ActivationTensor::new(
        output.request_id,
        output.token_position,
        output.source_layer,
        output.target_layer,
        vec![best.len() as u32],
        bitty_protocol::ActivationDType::TopLogits,
        bitty_inference::sampling::encode_top_logits(&best),
    )
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
        let output = self.run_stage(activation).await?;
        Ok(Response::new((&output).into()))
    }

    async fn final_logits(
        &self,
        request: Request<ProtoActivationTensor>,
    ) -> Result<Response<ProtoLogits>, Status> {
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
        let text = self.executor.decode_token_text(token_id).await;
        self.stats.record_token().await;
        metrics::record_generated_token(&self.node_id);
        Ok(Response::new(ProtoTokenOutput {
            request_id: activation.request_id,
            token_position: activation.token_position,
            token_id,
            text,
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

        // Fetches the stage file if this node does not hold it yet, then loads the layers.
        self.executor
            .prepare_range(&manifest.range)
            .await
            .map_err(|err| Status::failed_precondition(err.to_string()))?;

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
        self.executor.end_request(&request_id).await;
        self.chain_results.forget(&request_id);
        Ok(Response::new(CleanupResponse { cleaned: true }))
    }

    async fn forward_chain(
        &self,
        request: Request<ChainStep>,
    ) -> Result<Response<ChainAck>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let step = request.into_inner();
        let activation = step
            .activation
            .ok_or_else(|| Status::invalid_argument("missing activation"))?;
        let key = (activation.request_id.clone(), activation.token_position);
        if !step.error.is_empty() {
            // a worker before this one failed and sent the news here, to the end of the route
            self.chain_results.deliver(
                key,
                ChainResult {
                    activation: None,
                    error: step.error,
                    hop_micros: step.hop_micros,
                },
            );
            return Ok(Response::new(ChainAck { accepted: true }));
        }
        let activation = ActivationTensor::try_from(activation)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let worker = self.clone();
        tokio::spawn(async move {
            worker
                .run_chain_step(key, activation, step.route, step.hop_micros)
                .await;
        });
        Ok(Response::new(ChainAck { accepted: true }))
    }

    async fn await_chain_result(
        &self,
        request: Request<ChainResultRequest>,
    ) -> Result<Response<ChainResult>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let request = request.into_inner();
        let key = (request.request_id, request.token_position);
        let result = match self.chain_results.take_or_wait(key) {
            Ok(result) => result,
            Err(receiver) => receiver
                .await
                .map_err(|_| Status::cancelled("the request was ended"))?,
        };
        Ok(Response::new(result))
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

    fn chain_result(error: &str) -> ChainResult {
        ChainResult {
            activation: None,
            error: error.into(),
            hop_micros: Vec::new(),
        }
    }

    #[tokio::test]
    async fn chain_mailbox_serves_a_request_that_came_first() {
        let mailbox = ChainMailbox::default();
        let receiver = mailbox.take_or_wait(("req".into(), 3)).unwrap_err();
        mailbox.deliver(("req".into(), 3), chain_result("late"));

        assert_eq!(receiver.await.unwrap().error, "late");
    }

    #[test]
    fn chain_mailbox_keeps_a_result_until_asked_and_forgets_ended_requests() {
        let mailbox = ChainMailbox::default();
        mailbox.deliver(("req".into(), 0), chain_result("kept"));
        mailbox.deliver(("old".into(), 0), chain_result("stale"));
        mailbox.forget("old");

        assert_eq!(
            mailbox.take_or_wait(("req".into(), 0)).unwrap().error,
            "kept"
        );
        assert!(mailbox.take_or_wait(("old".into(), 0)).is_err());
    }

    #[test]
    fn endpoint_names_leave_the_token_out() {
        assert_eq!(endpoint_name("iroh://abc?token=secret"), "iroh://abc");
        assert_eq!(endpoint_name("127.0.0.1:50061"), "127.0.0.1:50061");
    }

    fn unique_temp_path() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("bitty-worker-test-{nanos}"))
    }
}
