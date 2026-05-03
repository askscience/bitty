use crate::security::{request_rate_limiter, AuthMode, SharedRateLimiter};
use crate::{Halda, Registry, SchedulerConfig};
#[cfg(test)]
use bitty_inference::{LayerExecutor, LowBitReferenceExecutor};
use bitty_observability::record_halda_run;
use bitty_protocol::iroh_transport::{
    self, IrohFrame, BITTY_WORKER_ALPN, DEFAULT_FRAME_LIMIT, WORKER_CLEANUP, WORKER_FINAL_LOGITS,
    WORKER_FORWARD_ACTIVATION, WORKER_LOAD_SHARD,
};
use bitty_protocol::pb::coordinator_service_server::{
    CoordinatorService, CoordinatorServiceServer,
};
use bitty_protocol::pb::worker_service_client::WorkerServiceClient;
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, ClusterStatusRequest, ClusterStatusResponse,
    GenerateRequest as ProtoGenerateRequest, HeartbeatRequest, HeartbeatResponse, LoadShardRequest,
    RegisterWorkerRequest, RegisterWorkerResponse, ShardManifest as ProtoShardManifest,
    TokenOutput as ProtoTokenOutput,
};
use bitty_protocol::{
    ActivationDType, ActivationTensor, HardwareProfile, Heartbeat, LayerMetadata,
    ShardManifestMessage,
};
use futures::stream;
use iroh::{Endpoint, EndpointAddr};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::time::interval;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

#[derive(Clone, Debug)]
pub struct NetworkCoordinator {
    registry: Arc<Mutex<Registry>>,
    layers: Arc<Vec<LayerMetadata>>,
    topology_epoch: Arc<Mutex<String>>,
    auth_mode: AuthMode,
    rate_limiter: SharedRateLimiter,
    model_path: Option<PathBuf>,
    iroh_endpoint: Option<Endpoint>,
    cluster_token: Option<String>,
}

impl NetworkCoordinator {
    pub fn new(layers: Vec<LayerMetadata>) -> Self {
        Self {
            registry: Arc::new(Mutex::new(Registry::default())),
            layers: Arc::new(layers),
            topology_epoch: Arc::new(Mutex::new("network-epoch-0".into())),
            auth_mode: AuthMode::InsecureLocal,
            rate_limiter: request_rate_limiter(NonZeroU32::new(100).expect("non-zero quota")),
            model_path: None,
            iroh_endpoint: None,
            cluster_token: None,
        }
    }

    pub fn with_model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(model_path.into());
        self
    }

    pub fn with_auth_mode(mut self, auth_mode: AuthMode) -> Self {
        self.auth_mode = auth_mode;
        self
    }

    pub fn with_iroh_endpoint(
        mut self,
        endpoint: Endpoint,
        cluster_token: impl Into<String>,
    ) -> Self {
        self.iroh_endpoint = Some(endpoint);
        self.cluster_token = Some(cluster_token.into());
        self
    }

    pub async fn serve(self, listen_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let addr = listen_addr.parse()?;
        println!("bitty-coordinator: listening on {listen_addr}");
        self.clone().spawn_eviction_loop(Duration::from_secs(15));
        Server::builder()
            .add_service(CoordinatorServiceServer::new(self))
            .serve(addr)
            .await?;
        Ok(())
    }

    fn spawn_eviction_loop(self, timeout: Duration) {
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(5));
            loop {
                ticker.tick().await;
                let evicted = self.registry.lock().await.evict_missing(timeout);
                if !evicted.is_empty() {
                    let epoch = self.advance_epoch().await;
                    println!(
                        "evicted {} missing worker(s); topology_epoch={epoch}",
                        evicted.len()
                    );
                }
            }
        });
    }

    async fn current_assignments(&self) -> Result<Vec<bitty_protocol::LayerAssignment>, Status> {
        let profiles = self.registry.lock().await.profiles();
        if profiles.is_empty() {
            return Ok(Vec::new());
        }

        let started = Instant::now();
        let assignments = Halda::new(SchedulerConfig::default())
            .assign(&profiles, &self.layers)
            .map_err(|err| Status::failed_precondition(err.to_string()))?;
        record_halda_run(started.elapsed().as_secs_f64() * 1000.0);
        Ok(assignments)
    }

    async fn advance_epoch(&self) -> String {
        let mut epoch = self.topology_epoch.lock().await;
        let next = epoch
            .rsplit_once('-')
            .and_then(|(_, suffix)| suffix.parse::<u64>().ok())
            .unwrap_or(0)
            + 1;
        *epoch = format!("network-epoch-{next}");
        epoch.clone()
    }

    async fn epoch(&self) -> String {
        self.topology_epoch.lock().await.clone()
    }

    fn authorize_status<T>(&self, request: &Request<T>) -> Option<Status> {
        let key = request
            .remote_addr()
            .map(|addr| addr.ip().to_string())
            .unwrap_or_else(|| "local".into());
        if self.rate_limiter.check_key(&key).is_err() {
            return Some(Status::resource_exhausted(
                "coordinator rate limit exceeded",
            ));
        }

        let token = request
            .metadata()
            .get("x-bitty-token")
            .and_then(|value| value.to_str().ok());
        if self.auth_mode.accepts_token(token) {
            None
        } else {
            Some(Status::unauthenticated("missing or invalid x-bitty-token"))
        }
    }

    #[cfg(test)]
    async fn reference_generate_tokens(
        &self,
        request: ProtoGenerateRequest,
    ) -> Result<Vec<ProtoTokenOutput>, Status> {
        let assignments = self.current_assignments().await?;
        let max_new_tokens = request.max_new_tokens.max(1);
        let request_id = if request.request_id.is_empty() {
            "network-generate".to_string()
        } else {
            request.request_id
        };
        let mut prompt_tokens = request.prompt_tokens;
        if prompt_tokens.is_empty() {
            prompt_tokens = request.prompt.bytes().map(u32::from).collect();
        }
        if prompt_tokens.is_empty() {
            prompt_tokens.push(0);
        }

        let executor = LowBitReferenceExecutor;
        let mut outputs = Vec::with_capacity(max_new_tokens as usize);
        let mut history = prompt_tokens;
        for token_position in 0..max_new_tokens {
            let started = Instant::now();
            let payload = history
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>();
            let mut activation = ActivationTensor::new(
                request_id.clone(),
                token_position,
                0,
                0,
                vec![history.len() as u32],
                ActivationDType::Fp16,
                payload,
            );

            for assignment in &assignments {
                activation = executor
                    .execute_range(&assignment.range, activation)
                    .await
                    .map_err(|err| Status::internal(err.to_string()))?;
            }

            let token_id = activation
                .payload
                .chunks_exact(4)
                .last()
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .unwrap_or(token_position)
                % 32_000;
            history.push(token_id);
            outputs.push(ProtoTokenOutput {
                request_id: request_id.clone(),
                token_position,
                token_id,
                text: format!("<bitnet-rs:{token_id}>"),
                finished: token_position + 1 == max_new_tokens,
                log_prob: 0.0,
                gen_latency_us: started.elapsed().as_micros() as u64,
            });
        }

        Ok(outputs)
    }

    async fn remote_generate_tokens(
        &self,
        request: ProtoGenerateRequest,
    ) -> Result<Vec<ProtoTokenOutput>, Status> {
        let assignments = self.current_assignments().await?;
        if assignments.is_empty() {
            return Err(Status::failed_precondition("no workers registered"));
        }
        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("coordinator has no --model path"))?;
        let max_new_tokens = request.max_new_tokens.max(1);
        let request_id = if request.request_id.is_empty() {
            "network-generate".to_string()
        } else {
            request.request_id
        };
        let mut current_input = if request.prompt_tokens.is_empty() {
            request.prompt.bytes().map(u32::from).collect::<Vec<_>>()
        } else {
            request.prompt_tokens
        };
        if current_input.is_empty() {
            current_input.push(0);
        }

        let mut worker_clients = Vec::with_capacity(assignments.len());
        for assignment in &assignments {
            let profile = self
                .registry
                .lock()
                .await
                .profile(&assignment.node_id)
                .ok_or_else(|| {
                    Status::failed_precondition("assignment references unknown worker")
                })?;
            if profile.worker_endpoint.is_empty() {
                return Err(Status::failed_precondition(format!(
                    "worker {} did not register a reachable endpoint",
                    profile.node_id.0
                )));
            }
            let mut client = self.worker_client(&profile.worker_endpoint).await?;
            let manifest = ShardManifestMessage {
                shard_id: format!(
                    "{}:{}..{}",
                    assignment.node_id.0,
                    assignment.range.start_layer,
                    assignment.range.end_layer_exclusive
                ),
                node_id: assignment.node_id.clone(),
                range: assignment.range.clone(),
                byte_len: assignment.assigned_weight_bytes,
                sha256_hex: String::new(),
                path: model_path.to_string_lossy().into_owned(),
            };
            client
                .load_shard(LoadShardRequest {
                    manifest: Some(ProtoShardManifest::from(&manifest)),
                })
                .await
                .map_err(|err| Status::failed_precondition(err.to_string()))?;
            worker_clients.push(client);
        }

        let mut outputs = Vec::with_capacity(max_new_tokens as usize);
        for token_position in 0..max_new_tokens {
            let started = Instant::now();
            let payload = current_input
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>();
            let mut activation = ActivationTensor::new(
                request_id.clone(),
                token_position,
                0,
                0,
                vec![current_input.len() as u32],
                ActivationDType::Fp16,
                payload,
            );

            for client in &mut worker_clients {
                activation = client.forward_activation(&activation).await?;
            }

            let last = worker_clients
                .last_mut()
                .ok_or_else(|| Status::failed_precondition("no last worker"))?;
            let logits = bitty_protocol::BitNetLogits::from(last.final_logits(&activation).await?);
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
            let text = format!("<bitnet-rs:{token_id}>");
            current_input.clear();
            current_input.push(token_id);
            outputs.push(ProtoTokenOutput {
                request_id: request_id.clone(),
                token_position,
                token_id,
                text,
                finished: token_position + 1 == max_new_tokens,
                log_prob: 0.0,
                gen_latency_us: started.elapsed().as_micros() as u64,
            });
        }

        for client in &mut worker_clients {
            let _ = client
                .cleanup(bitty_protocol::pb::CleanupRequest {
                    request_id: request_id.clone(),
                })
                .await;
        }

        Ok(outputs)
    }

    async fn worker_client(&self, endpoint: &str) -> Result<WorkerRpcClient, Status> {
        if let Some(target) = bitty_protocol::iroh_transport::parse_iroh_target(endpoint) {
            let iroh_endpoint = self
                .iroh_endpoint
                .clone()
                .ok_or_else(|| Status::failed_precondition("coordinator has no iroh endpoint"))?;
            return Ok(WorkerRpcClient::Iroh(IrohWorkerClient {
                endpoint: iroh_endpoint,
                endpoint_addr: target.endpoint_addr,
                token: target
                    .token
                    .or_else(|| self.cluster_token.clone())
                    .unwrap_or_default(),
            }));
        }
        let client = WorkerServiceClient::connect(normalize_endpoint(endpoint))
            .await
            .map_err(|err| Status::unavailable(err.to_string()))?;
        Ok(WorkerRpcClient::Tcp(client))
    }
}

enum WorkerRpcClient {
    Tcp(WorkerServiceClient<tonic::transport::Channel>),
    Iroh(IrohWorkerClient),
}

impl WorkerRpcClient {
    async fn load_shard(&mut self, request: LoadShardRequest) -> Result<(), Status> {
        match self {
            Self::Tcp(client) => {
                client
                    .load_shard(request)
                    .await
                    .map_err(|err| Status::failed_precondition(err.to_string()))?;
            }
            Self::Iroh(client) => {
                let _: bitty_protocol::pb::LoadShardResponse = client
                    .request(WORKER_LOAD_SHARD, &request)
                    .await
                    .map_err(|err| Status::failed_precondition(err.to_string()))?;
            }
        }
        Ok(())
    }

    async fn forward_activation(
        &mut self,
        activation: &ActivationTensor,
    ) -> Result<ActivationTensor, Status> {
        match self {
            Self::Tcp(client) => ActivationTensor::try_from(
                client
                    .forward_activation(ProtoActivationTensor::from(activation))
                    .await
                    .map_err(|err| Status::internal(err.to_string()))?
                    .into_inner(),
            )
            .map_err(|err| Status::internal(err.to_string())),
            Self::Iroh(client) => {
                let response = client
                    .request::<_, ProtoActivationTensor>(
                        WORKER_FORWARD_ACTIVATION,
                        &ProtoActivationTensor::from(activation),
                    )
                    .await
                    .map_err(|err| Status::internal(err.to_string()))?;
                ActivationTensor::try_from(response)
                    .map_err(|err| Status::internal(err.to_string()))
            }
        }
    }

    async fn final_logits(
        &mut self,
        activation: &ActivationTensor,
    ) -> Result<bitty_protocol::pb::BitNetLogits, Status> {
        match self {
            Self::Tcp(client) => Ok(client
                .final_logits(ProtoActivationTensor::from(activation))
                .await
                .map_err(|err| Status::internal(err.to_string()))?
                .into_inner()),
            Self::Iroh(client) => client
                .request(
                    WORKER_FINAL_LOGITS,
                    &ProtoActivationTensor::from(activation),
                )
                .await
                .map_err(|err| Status::internal(err.to_string())),
        }
    }

    async fn cleanup(&mut self, request: bitty_protocol::pb::CleanupRequest) -> Result<(), Status> {
        match self {
            Self::Tcp(client) => {
                let _ = client.cleanup(request).await;
            }
            Self::Iroh(client) => {
                let _ = client
                    .request::<_, bitty_protocol::pb::CleanupResponse>(WORKER_CLEANUP, &request)
                    .await;
            }
        }
        Ok(())
    }
}

struct IrohWorkerClient {
    endpoint: Endpoint,
    endpoint_addr: EndpointAddr,
    token: String,
}

impl IrohWorkerClient {
    async fn request<M, R>(
        &self,
        op: u8,
        message: &M,
    ) -> Result<R, iroh_transport::IrohTransportError>
    where
        M: prost::Message,
        R: prost::Message + Default,
    {
        let frame = IrohFrame::message(op, self.token.clone(), message);
        let response = iroh_transport::request_addr(
            &self.endpoint,
            self.endpoint_addr.clone(),
            BITTY_WORKER_ALPN,
            frame,
            DEFAULT_FRAME_LIMIT,
        )
        .await?;
        response.decode_message(op)
    }
}

fn normalize_endpoint(endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.into()
    } else {
        format!("http://{endpoint}")
    }
}

#[tonic::async_trait]
impl CoordinatorService for NetworkCoordinator {
    type GenerateStream =
        Pin<Box<dyn futures::Stream<Item = Result<ProtoTokenOutput, Status>> + Send + 'static>>;
    type StreamTokensStream =
        Pin<Box<dyn futures::Stream<Item = Result<ProtoTokenOutput, Status>> + Send + 'static>>;

    async fn register_worker(
        &self,
        request: Request<RegisterWorkerRequest>,
    ) -> Result<Response<RegisterWorkerResponse>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let profile = request
            .into_inner()
            .profile
            .ok_or_else(|| Status::invalid_argument("missing worker profile"))?;
        let profile = HardwareProfile::try_from(profile)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let node_id = profile.node_id.clone();

        self.registry.lock().await.register(profile);
        let topology_epoch = self.advance_epoch().await;
        let assignments = self.current_assignments().await?;

        println!(
            "registered worker {node_id}; active_nodes={} assignments={}",
            self.registry.lock().await.profiles().len(),
            assignments.len()
        );

        Ok(Response::new(RegisterWorkerResponse {
            assignments: assignments.iter().map(Into::into).collect(),
            topology_epoch,
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let heartbeat = Heartbeat::from(request.into_inner());
        let accepted = self.registry.lock().await.heartbeat(heartbeat);

        Ok(Response::new(HeartbeatResponse {
            accepted,
            topology_epoch: self.epoch().await,
        }))
    }

    async fn generate(
        &self,
        request: Request<ProtoGenerateRequest>,
    ) -> Result<Response<Self::StreamTokensStream>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let request = request.into_inner();
        let tokens = self
            .remote_generate_tokens(request)
            .await?
            .into_iter()
            .map(Ok);

        Ok(Response::new(Box::pin(stream::iter(tokens))))
    }

    async fn stream_tokens(
        &self,
        request: Request<ProtoActivationTensor>,
    ) -> Result<Response<Self::StreamTokensStream>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let activation = request.into_inner();
        let token = ProtoTokenOutput {
            request_id: activation.request_id,
            token_position: activation.token_position,
            token_id: 0,
            text: "<network-token-placeholder>".into(),
            finished: true,
            log_prob: 0.0,
            gen_latency_us: 0,
        };

        Ok(Response::new(Box::pin(stream::iter([Ok(token)]))))
    }

    async fn cluster_status(
        &self,
        request: Request<ClusterStatusRequest>,
    ) -> Result<Response<ClusterStatusResponse>, Status> {
        if let Some(status) = self.authorize_status(&request) {
            return Err(status);
        }
        let assignments = self.current_assignments().await?;
        let active_workers = self.registry.lock().await.len() as u32;
        let model_path = self
            .model_path
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(Response::new(ClusterStatusResponse {
            leader_id: "local-leader".into(),
            topology_epoch: self.epoch().await,
            active_workers,
            assignments: assignments.iter().map(Into::into).collect(),
            model_ready: !model_path.is_empty(),
            model_path,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reference_generate_is_deterministic() {
        let coordinator = NetworkCoordinator::new(vec![LayerMetadata {
            layer_id: 0,
            weight_bytes: 128,
            activation_bytes: 8,
            estimated_flops: 1.0,
            precision_critical: true,
        }]);
        let request = ProtoGenerateRequest {
            request_id: "req".into(),
            prompt_tokens: vec![1, 2, 3],
            prompt: String::new(),
            max_new_tokens: 2,
            temperature: 0.0,
        };

        let first = coordinator
            .reference_generate_tokens(request.clone())
            .await
            .unwrap();
        let second = coordinator
            .reference_generate_tokens(request)
            .await
            .unwrap();

        assert_eq!(
            first
                .iter()
                .map(|token| (token.token_id, token.text.as_str(), token.finished))
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|token| (token.token_id, token.text.as_str(), token.finished))
                .collect::<Vec<_>>()
        );
        assert_eq!(first.len(), 2);
        assert!(first.last().unwrap().finished);
    }
}
