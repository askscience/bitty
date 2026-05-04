use crate::security::{request_rate_limiter, AuthMode, SharedRateLimiter};
use crate::worker_client::{connect_worker, WorkerRpcClient};
use crate::{Halda, Registry, SchedulerConfig};
#[cfg(test)]
use bitty_inference::{LayerExecutor, LowBitReferenceExecutor};
use bitty_observability::record_halda_run;
use bitty_protocol::endpoint::validate_worker_endpoint_for_dial;
use bitty_protocol::pb::coordinator_service_server::{
    CoordinatorService, CoordinatorServiceServer,
};
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, ClusterStatusRequest, ClusterStatusResponse,
    GenerateRequest as ProtoGenerateRequest, HeartbeatRequest, HeartbeatResponse, LoadShardRequest,
    RegisterWorkerRequest, RegisterWorkerResponse, SampleTokenRequest,
    ShardManifest as ProtoShardManifest, TokenOutput as ProtoTokenOutput,
};
use bitty_protocol::security::BITTY_TOKEN_HEADER;
use bitty_protocol::validation::MAX_ACTIVATION_PAYLOAD_BYTES;
use bitty_protocol::{
    ActivationDType, ActivationTensor, HardwareProfile, Heartbeat, LayerMetadata,
    ShardManifestMessage,
};
use futures::stream;
use iroh::Endpoint;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::mpsc;
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
    shard_cache: Arc<Mutex<HashMap<String, ShardManifestMessage>>>,
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
            shard_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(model_path.into());
        self
    }

    pub fn with_auth_mode(mut self, auth_mode: AuthMode) -> Self {
        if let AuthMode::PreSharedToken(token) = &auth_mode {
            self.cluster_token = Some(token.clone());
        }
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
        let service = CoordinatorServiceServer::new(self)
            .max_decoding_message_size(MAX_ACTIVATION_PAYLOAD_BYTES)
            .max_encoding_message_size(MAX_ACTIVATION_PAYLOAD_BYTES);
        Server::builder().add_service(service).serve(addr).await?;
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
            .get(BITTY_TOKEN_HEADER)
            .and_then(|value| value.to_str().ok());
        if self.auth_mode.accepts(token, request.remote_addr()) {
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
        sender: Option<mpsc::Sender<Result<ProtoTokenOutput, Status>>>,
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
            let worker_model_path = if profile.model_path.is_empty() {
                model_path.to_string_lossy().into_owned()
            } else {
                profile.model_path.clone()
            };
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
                path: worker_model_path,
            };
            let cache_key = format!("{}@{}", manifest.node_id.0, profile.worker_endpoint);
            let needs_load = self
                .shard_cache
                .lock()
                .await
                .get(&cache_key)
                .map(|cached| cached != &manifest)
                .unwrap_or(true);
            if needs_load {
                client
                    .load_shard(LoadShardRequest {
                        manifest: Some(ProtoShardManifest::from(&manifest)),
                    })
                    .await
                    .map_err(|err| Status::failed_precondition(err.to_string()))?;
                self.shard_cache.lock().await.insert(cache_key, manifest);
            }
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
            let mut token = last
                .sample_token(SampleTokenRequest {
                    activation: Some((&activation).into()),
                    temperature: request.temperature,
                    finished: token_position + 1 == max_new_tokens,
                })
                .await?;
            token.gen_latency_us = started.elapsed().as_micros() as u64;
            current_input.clear();
            current_input.push(token.token_id);
            if let Some(sender) = &sender {
                if sender.send(Ok(token.clone())).await.is_err() {
                    break;
                }
            }
            outputs.push(token);
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
        connect_worker(
            endpoint,
            self.iroh_endpoint.clone(),
            self.cluster_token.clone(),
        )
        .await
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
        let register = request.into_inner();
        bitty_protocol::validate_register_worker(&register)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let profile = register
            .profile
            .ok_or_else(|| Status::invalid_argument("missing worker profile"))?;
        let profile = HardwareProfile::try_from(profile)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        if !profile.worker_endpoint.is_empty() {
            validate_worker_endpoint_for_dial(&profile.worker_endpoint)
                .map_err(|err| Status::invalid_argument(err.to_string()))?;
        }
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
        bitty_protocol::GenerateRequest::try_from(request.clone())
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let coordinator = self.clone();
        let (sender, receiver) = mpsc::channel(8);
        tokio::spawn(async move {
            match coordinator
                .remote_generate_tokens(request, Some(sender.clone()))
                .await
            {
                Ok(tokens) => {
                    let _ = tokens;
                }
                Err(err) => {
                    let _ = sender.send(Err(err)).await;
                }
            }
        });
        let tokens = stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|item| (item, receiver))
        });

        Ok(Response::new(Box::pin(tokens)))
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
        let profiles = self.registry.lock().await.profiles();
        let active_workers = profiles.len() as u32;
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
            profiles: profiles.iter().map(Into::into).collect(),
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

    #[test]
    fn pre_shared_token_rejects_missing_metadata() {
        let coordinator = NetworkCoordinator::new(Vec::new())
            .with_auth_mode(AuthMode::PreSharedToken("secret".into()));
        let request = Request::new(ClusterStatusRequest {});

        let status = coordinator.authorize_status(&request).unwrap();

        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn pre_shared_token_accepts_matching_metadata() {
        let coordinator = NetworkCoordinator::new(Vec::new())
            .with_auth_mode(AuthMode::PreSharedToken("secret".into()));
        let mut request = Request::new(ClusterStatusRequest {});
        request
            .metadata_mut()
            .insert(BITTY_TOKEN_HEADER, "secret".parse().unwrap());

        assert!(coordinator.authorize_status(&request).is_none());
    }
}
