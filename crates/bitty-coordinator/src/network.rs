use crate::security::{request_rate_limiter, AuthMode, SharedRateLimiter};
use crate::worker_client::{connect_worker, WorkerRpcClient};
use crate::{Halda, Registry, SchedulerConfig};
#[cfg(test)]
use bitty_inference::{LayerExecutor, LowBitReferenceExecutor};
use bitty_inference::{
    Sampler, SamplingOptions, TextCodec, TextStream, DEFAULT_TOP_K, DEFAULT_TOP_P,
};
use bitty_observability::record_halda_run;
use bitty_protocol::endpoint::validate_worker_endpoint_for_dial;
use bitty_protocol::pb::coordinator_service_server::{
    CoordinatorService, CoordinatorServiceServer,
};
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, ChainStep, ClusterStatusRequest, ClusterStatusResponse,
    GenerateRequest as ProtoGenerateRequest, HeartbeatRequest, HeartbeatResponse, LoadShardRequest,
    RegisterWorkerRequest, RegisterWorkerResponse,
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
    text_codec: Arc<Mutex<Option<Arc<TextCodec>>>>,
    /// Held while shards are being loaded, so a request waits for a running preparation
    /// instead of asking the same workers twice.
    prepare_lock: Arc<Mutex<()>>,
    /// The topology in which the workers could not pass a token to each other. While it
    /// lasts, every worker answers the coordinator instead.
    chain_broken: Arc<Mutex<Option<String>>>,
}

/// `BITTY_CHAIN=0` makes every worker answer the coordinator, as it was before the workers
/// learned to pass the result to each other.
fn chain_wanted() -> bool {
    std::env::var("BITTY_CHAIN")
        .map(|value| value.trim() != "0")
        .unwrap_or(true)
}

/// How long the coordinator waits for a token that is going from worker to worker.
/// `BITTY_CHAIN_TIMEOUT_SECS` changes it. It has to cover a long prompt on slow machines.
fn chain_timeout() -> Duration {
    let seconds = std::env::var("BITTY_CHAIN_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(600);
    Duration::from_secs(seconds)
}

fn is_loopback_endpoint(endpoint: &str) -> bool {
    let rest = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    let authority = rest.split('/').next().unwrap_or(rest);
    let host = match authority.strip_prefix('[') {
        Some(inner) => inner.split(']').next().unwrap_or(inner),
        None => authority.rsplit_once(':').map_or(authority, |(host, _)| host),
    };
    host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

/// How long the topology has to stay unchanged before the workers are asked to load their
/// layers. `BITTY_PREPARE_SETTLE_SECS` changes it; 0 turns the early preparation off, and
/// the layers are then loaded by the first request.
fn prepare_settle_time() -> Option<Duration> {
    let seconds = std::env::var("BITTY_PREPARE_SETTLE_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(10);
    (seconds > 0).then(|| Duration::from_secs(seconds))
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
            text_codec: Arc::new(Mutex::new(None)),
            prepare_lock: Arc::new(Mutex::new(())),
            chain_broken: Arc::new(Mutex::new(None)),
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
        self.spawn_background_loops();
        let service = CoordinatorServiceServer::new(self)
            .max_decoding_message_size(MAX_ACTIVATION_PAYLOAD_BYTES)
            .max_encoding_message_size(MAX_ACTIVATION_PAYLOAD_BYTES);
        Server::builder().add_service(service).serve(addr).await?;
        Ok(())
    }

    /// Starts the loops that run beside the request handlers: eviction of silent workers and
    /// early preparation of the shards. `serve` calls it; a coordinator reached only through
    /// iroh has to call it itself.
    pub fn spawn_background_loops(&self) {
        self.clone().spawn_eviction_loop(Duration::from_secs(15));
        if let Some(settle) = prepare_settle_time() {
            self.clone().spawn_prepare_loop(settle);
        }
    }

    /// Asks the workers to load their layers once the topology has stopped changing, so the
    /// download and the load of a stage happen before the first request and not during it.
    /// The wait matters: every node that joins moves the ranges of the others, and a range
    /// that is about to change is not worth a download of several gigabytes.
    fn spawn_prepare_loop(self, settle: Duration) {
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(1));
            let mut seen = self.epoch().await;
            let mut seen_at = Instant::now();
            let mut prepared = seen.clone();
            loop {
                ticker.tick().await;
                let epoch = self.epoch().await;
                if epoch != seen {
                    seen = epoch;
                    seen_at = Instant::now();
                    continue;
                }
                if seen == prepared || seen_at.elapsed() < settle {
                    continue;
                }
                prepared = seen.clone();
                let Some(model_path) = self.model_path.clone() else {
                    continue;
                };
                let assignments = match self.current_assignments().await {
                    Ok(assignments) if !assignments.is_empty() => assignments,
                    _ => continue,
                };
                println!(
                    "preparing {} worker(s) for topology_epoch={seen}",
                    assignments.len()
                );
                let started = Instant::now();
                // A node that joins while layers are still being fetched changes every range:
                // waiting for the old ones would only delay the new plan.
                let changed = async {
                    loop {
                        ticker.tick().await;
                        if self.epoch().await != seen {
                            break;
                        }
                    }
                };
                tokio::select! {
                    result = self.prepare_workers(&assignments, &model_path) => match result {
                        Ok(_) => println!(
                            "workers ready in {:.1}s; topology_epoch={seen}",
                            started.elapsed().as_secs_f64()
                        ),
                        Err(err) => println!("could not prepare the workers: {}", err.message()),
                    },
                    _ = changed => println!(
                        "the network changed while preparing topology_epoch={seen}; planning again"
                    ),
                }
            }
        });
    }

    /// Connects to the worker of every assignment and makes it load its layers, unless it
    /// already holds exactly that range. Workers load in parallel.
    async fn prepare_workers(
        &self,
        assignments: &[bitty_protocol::LayerAssignment],
        model_path: &std::path::Path,
    ) -> Result<Vec<WorkerRpcClient>, Status> {
        let _guard = self.prepare_lock.lock().await;
        let jobs = assignments
            .iter()
            .map(|assignment| self.prepare_worker(assignment, model_path));
        futures::future::join_all(jobs).await.into_iter().collect()
    }

    async fn prepare_worker(
        &self,
        assignment: &bitty_protocol::LayerAssignment,
        model_path: &std::path::Path,
    ) -> Result<WorkerRpcClient, Status> {
        let profile = self
            .registry
            .lock()
            .await
            .profile(&assignment.node_id)
            .ok_or_else(|| Status::failed_precondition("assignment references unknown worker"))?;
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
        Ok(client)
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
                text: format!("<token:{token_id}>"),
                finished: token_position + 1 == max_new_tokens,
                log_prob: 0.0,
                gen_latency_us: started.elapsed().as_micros() as u64,
            });
        }

        Ok(outputs)
    }

    /// The endpoints the workers use to reach each other, in the order of the layers, or
    /// `None` when the tokens have to go through the coordinator.
    async fn chain_route(
        &self,
        assignments: &[bitty_protocol::LayerAssignment],
    ) -> Option<Vec<String>> {
        if !chain_wanted() || assignments.len() < 2 {
            return None;
        }
        if self.chain_broken.lock().await.as_deref() == Some(self.epoch().await.as_str()) {
            return None;
        }
        let registry = self.registry.lock().await;
        let mut route = Vec::with_capacity(assignments.len());
        for assignment in assignments {
            let endpoint = registry.profile(&assignment.node_id)?.worker_endpoint;
            // the worker beside an iroh coordinator is reached on the loopback address,
            // which is another machine for everybody else: they get the iroh address
            let endpoint = match &self.iroh_endpoint {
                Some(iroh) if is_loopback_endpoint(&endpoint) => bitty_protocol::iroh_transport::iroh_uri(
                    iroh.id(),
                    self.cluster_token.as_deref().unwrap_or_default(),
                ),
                _ => endpoint,
            };
            route.push(endpoint);
        }
        Some(route)
    }

    /// Sends one token through the workers: the first gets it from here, each hands its
    /// result to the next, and the last one keeps the logits for the coordinator. Returns
    /// them with the time each worker spent on its layers.
    async fn chain_token(
        worker_clients: &mut [WorkerRpcClient],
        route: &[String],
        activation: &ActivationTensor,
    ) -> Result<(ActivationTensor, Vec<u64>), Status> {
        let (first, rest) = worker_clients
            .split_first_mut()
            .ok_or_else(|| Status::failed_precondition("no workers registered"))?;
        let last = rest
            .last_mut()
            .ok_or_else(|| Status::failed_precondition("a chain needs two workers"))?;
        let step = ChainStep {
            activation: Some(ProtoActivationTensor::from(activation)),
            route: route[1..].to_vec(),
            error: String::new(),
            hop_micros: Vec::new(),
        };
        let limit = chain_timeout();
        let wait = async {
            tokio::time::timeout(
                limit,
                last.await_chain_result(&activation.request_id, activation.token_position),
            )
            .await
            .unwrap_or_else(|_| {
                Err(Status::deadline_exceeded(format!(
                    "no answer from the workers in {} seconds",
                    limit.as_secs()
                )))
            })
        };
        let (_, result) = tokio::try_join!(first.forward_chain(step), wait)?;
        if !result.error.is_empty() {
            return Err(Status::internal(result.error));
        }
        let output = result
            .activation
            .ok_or_else(|| Status::internal("the last worker returned no result"))?;
        let output =
            ActivationTensor::try_from(output).map_err(|err| Status::internal(err.to_string()))?;
        Ok((output, result.hop_micros))
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
        let codec = self.text_codec(model_path).await?;
        let text_error =
            |err: bitty_inference::StageProcessError| Status::internal(err.to_string());
        let mut current_input = if !request.prompt_tokens.is_empty() {
            request.prompt_tokens
        } else if request.raw {
            codec.tokenize(&request.prompt).map_err(text_error)?
        } else {
            codec
                .chat_tokens(&[("user", request.prompt.as_str())], request.thinking)
                .map_err(text_error)?
        };
        if current_input.is_empty() {
            return Err(Status::invalid_argument("the prompt is empty"));
        }
        let mut sampler = Sampler::new(SamplingOptions {
            temperature: request.temperature,
            top_k: if request.top_k == 0 {
                DEFAULT_TOP_K
            } else {
                request.top_k
            },
            top_p: if request.top_p <= 0.0 {
                DEFAULT_TOP_P
            } else {
                request.top_p
            },
            seed: request.seed,
        });
        let mut text = TextStream::default();

        let mut worker_clients = self.prepare_workers(&assignments, model_path).await?;

        let mut route = self.chain_route(&assignments).await;
        // the name the workers know the request by; it changes when the request starts again
        let mut run_id = request_id.clone();

        let mut outputs = Vec::with_capacity(max_new_tokens as usize);
        let mut failure = None;
        // prompt excluded, BITTY_TIMING=1 prints them: the time of each worker (its own
        // count when the workers pass the token on, else the wait of the coordinator for
        // it), and the whole time of the tokens
        let mut hop_micros = vec![0u64; worker_clients.len()];
        let mut token_micros = 0u64;
        let generation_started = Instant::now();
        let mut token_position = 0;
        while token_position < max_new_tokens {
            let started = Instant::now();
            let payload = current_input
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>();
            let mut activation = ActivationTensor::new(
                run_id.clone(),
                token_position,
                0,
                0,
                vec![current_input.len() as u32],
                ActivationDType::Tokens,
                payload,
            );

            if let Some(endpoints) = &route {
                match Self::chain_token(&mut worker_clients, endpoints, &activation).await {
                    Ok((output, micros)) => {
                        if token_position > 0 {
                            for (total, micros) in hop_micros.iter_mut().zip(micros) {
                                *total += micros;
                            }
                        }
                        activation = output;
                    }
                    // Nothing has been produced yet and no worker is still busy, so the
                    // request can start again the old way. After a timeout a worker may
                    // still be busy, and later in the text the workers hold a state
                    // that cannot be replayed.
                    Err(err)
                        if token_position == 0
                            && err.code() != tonic::Code::DeadlineExceeded =>
                    {
                        eprintln!(
                            "bitty-coordinator: the workers could not pass the token to each other ({}); every worker answers the coordinator until the cluster changes",
                            bitty_protocol::worker_client::status_text(&err)
                        );
                        *self.chain_broken.lock().await = Some(self.epoch().await);
                        route = None;
                        // a worker may still hold, or still be about to build, the state
                        // of the failed attempt: the new one must not meet it
                        for client in &mut worker_clients {
                            let _ = client
                                .cleanup(bitty_protocol::pb::CleanupRequest {
                                    request_id: run_id.clone(),
                                })
                                .await;
                        }
                        run_id = format!("{request_id}-direct");
                        continue;
                    }
                    Err(err) => {
                        failure = Some(err);
                        break;
                    }
                }
            } else {
                for (index, client) in worker_clients.iter_mut().enumerate() {
                    let hop_started = Instant::now();
                    match client.forward_activation(&activation).await {
                        Ok(output) => {
                            if token_position > 0 {
                                hop_micros[index] += hop_started.elapsed().as_micros() as u64;
                            }
                            activation = output
                        }
                        Err(err) => {
                            failure = Some(err);
                            break;
                        }
                    }
                }
                if failure.is_some() {
                    break;
                }
            }

            // the final stage answers with the logits of the last token: the best ones, or
            // the whole vocabulary from a worker that was told not to cut them
            let candidates = match activation.dtype {
                ActivationDType::TopLogits => {
                    bitty_inference::sampling::decode_top_logits(&activation.payload)
                }
                ActivationDType::F32
                    if activation.payload.len() == codec.vocab_size() as usize * 4 =>
                {
                    Some(
                        activation
                            .payload
                            .chunks_exact(4)
                            .enumerate()
                            .map(|(token, chunk)| {
                                let logit =
                                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                                (token as u32, logit)
                            })
                            .collect(),
                    )
                }
                _ => None,
            };
            let Some(candidates) = candidates else {
                failure = Some(Status::internal(
                    "the last worker did not return the logits of the model",
                ));
                break;
            };
            let Some(sampled) = sampler.sample_candidates(candidates) else {
                failure = Some(Status::internal("the model returned no usable logits"));
                break;
            };
            let ended = codec.is_end(sampled.token);
            let finished = ended || token_position + 1 == max_new_tokens;
            let mut piece = if ended {
                String::new()
            } else {
                match codec.piece(sampled.token) {
                    Ok(bytes) => text.push(&bytes),
                    Err(err) => {
                        failure = Some(text_error(err));
                        break;
                    }
                }
            };
            if finished {
                piece.push_str(&text.finish());
            }
            let token = ProtoTokenOutput {
                request_id: request_id.clone(),
                token_position,
                token_id: sampled.token,
                text: piece,
                finished,
                log_prob: sampled.log_prob,
                gen_latency_us: started.elapsed().as_micros() as u64,
            };
            if token_position > 0 {
                token_micros += started.elapsed().as_micros() as u64;
            }
            token_position += 1;
            current_input.clear();
            current_input.push(sampled.token);
            if let Some(sender) = &sender {
                if sender.send(Ok(token.clone())).await.is_err() {
                    break;
                }
            }
            outputs.push(token);
            if finished {
                break;
            }
        }

        if std::env::var_os("BITTY_TIMING").is_some() && outputs.len() > 1 {
            let steps = (outputs.len() - 1) as f64;
            let hops: Vec<String> = assignments
                .iter()
                .zip(&hop_micros)
                .map(|(assignment, micros)| {
                    format!(
                        "{} {}..{} {:.1}",
                        assignment.node_id.0,
                        assignment.range.start_layer,
                        assignment.range.end_layer_exclusive,
                        *micros as f64 / steps / 1000.0
                    )
                })
                .collect();
            let in_hops = hop_micros.iter().sum::<u64>() as f64 / steps / 1000.0;
            println!(
                "timing ({}): {} tokens in {:.1}s; ms per token after the first: {:.1}, of which workers {:.1} [{}]",
                if route.is_some() {
                    "worker to worker"
                } else {
                    "through the coordinator"
                },
                outputs.len(),
                generation_started.elapsed().as_secs_f64(),
                token_micros as f64 / steps / 1000.0,
                in_hops,
                hops.join(", ")
            );
        }

        for client in &mut worker_clients {
            let _ = client
                .cleanup(bitty_protocol::pb::CleanupRequest {
                    request_id: run_id.clone(),
                })
                .await;
            if run_id != request_id {
                let _ = client
                    .cleanup(bitty_protocol::pb::CleanupRequest {
                        request_id: request_id.clone(),
                    })
                    .await;
            }
        }

        match failure {
            Some(status) => Err(status),
            None => Ok(outputs),
        }
    }

    /// The vocabulary of the model, opened on first use. Text is handled here, on the machine
    /// that takes the request: workers only see token ids and hidden states.
    async fn text_codec(&self, model_path: &std::path::Path) -> Result<Arc<TextCodec>, Status> {
        let mut slot = self.text_codec.lock().await;
        if let Some(codec) = slot.as_ref() {
            return Ok(codec.clone());
        }
        let path = model_path.to_path_buf();
        let codec = tokio::task::spawn_blocking(move || TextCodec::open(&path))
            .await
            .map_err(|err| Status::internal(err.to_string()))?
            .map_err(|err| {
                Status::failed_precondition(format!(
                    "cannot open the vocabulary of {}: {err}",
                    model_path.display()
                ))
            })?;
        let codec = Arc::new(codec);
        *slot = Some(codec.clone());
        Ok(codec)
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

    #[test]
    fn loopback_endpoints_are_recognised() {
        assert!(is_loopback_endpoint("127.0.0.1:50061"));
        assert!(is_loopback_endpoint("http://localhost:50061"));
        assert!(is_loopback_endpoint("[::1]:50061"));
        assert!(!is_loopback_endpoint("192.168.1.20:50061"));
        assert!(!is_loopback_endpoint("iroh://abc?token=secret"));
    }

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
            ..Default::default()
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
