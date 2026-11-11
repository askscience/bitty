use crate::security::{request_rate_limiter, AuthMode, SharedRateLimiter};
use crate::worker_client::{connect_worker, WorkerRpcClient};
use crate::{Halda, Registry, SchedulerConfig};
use bitty_inference::{
    DraftModel, Sampler, SamplingOptions, StageOptions, TextCodec, TextStream, DEFAULT_TOP_K,
    DEFAULT_TOP_P,
};
#[cfg(test)]
use bitty_inference::{LayerExecutor, LowBitReferenceExecutor};
use bitty_observability::record_halda_run;
use bitty_protocol::endpoint::validate_worker_endpoint_for_dial;
use bitty_protocol::pb::coordinator_service_server::{
    CoordinatorService, CoordinatorServiceServer,
};
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, ChainStep, ClusterStatusRequest,
    ClusterStatusResponse, GenerateRequest as ProtoGenerateRequest, HeartbeatRequest,
    HeartbeatResponse, LoadShardRequest, RegisterWorkerRequest, RegisterWorkerResponse,
    ShardManifest as ProtoShardManifest, TokenOutput as ProtoTokenOutput,
};
use bitty_protocol::security::BITTY_TOKEN_HEADER;
use bitty_protocol::validation::MAX_ACTIVATION_PAYLOAD_BYTES;
use bitty_protocol::{
    ActivationDType, ActivationTensor, HardwareProfile, Heartbeat, LayerMetadata, NodeId,
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
    /// The small model that guesses tokens for the workers to check, opened on first use.
    draft_path: Option<PathBuf>,
    draft: Arc<Mutex<DraftSlot>>,
    /// The plan in use. It is kept while it holds, also across a restart of the leader.
    plan: Arc<Mutex<KeptPlan>>,
    /// The model as the nodes can fetch it, told to each node that registers.
    model_name: Option<String>,
    /// What each node last said it holds, and when that was logged.
    holdings: Arc<Mutex<HashMap<NodeId, (String, Instant)>>>,
}

/// How often the progress of a download on a node is logged.
const HOLDINGS_LOG_EVERY: Duration = Duration::from_secs(300);

/// What a node holds without the numbers that move: "fetching 10..34, 512 MiB so far" and
/// "fetching 10..34, 600 MiB so far" are the same state.
fn holdings_kind(holdings: &str) -> String {
    holdings
        .split("; ")
        .map(|part| part.split(", ").next().unwrap_or(part))
        .collect::<Vec<_>>()
        .join("; ")
}

/// The ranges of the last plan, and the file they are written to.
#[derive(Debug)]
struct KeptPlan {
    ranges: Vec<(NodeId, u32, u32)>,
    file: Option<PathBuf>,
    /// Until then a node of the plan that is missing is waited for.
    wait_until: Instant,
}

/// Nodes the leader gives no layers to: `BITTY_LEAVE_OUT`, node ids separated by commas.
/// They stay registered. The setting is read when a node registers.
fn left_out_nodes() -> Vec<String> {
    std::env::var("BITTY_LEAVE_OUT")
        .unwrap_or_default()
        .split(',')
        .map(|node| node.trim().to_string())
        .filter(|node| !node.is_empty())
        .collect()
}

/// How long the leader waits for a node of its plan to come back before it plans without
/// it: `BITTY_PLAN_WAIT_SECS`, 600 by default. A node that restarts for an update is back
/// well within that, and its layers stay where they are.
fn plan_wait_time() -> Duration {
    let seconds = std::env::var("BITTY_PLAN_WAIT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(600);
    Duration::from_secs(seconds)
}

fn read_plan(file: &std::path::Path) -> Vec<(NodeId, u32, u32)> {
    let Ok(contents) = std::fs::read_to_string(file) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let node = parts.next()?;
            let start = parts.next()?.parse().ok()?;
            let end = parts.next()?.parse().ok()?;
            Some((NodeId::new(node), start, end))
        })
        .collect()
}

type SharedDraft = Arc<std::sync::Mutex<DraftModel>>;

#[derive(Debug, Default)]
enum DraftSlot {
    #[default]
    NotOpened,
    Open(SharedDraft),
    /// It could not be opened or does not fit the model; not tried again.
    Unusable,
}

/// How many tokens the draft model guesses for one pass when the request does not say:
/// `BITTY_DRAFT_TOKENS`. None by default: whether a draft makes a ring faster depends on
/// how long a pass waits on the network, so it is asked for with the request.
fn default_draft_tokens() -> usize {
    std::env::var("BITTY_DRAFT_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
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
        None => authority
            .rsplit_once(':')
            .map_or(authority, |(host, _)| host),
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

/// How long a worker may stay silent before it is taken out of the plan
/// (`BITTY_EVICT_SECS`). Taking a node out moves the layers of every other node, which costs
/// them a download of several gigabytes, so the time has to cover what a real network does: a
/// heartbeat that crosses a relay while the same line is busy fetching layers can take longer
/// than a quarter of a minute without anything being wrong.
/// The number that follows `key` in a message, as in `free_memory_mb=2700`.
fn number_after(message: &str, key: &str) -> Option<u64> {
    let rest = &message[message.find(key)? + key.len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn eviction_time() -> Duration {
    let seconds = std::env::var("BITTY_EVICT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(90);
    Duration::from_secs(seconds)
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
            draft_path: None,
            draft: Arc::default(),
            plan: Arc::new(Mutex::new(KeptPlan {
                ranges: Vec::new(),
                file: None,
                wait_until: Instant::now(),
            })),
            model_name: None,
            holdings: Arc::default(),
        }
    }

    /// The name or URL of the model. A node that runs another model takes this one.
    pub fn with_model_name(mut self, name: Option<String>) -> Self {
        self.model_name = name;
        self
    }

    /// Where the plan is written. A leader that starts again finds it there and waits a
    /// while for the nodes it names, so that no layer has to move because of the restart.
    pub fn with_plan_file(self, file: impl Into<PathBuf>) -> Self {
        let file = file.into();
        let ranges = read_plan(&file);
        if let Ok(mut plan) = self.plan.try_lock() {
            if !ranges.is_empty() {
                plan.wait_until = Instant::now() + plan_wait_time();
            }
            plan.ranges = ranges;
            plan.file = Some(file);
        }
        self
    }

    /// A small model with the vocabulary of the served one. It guesses the next tokens and
    /// the workers check several of them in one pass.
    pub fn with_draft_model(mut self, draft_path: impl Into<PathBuf>) -> Self {
        self.draft_path = Some(draft_path.into());
        self
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
        self.clone().spawn_eviction_loop(eviction_time());
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
                    Ok(_) => continue,
                    Err(err) => {
                        println!("no plan for topology_epoch={seen}: {}", err.message());
                        // a plan that waits for a node is asked for again
                        if err.code() == tonic::Code::Unavailable {
                            prepared = String::new();
                            seen_at = Instant::now();
                        }
                        continue;
                    }
                };
                println!(
                    "preparing {} worker(s) for topology_epoch={seen}: {}",
                    assignments.len(),
                    assignments
                        .iter()
                        .map(|assignment| format!(
                            "{} {}..{}",
                            assignment.node_id.0,
                            assignment.range.start_layer,
                            assignment.range.end_layer_exclusive
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
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
                        Err(err) => {
                            println!(
                                "could not prepare the workers: {}; trying again",
                                err.message()
                            );
                            // the same plan is tried again after the settle time
                            prepared = String::new();
                            seen_at = Instant::now();
                        }
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
            let loaded = client
                .load_shard(LoadShardRequest {
                    manifest: Some(ProtoShardManifest::from(&manifest)),
                })
                .await;
            if let Err(err) = &loaded {
                let memory_mb = number_after(err.message(), "free_memory_mb=");
                let disk_mb = number_after(err.message(), "free_disk_mb=");
                if memory_mb.is_some() || disk_mb.is_some() {
                    println!(
                        "worker {} refused layers {}..{}: {}",
                        manifest.node_id.0,
                        manifest.range.start_layer,
                        manifest.range.end_layer_exclusive,
                        err.message()
                    );
                    self.registry
                        .lock()
                        .await
                        .note_free(&manifest.node_id, memory_mb, disk_mb);
                } else if err.message().contains("stage engine exited") {
                    // An engine that dies while it loads most often ran out of memory, and
                    // the node cannot say so: the next plan asks less of it. A second death
                    // is not about the size, and the node is left out.
                    let lowered_mb = manifest.byte_len / (1024 * 1024) * 3 / 4;
                    let left_out = self
                        .registry
                        .lock()
                        .await
                        .engine_died(&manifest.node_id, lowered_mb);
                    println!(
                        "the engine of worker {} died loading layers {}..{}: {}",
                        manifest.node_id.0,
                        manifest.range.start_layer,
                        manifest.range.end_layer_exclusive,
                        if left_out {
                            "it gets no layers until it runs another build".to_string()
                        } else {
                            format!("planning it with at most {lowered_mb} MB free")
                        }
                    );
                }
            }
            loaded.map_err(|err| {
                Status::failed_precondition(format!(
                    "worker {} could not load layers {}..{}: {}",
                    manifest.node_id.0,
                    manifest.range.start_layer,
                    manifest.range.end_layer_exclusive,
                    err.message()
                ))
            })?;
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
                    // a node that went away may be restarting: its layers wait for it
                    self.plan.lock().await.wait_until = Instant::now() + plan_wait_time();
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
        let halda = Halda::new(SchedulerConfig::default());
        let fresh = halda.assign(&profiles, &self.layers);
        record_halda_run(started.elapsed().as_secs_f64() * 1000.0);

        let mut plan = self.plan.lock().await;
        if !plan.ranges.is_empty() {
            let missing = plan
                .ranges
                .iter()
                .filter(|(node, _, _)| profiles.iter().all(|profile| &profile.node_id != node))
                .map(|(node, _, _)| node.0.clone())
                .collect::<Vec<_>>();
            if !missing.is_empty() && Instant::now() < plan.wait_until {
                return Err(Status::unavailable(format!(
                    "waiting {}s more for {} to come back with its layers",
                    plan.wait_until
                        .saturating_duration_since(Instant::now())
                        .as_secs(),
                    missing.join(", ")
                )));
            }
            // The plan stays unless a machine outside it would now get layers, or it no
            // longer fits its own machines. Measured speeds alone do not move layers.
            let no_new_node = fresh.as_ref().map_or(true, |fresh| {
                fresh.iter().all(|assignment| {
                    plan.ranges
                        .iter()
                        .any(|(node, _, _)| node == &assignment.node_id)
                })
            });
            let dropped = if !missing.is_empty() {
                format!("{} did not come back", missing.join(", "))
            } else if !no_new_node {
                "a machine outside it can take layers now".to_string()
            } else {
                match halda.keep(&plan.ranges, &profiles, &self.layers) {
                    Ok(kept) => return Ok(kept),
                    Err(err) => format!("it no longer fits its machines: {err}"),
                }
            };
            println!("the kept plan is dropped: {dropped}");
        }

        let assignments = fresh.map_err(|err| Status::failed_precondition(err.to_string()))?;
        let ranges = assignments
            .iter()
            .map(|assignment| {
                (
                    assignment.node_id.clone(),
                    assignment.range.start_layer,
                    assignment.range.end_layer_exclusive,
                )
            })
            .collect::<Vec<_>>();
        if ranges != plan.ranges {
            if let Some(file) = &plan.file {
                let text = ranges
                    .iter()
                    .map(|(node, start, end)| format!("{} {start} {end}\n", node.0))
                    .collect::<String>();
                if let Err(err) = std::fs::write(file, text) {
                    eprintln!(
                        "bitty-coordinator: the plan was not written to {}: {err}",
                        file.display()
                    );
                }
            }
            plan.ranges = ranges;
        }
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

    /// Logs what a node holds when it changes, and the progress of a download now and then.
    async fn note_holdings(&self, heartbeat: &Heartbeat) {
        if heartbeat.holdings.is_empty() {
            return;
        }
        let mut seen = self.holdings.lock().await;
        let fresh = match seen.get(&heartbeat.node_id) {
            Some((last, logged)) => {
                holdings_kind(last) != holdings_kind(&heartbeat.holdings)
                    || (*last != heartbeat.holdings && logged.elapsed() >= HOLDINGS_LOG_EVERY)
            }
            None => true,
        };
        if fresh {
            println!("{}: {}", heartbeat.node_id.0, heartbeat.holdings);
            seen.insert(
                heartbeat.node_id.clone(),
                (heartbeat.holdings.clone(), Instant::now()),
            );
        }
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
                Some(iroh) if is_loopback_endpoint(&endpoint) => {
                    bitty_protocol::iroh_transport::iroh_uri(
                        iroh.id(),
                        self.cluster_token.as_deref().unwrap_or_default(),
                    )
                }
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
        // Speculative decoding: a small model guesses the next tokens, the workers run the
        // last certain token and the guesses in one pass and answer for each of them. The
        // guesses are kept as long as the model chooses the same token; at the first other
        // one the workers go back. The text is the one the model would have written alone.
        let (mut draft, draft_tokens) = self
            .draft_for(request.draft_tokens, codec.vocab_size())
            .await;
        // the whole text: the prompt and every token chosen so far
        let mut text_tokens = std::mem::take(&mut current_input);
        // how many of them the workers hold
        let mut fed = 0usize;
        // the workers hold guesses after `fed` that were wrong
        let mut rewind = false;
        let (mut guessed, mut kept) = (0usize, 0usize);
        let mut first_pass_tokens = 0usize;
        let mut step = 0u32;
        let mut produced = 0u32;
        'passes: while produced < max_new_tokens {
            let started = Instant::now();
            let pending = text_tokens[fed..].to_vec();
            let mut guesses = Vec::new();
            // The workers can only take back tokens of their last pass, and not its first
            // one: a guess goes with one certain token, not with the prompt.
            if let (Some(model), 1) = (&draft, pending.len()) {
                let count =
                    draft_tokens.min((max_new_tokens - produced).saturating_sub(1) as usize);
                let (model, codec) = (model.clone(), codec.clone());
                let (request, text) = (request_id.clone(), text_tokens.clone());
                let proposed = tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().unwrap_or_else(|poison| poison.into_inner());
                    model.propose(&request, &text, count, |token| codec.is_end(token))
                })
                .await
                .map_err(|err| err.to_string())
                .and_then(|result| result.map_err(|err| err.to_string()));
                match proposed {
                    Ok(tokens) => guesses = tokens,
                    Err(err) => {
                        eprintln!(
                            "bitty-coordinator: the draft model failed ({err}); the request goes on without it"
                        );
                        draft = None;
                    }
                }
            }
            let payload = pending
                .iter()
                .chain(&guesses)
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>();
            let mut activation = ActivationTensor::new(
                run_id.clone(),
                step,
                0,
                0,
                vec![(pending.len() + guesses.len()) as u32],
                ActivationDType::Tokens,
                payload,
            );
            activation.every_row = !guesses.is_empty();
            if rewind {
                activation.rewind_to = fed as u32;
            }

            if let Some(endpoints) = &route {
                match Self::chain_token(&mut worker_clients, endpoints, &activation).await {
                    Ok((output, micros)) => {
                        if step > 0 {
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
                    Err(err) if step == 0 && err.code() != tonic::Code::DeadlineExceeded => {
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
                            if step > 0 {
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

            // The final stage answers with the logits of the last token: the best ones, or
            // the whole vocabulary from a worker that was told not to cut them. With guesses
            // in the pass it answers for every token from the last certain one on.
            let rows = match activation.dtype {
                ActivationDType::TopLogitRows => {
                    bitty_inference::sampling::decode_top_logit_rows(&activation.payload)
                }
                ActivationDType::TopLogits => {
                    bitty_inference::sampling::decode_top_logits(&activation.payload)
                        .map(|row| vec![row])
                }
                ActivationDType::F32
                    if activation.payload.len() == codec.vocab_size() as usize * 4 =>
                {
                    Some(vec![activation
                        .payload
                        .chunks_exact(4)
                        .enumerate()
                        .map(|(token, chunk)| {
                            let logit =
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            (token as u32, logit)
                        })
                        .collect()])
                }
                _ => None,
            };
            let Some(rows) = rows else {
                failure = Some(Status::internal(
                    "the last worker did not return the logits of the model",
                ));
                break;
            };
            if rows.len() != guesses.len() + 1 {
                failure = Some(Status::internal(if guesses.is_empty() {
                    "the last worker returned the logits of the wrong number of tokens"
                } else {
                    "the workers did not check the tokens of the draft model; a worker may run an older build of bitty"
                }));
                break;
            }

            let mut accepted = 0usize;
            let mut done = false;
            for (index, candidates) in rows.into_iter().enumerate() {
                let Some(sampled) = sampler.sample_candidates(candidates) else {
                    failure = Some(Status::internal("the model returned no usable logits"));
                    break 'passes;
                };
                let ended = codec.is_end(sampled.token);
                let finished = ended || produced + 1 == max_new_tokens;
                let mut piece = if ended {
                    String::new()
                } else {
                    match codec.piece(sampled.token) {
                        Ok(bytes) => text.push(&bytes),
                        Err(err) => {
                            failure = Some(text_error(err));
                            break 'passes;
                        }
                    }
                };
                if finished {
                    piece.push_str(&text.finish());
                }
                let token = ProtoTokenOutput {
                    request_id: request_id.clone(),
                    token_position: produced,
                    token_id: sampled.token,
                    text: piece,
                    finished,
                    log_prob: sampled.log_prob,
                    gen_latency_us: started.elapsed().as_micros() as u64,
                };
                produced += 1;
                text_tokens.push(sampled.token);
                if let Some(sender) = &sender {
                    if sender.send(Ok(token.clone())).await.is_err() {
                        break 'passes;
                    }
                }
                outputs.push(token);
                if finished {
                    done = true;
                    break;
                }
                // the guess was the model's own choice: the next row holds what follows it
                if guesses.get(index) == Some(&sampled.token) {
                    accepted += 1;
                } else {
                    break;
                }
            }
            if step > 0 {
                token_micros += started.elapsed().as_micros() as u64;
            } else {
                first_pass_tokens = outputs.len();
            }
            fed += pending.len() + accepted;
            rewind = accepted < guesses.len();
            guessed += guesses.len();
            kept += accepted;
            if let (Some(model), false) = (&draft, guesses.is_empty()) {
                model
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .settle(&request_id, accepted);
            }
            step += 1;
            if done {
                break;
            }
        }
        if let Some(model) = &draft {
            let (model, request) = (model.clone(), request_id.clone());
            let _ = tokio::task::spawn_blocking(move || {
                model
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .end(&request)
            })
            .await;
        }

        if std::env::var_os("BITTY_TIMING").is_some() && outputs.len() > first_pass_tokens {
            let steps = (outputs.len() - first_pass_tokens) as f64;
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
                "timing ({}): {} tokens in {:.1}s, {} passes, {} of {} draft tokens kept; ms per token after the first pass: {:.1}, of which workers {:.1} [{}]",
                if route.is_some() {
                    "worker to worker"
                } else {
                    "through the coordinator"
                },
                outputs.len(),
                generation_started.elapsed().as_secs_f64(),
                step,
                kept,
                guessed,
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

    /// The draft model and how many tokens it guesses for one pass, or `None` when the
    /// request goes without. `wanted` comes with the request: negative for no draft, 0 for the
    /// default. The workers cannot take back more tokens than their engines keep copies for.
    async fn draft_for(&self, wanted: i32, vocab_size: u32) -> (Option<SharedDraft>, usize) {
        let tokens = match wanted {
            wanted if wanted < 0 => 0,
            0 => default_draft_tokens(),
            wanted => wanted as usize,
        }
        .min(StageOptions::from_env().rollback as usize);
        let Some(path) = self.draft_path.clone() else {
            return (None, 0);
        };
        // the file may still be on its way
        if tokens == 0 || !path.is_file() {
            return (None, 0);
        }
        let mut slot = self.draft.lock().await;
        if let DraftSlot::NotOpened = *slot {
            let opened = tokio::task::spawn_blocking(move || {
                let engine = bitty_inference::find_stage_engine()
                    .ok_or_else(|| "bitty-stage was not found".to_string())?;
                let read = |name: &str| {
                    std::env::var(name)
                        .ok()
                        .and_then(|value| value.trim().parse::<u32>().ok())
                        .unwrap_or(0)
                };
                let model = DraftModel::open(
                    &engine,
                    &path,
                    read("BITTY_DRAFT_THREADS"),
                    read("BITTY_DRAFT_GPU_LAYERS"),
                )
                .map_err(|err| format!("{}: {err}", path.display()))?;
                if model.vocab_size() != vocab_size {
                    return Err(format!(
                        "{} has {} tokens in its vocabulary, the model has {vocab_size}",
                        path.display(),
                        model.vocab_size()
                    ));
                }
                Ok(model)
            })
            .await
            .map_err(|err| err.to_string())
            .and_then(|result| result);
            *slot = match opened {
                Ok(model) => {
                    println!("bitty-coordinator: draft model loaded, {tokens} tokens for one pass");
                    DraftSlot::Open(Arc::new(std::sync::Mutex::new(model)))
                }
                Err(err) => {
                    eprintln!("bitty-coordinator: no draft model: {err}");
                    DraftSlot::Unusable
                }
            };
        }
        match &*slot {
            DraftSlot::Open(model) => (Some(model.clone()), tokens),
            _ => (None, 0),
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
        let mut profile = HardwareProfile::try_from(profile)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        if !profile.worker_endpoint.is_empty() {
            validate_worker_endpoint_for_dial(&profile.worker_endpoint)
                .map_err(|err| Status::invalid_argument(err.to_string()))?;
        }
        let node_id = profile.node_id.clone();
        let left_out = left_out_nodes().contains(&node_id.0);
        if left_out {
            profile.layer_eligible = false;
        }
        let about = format!(
            "build={} os={} memory={} MB, free {} MB, disk for layers {} MB, at {}{}",
            if profile.build.is_empty() {
                "unknown"
            } else {
                profile.build.as_str()
            },
            profile.os,
            profile.ram_mb,
            profile.available_ram_mb,
            profile.free_disk_mb,
            if profile.worker_endpoint.is_empty() {
                "no address"
            } else {
                profile.worker_endpoint.as_str()
            },
            if left_out {
                ", left out by BITTY_LEAVE_OUT"
            } else {
                ""
            }
        );

        // A node that registers again has restarted and holds no layers: it is told to load.
        let cached = format!("{}@", node_id.0);
        self.shard_cache
            .lock()
            .await
            .retain(|key, _| !key.starts_with(&cached));
        self.registry.lock().await.register(profile);
        let topology_epoch = self.advance_epoch().await;
        // A network that cannot hold the model yet still takes the node: the next one that
        // joins may be what was missing.
        let assignments = self.current_assignments().await.unwrap_or_default();

        println!(
            "registered worker {node_id} ({about}); active_nodes={} assignments={}",
            self.registry.lock().await.profiles().len(),
            assignments.len()
        );

        Ok(Response::new(RegisterWorkerResponse {
            assignments: assignments.iter().map(Into::into).collect(),
            topology_epoch,
            model: self.model_name.clone().unwrap_or_default(),
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
        self.note_holdings(&heartbeat).await;
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
    fn the_progress_of_a_download_is_not_a_new_state() {
        let early = "holds 0..4; fetching 4..10, 512 MiB so far";
        let later = "holds 0..4; fetching 4..10, 900 MiB so far";
        assert_eq!(holdings_kind(early), holdings_kind(later));
        assert_eq!(holdings_kind(early), "holds 0..4; fetching 4..10");
        assert_ne!(
            holdings_kind(later),
            holdings_kind("holds 0..4; holds 4..10")
        );
    }

    #[test]
    fn a_refusal_carries_what_the_node_has_free() {
        let message =
            "these layers need 6300 MB and this machine can give 2430 MB now: free_memory_mb=2700";
        assert_eq!(number_after(message, "free_memory_mb="), Some(2700));
        assert_eq!(number_after(message, "free_disk_mb="), None);
        assert_eq!(
            number_after("no space: free_disk_mb=812, needed 5000", "free_disk_mb="),
            Some(812)
        );
        assert_eq!(number_after("free_memory_mb=", "free_memory_mb="), None);
    }

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
