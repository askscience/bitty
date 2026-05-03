use bitty_coordinator::network::NetworkCoordinator;
use bitty_inference::BitNetLayerExecutor;
use bitty_protocol::iroh_transport::{
    self, IrohFrame, BITTY_SCHEDULER_ALPN, BITTY_WORKER_ALPN, DEFAULT_FRAME_LIMIT,
    SCHEDULER_CLUSTER_STATUS, SCHEDULER_GENERATE, SCHEDULER_HEARTBEAT, SCHEDULER_REGISTER_WORKER,
    WORKER_APPLY_TOPOLOGY, WORKER_CLEANUP, WORKER_FINAL_LOGITS, WORKER_FORWARD_ACTIVATION,
    WORKER_LOAD_SHARD,
};
use bitty_protocol::pb::coordinator_service_client::CoordinatorServiceClient;
use bitty_protocol::pb::coordinator_service_server::CoordinatorService;
use bitty_protocol::pb::worker_service_server::WorkerService;
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, CleanupRequest, ClusterStatusRequest,
    ClusterStatusResponse, GenerateRequest, GenerateResponse, HeartbeatRequest, HeartbeatResponse,
    LoadShardRequest, RegisterWorkerRequest, RegisterWorkerResponse, TopologyUpdate,
};
use bitty_protocol::{LayerMetadata, NodeId};
use bitty_worker::{network::NetworkWorker, HardwareProfiler};
use iroh::{endpoint::presets, Endpoint, EndpointId, SecretKey};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse(std::env::args().skip(1).collect()) {
        Ok(CliCommand::Node(config)) => run_node(config).await,
        Ok(CliCommand::Generate(config)) => run_generate(config).await,
        Ok(CliCommand::Chat(config)) => run_chat(config).await,
        Ok(CliCommand::Status(config)) => run_status(config).await,
        Ok(CliCommand::Help) => {
            print_help();
            Ok(())
        }
        Err(err) => {
            eprintln!("{err}");
            print_help();
            std::process::exit(2);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CliCommand {
    Node(NodeConfig),
    Generate(GenerateConfig),
    Chat(ChatConfig),
    Status(StatusConfig),
    Help,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NodeConfig {
    model: String,
    node_id: String,
    listen: String,
    worker_listen: Option<String>,
    public_endpoint: Option<String>,
    join: Option<String>,
    layers: u32,
    heartbeat_interval_ms: u64,
    iroh: bool,
    data_dir: Option<String>,
    cluster_token: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GenerateConfig {
    node: String,
    prompt: String,
    max_tokens: u32,
    temperature: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChatConfig {
    node: String,
    prompt: Option<String>,
    max_tokens: u32,
    temperature: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StatusConfig {
    node: String,
}

struct Cli;

impl Cli {
    fn parse(args: Vec<String>) -> Result<CliCommand, String> {
        let Some(command) = args.first().cloned() else {
            return Ok(CliCommand::Help);
        };
        let mut args = args.into_iter().skip(1);
        match command.as_str() {
            "node" => parse_node(&mut args).map(CliCommand::Node),
            "generate" => parse_generate(&mut args).map(CliCommand::Generate),
            "chat" => parse_chat(&mut args).map(CliCommand::Chat),
            "status" => parse_status(&mut args).map(CliCommand::Status),
            "-h" | "--help" | "help" => Ok(CliCommand::Help),
            other => Err(format!("unknown command: {other}")),
        }
    }
}

async fn run_node(config: NodeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let executor = Arc::new(BitNetLayerExecutor::load(&config.model).await?);
    let worker = NetworkWorker::new(NodeId::new(config.node_id.clone()), executor);

    let use_iroh = config.iroh || config.join.as_deref().is_some_and(is_iroh_join_target);
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    let iroh_node = if use_iroh {
        Some(start_iroh_node(&data_dir).await?)
    } else {
        None
    };
    let cluster_token = config
        .cluster_token
        .clone()
        .or_else(|| token_from_join(config.join.as_deref()))
        .unwrap_or_else(|| load_or_create_cluster_token(&data_dir));

    if !use_iroh {
        let worker_listen = config.worker_listen.clone().unwrap_or_else(|| {
            if config.join.is_some() {
                config.listen.clone()
            } else {
                "0.0.0.0:50061".into()
            }
        });
        let worker_server = worker.clone();
        let worker_addr = worker_listen.clone();
        tokio::spawn(async move {
            if let Err(err) = worker_server.serve(&worker_addr).await {
                eprintln!("bitty node worker stopped: {err}");
            }
        });
    }

    let mut scheduler_service = None;
    let scheduler_target = if let Some(join) = config.join.clone() {
        resolve_scheduler_target(&join, iroh_node.as_ref(), &cluster_token).await?
    } else {
        let leader = listen_as_connect_endpoint(&config.listen);
        let mut coordinator =
            NetworkCoordinator::new(demo_layers(config.layers)).with_model_path(&config.model);
        if let Some(iroh_node) = &iroh_node {
            coordinator =
                coordinator.with_iroh_endpoint(iroh_node.endpoint.clone(), cluster_token.clone());
            scheduler_service = Some(coordinator.clone());
        } else {
            let leader_addr = config.listen.clone();
            tokio::spawn(async move {
                if let Err(err) = coordinator.serve(&leader_addr).await {
                    eprintln!("bitty node scheduler stopped: {err}");
                }
            });
        }
        match &iroh_node {
            Some(iroh_node) => SchedulerTarget::Iroh {
                endpoint: iroh_node.endpoint.clone(),
                remote: iroh_node.endpoint.id(),
                token: cluster_token.clone(),
            },
            None => SchedulerTarget::Tcp(leader),
        }
    };

    if let Some(iroh_node) = &iroh_node {
        iroh_node.serve_protocols(scheduler_service, worker.clone(), cluster_token.clone());
        println!("iroh node id: {}", iroh_node.node_id);
        println!("iroh bound sockets: {}", iroh_node.bound_sockets.join(", "));
        if config.join.is_none() {
            println!(
                "bitty join: {}",
                iroh_transport::iroh_uri(&iroh_node.node_id, &cluster_token)
            );
        }
    } else if config.join.is_none() {
        println!(
            "bitty tcp join ticket: bitty://{leader}",
            leader = config.listen
        );
    }

    sleep(Duration::from_millis(300)).await;
    register_and_heartbeat(
        &scheduler_target,
        &config,
        iroh_node.as_ref(),
        &cluster_token,
    )
    .await
}

#[derive(Clone)]
enum SchedulerTarget {
    Tcp(String),
    Iroh {
        endpoint: Endpoint,
        remote: EndpointId,
        token: String,
    },
}

async fn register_and_heartbeat(
    target: &SchedulerTarget,
    config: &NodeConfig,
    iroh_node: Option<&IrohNode>,
    cluster_token: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut profile = HardwareProfiler::new(config.node_id.clone()).profile();
    profile.worker_endpoint = if let Some(iroh_node) = iroh_node {
        iroh_transport::iroh_uri(&iroh_node.node_id, cluster_token)
    } else {
        let worker_listen = config.worker_listen.clone().unwrap_or_else(|| {
            if config.join.is_some() {
                config.listen.clone()
            } else {
                "0.0.0.0:50061".into()
            }
        });
        config
            .public_endpoint
            .clone()
            .unwrap_or_else(|| public_endpoint_from_listen(&worker_listen))
    };

    match target {
        SchedulerTarget::Tcp(leader) => {
            let endpoint = normalize_endpoint(leader);
            let mut client = CoordinatorServiceClient::connect(endpoint.clone()).await?;
            let registration = client
                .register_worker(RegisterWorkerRequest {
                    profile: Some((&profile).into()),
                })
                .await?
                .into_inner();
            println!(
                "bitty node joined scheduler at {endpoint}; topology_epoch={} assignments={}",
                registration.topology_epoch,
                registration.assignments.len()
            );
            loop {
                sleep(Duration::from_millis(config.heartbeat_interval_ms)).await;
                let response = client
                    .heartbeat(HeartbeatRequest {
                        node_id: profile.node_id.0.clone(),
                        observed_tokens_per_second: profile.effective_compute_score(),
                    })
                    .await;
                match response {
                    Ok(response) => {
                        if !response.into_inner().accepted {
                            eprintln!(
                                "scheduler rejected heartbeat for node {}",
                                profile.node_id.0
                            );
                        }
                    }
                    Err(err) => eprintln!("heartbeat failed: {err}"),
                }
            }
        }
        SchedulerTarget::Iroh {
            endpoint,
            remote,
            token,
        } => {
            let client = IrohSchedulerClient {
                endpoint: endpoint.clone(),
                remote: *remote,
                token: token.clone(),
            };
            let registration: RegisterWorkerResponse = client
                .request(
                    SCHEDULER_REGISTER_WORKER,
                    &RegisterWorkerRequest {
                        profile: Some((&profile).into()),
                    },
                )
                .await?;
            println!(
                "bitty node joined iroh scheduler {}; topology_epoch={} assignments={}",
                remote,
                registration.topology_epoch,
                registration.assignments.len()
            );
            loop {
                sleep(Duration::from_millis(config.heartbeat_interval_ms)).await;
                let response = client
                    .request::<_, HeartbeatResponse>(
                        SCHEDULER_HEARTBEAT,
                        &HeartbeatRequest {
                            node_id: profile.node_id.0.clone(),
                            observed_tokens_per_second: profile.effective_compute_score(),
                        },
                    )
                    .await;
                match response {
                    Ok(response) => {
                        if !response.accepted {
                            eprintln!(
                                "scheduler rejected heartbeat for node {}",
                                profile.node_id.0
                            );
                        }
                    }
                    Err(err) => eprintln!("heartbeat failed: {err}"),
                }
            }
        }
    }
}

async fn run_generate(config: GenerateConfig) -> Result<(), Box<dyn std::error::Error>> {
    if let Some((endpoint_id, token)) = iroh_transport::parse_iroh_uri(&config.node) {
        let data_dir = bitty_data_dir(None);
        let iroh_node = start_iroh_node(&data_dir).await?;
        let client = IrohSchedulerClient {
            endpoint: iroh_node.endpoint,
            remote: endpoint_id.parse()?,
            token: token.unwrap_or_default().to_string(),
        };
        let response: GenerateResponse = client
            .request(
                SCHEDULER_GENERATE,
                &GenerateRequest {
                    request_id: request_id(),
                    prompt_tokens: Vec::new(),
                    prompt: config.prompt,
                    max_new_tokens: config.max_tokens,
                    temperature: config.temperature.parse().unwrap_or(0.0),
                },
            )
            .await?;
        for token in response.tokens {
            print!("{}", token.text);
            if token.finished {
                println!();
            }
        }
        return Ok(());
    }

    let mut client = CoordinatorServiceClient::connect(normalize_endpoint(&config.node)).await?;
    let mut stream = client
        .generate(GenerateRequest {
            request_id: request_id(),
            prompt_tokens: Vec::new(),
            prompt: config.prompt,
            max_new_tokens: config.max_tokens,
            temperature: config.temperature.parse().unwrap_or(0.0),
        })
        .await?
        .into_inner();

    while let Some(token) = stream.message().await? {
        print!("{}", token.text);
        if token.finished {
            println!();
        }
    }
    Ok(())
}

async fn run_chat(config: ChatConfig) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(prompt) = config.prompt {
        return run_generate(GenerateConfig {
            node: config.node,
            prompt,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
        })
        .await;
    }

    println!("Bitty chat. Type /exit to quit.");
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        if io::stdin().read_line(&mut prompt)? == 0 {
            break;
        }
        let prompt = prompt.trim();
        if prompt == "/exit" || prompt == "/quit" {
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        run_generate(GenerateConfig {
            node: config.node.clone(),
            prompt: prompt.into(),
            max_tokens: config.max_tokens,
            temperature: config.temperature.clone(),
        })
        .await?;
    }
    Ok(())
}

async fn run_status(config: StatusConfig) -> Result<(), Box<dyn std::error::Error>> {
    let status = if let Some((endpoint_id, token)) = iroh_transport::parse_iroh_uri(&config.node) {
        let data_dir = bitty_data_dir(None);
        let iroh_node = start_iroh_node(&data_dir).await?;
        let client = IrohSchedulerClient {
            endpoint: iroh_node.endpoint,
            remote: endpoint_id.parse()?,
            token: token.unwrap_or_default().to_string(),
        };
        client
            .request::<_, ClusterStatusResponse>(SCHEDULER_CLUSTER_STATUS, &ClusterStatusRequest {})
            .await?
    } else {
        let mut client =
            CoordinatorServiceClient::connect(normalize_endpoint(&config.node)).await?;
        client
            .cluster_status(ClusterStatusRequest {})
            .await?
            .into_inner()
    };
    println!("leader: {}", status.leader_id);
    println!("topology_epoch: {}", status.topology_epoch);
    println!("active_workers: {}", status.active_workers);
    println!("model_ready: {}", status.model_ready);
    if !status.model_path.is_empty() {
        println!("model: {}", status.model_path);
    }
    println!("assignments: {}", status.assignments.len());
    for assignment in status.assignments {
        let range = assignment.range;
        if let Some(range) = range {
            println!(
                "  {} layers {}..{} stage={} next={}",
                assignment.node_id,
                range.start_layer,
                range.end_layer_exclusive,
                assignment.model_stage,
                assignment.next_node_id
            );
        }
    }
    Ok(())
}

fn parse_node(args: &mut impl Iterator<Item = String>) -> Result<NodeConfig, String> {
    let mut config = NodeConfig {
        model: String::new(),
        node_id: default_node_id(),
        listen: "0.0.0.0:50051".into(),
        worker_listen: None,
        public_endpoint: None,
        join: None,
        layers: 30,
        heartbeat_interval_ms: 1000,
        iroh: true,
        data_dir: None,
        cluster_token: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => config.model = required_next(args, "--model")?,
            "--node-id" => config.node_id = required_next(args, "--node-id")?,
            "--listen" => config.listen = required_next(args, "--listen")?,
            "--worker-listen" => {
                config.worker_listen = Some(required_next(args, "--worker-listen")?)
            }
            "--public-endpoint" => {
                config.public_endpoint = Some(required_next(args, "--public-endpoint")?)
            }
            "--join" => config.join = Some(required_next(args, "--join")?),
            "--layers" => config.layers = parse_next(args, "--layers")?,
            "--heartbeat-interval-ms" => {
                config.heartbeat_interval_ms = parse_next(args, "--heartbeat-interval-ms")?
            }
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            "--cluster-token" => {
                config.cluster_token = Some(required_next(args, "--cluster-token")?)
            }
            "--no-iroh" => config.iroh = false,
            other => return Err(format!("unknown node argument: {other}")),
        }
    }
    if config.model.is_empty() {
        return Err("bitty node requires --model PATH".into());
    }
    Ok(config)
}

fn parse_generate(args: &mut impl Iterator<Item = String>) -> Result<GenerateConfig, String> {
    let mut config = GenerateConfig {
        node: "127.0.0.1:50051".into(),
        prompt: String::new(),
        max_tokens: 32,
        temperature: "0".into(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = required_next(args, "--node")?,
            "--prompt" => config.prompt = required_next(args, "--prompt")?,
            "--max-tokens" => config.max_tokens = parse_next(args, "--max-tokens")?,
            "--temperature" => config.temperature = required_next(args, "--temperature")?,
            other => return Err(format!("unknown generate argument: {other}")),
        }
    }
    if config.prompt.is_empty() {
        return Err("bitty generate requires --prompt TEXT".into());
    }
    Ok(config)
}

fn parse_chat(args: &mut impl Iterator<Item = String>) -> Result<ChatConfig, String> {
    let mut config = ChatConfig {
        node: "127.0.0.1:50051".into(),
        prompt: None,
        max_tokens: 128,
        temperature: "0.7".into(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = required_next(args, "--node")?,
            "--prompt" => config.prompt = Some(required_next(args, "--prompt")?),
            "--max-tokens" => config.max_tokens = parse_next(args, "--max-tokens")?,
            "--temperature" => config.temperature = required_next(args, "--temperature")?,
            other => return Err(format!("unknown chat argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_status(args: &mut impl Iterator<Item = String>) -> Result<StatusConfig, String> {
    let mut config = StatusConfig {
        node: "127.0.0.1:50051".into(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = required_next(args, "--node")?,
            other => return Err(format!("unknown status argument: {other}")),
        }
    }
    Ok(config)
}

fn required_next(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    required_next(args, name)?
        .parse()
        .map_err(|err| format!("invalid value for {name}: {err}"))
}

fn normalize_endpoint(endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.into()
    } else {
        format!("http://{endpoint}")
    }
}

struct IrohNode {
    endpoint: Endpoint,
    node_id: String,
    bound_sockets: Vec<String>,
}

impl IrohNode {
    fn serve_protocols(
        &self,
        coordinator: Option<NetworkCoordinator>,
        worker: NetworkWorker<BitNetLayerExecutor>,
        cluster_token: String,
    ) {
        let endpoint = self.endpoint.clone();
        tokio::spawn(async move {
            while let Some(incoming) = endpoint.accept().await {
                let coordinator = coordinator.clone();
                let worker = worker.clone();
                let cluster_token = cluster_token.clone();
                tokio::spawn(async move {
                    if let Err(err) =
                        handle_iroh_request(incoming, coordinator, worker, cluster_token).await
                    {
                        eprintln!("iroh request failed: {err}");
                    }
                });
            }
        });
    }
}

async fn start_iroh_node(data_dir: &PathBuf) -> Result<IrohNode, Box<dyn std::error::Error>> {
    let secret_key = load_or_create_iroh_secret(data_dir)?;
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret_key)
        .alpns(vec![
            BITTY_SCHEDULER_ALPN.to_vec(),
            BITTY_WORKER_ALPN.to_vec(),
        ])
        .bind()
        .await?;
    let node_id = endpoint.id().to_string();
    let bound_sockets = endpoint
        .bound_sockets()
        .into_iter()
        .map(|addr| addr.to_string())
        .collect();
    Ok(IrohNode {
        endpoint,
        node_id,
        bound_sockets,
    })
}

async fn handle_iroh_request(
    incoming: iroh::endpoint::Incoming,
    coordinator: Option<NetworkCoordinator>,
    worker: NetworkWorker<BitNetLayerExecutor>,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let connection = incoming.accept()?.await?;
    let alpn = connection.alpn().to_vec();
    if alpn == BITTY_WORKER_ALPN {
        return handle_worker_connection(connection, worker, cluster_token).await;
    }
    let Some(coordinator) = coordinator else {
        return Err("this node is not a scheduler leader".into());
    };
    let remote_id = connection.remote_id();
    handle_scheduler_connection(connection, remote_id, coordinator, cluster_token).await
}

async fn handle_scheduler_connection(
    connection: iroh::endpoint::Connection,
    remote_id: EndpointId,
    coordinator: NetworkCoordinator,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut send, mut recv) = connection.accept_bi().await?;
    let frame = iroh_transport::read_frame(&mut recv, DEFAULT_FRAME_LIMIT).await?;
    if frame.token != cluster_token {
        return Err("invalid cluster token".into());
    }
    let response = match frame.op {
        SCHEDULER_REGISTER_WORKER => {
            let mut request: RegisterWorkerRequest =
                frame.decode_message(SCHEDULER_REGISTER_WORKER)?;
            if let Some(profile) = request.profile.as_mut() {
                profile.worker_endpoint = iroh_transport::iroh_uri(remote_id, &cluster_token);
            }
            let response = coordinator
                .register_worker(tonic::Request::new(request))
                .await?;
            IrohFrame::message(SCHEDULER_REGISTER_WORKER, "", &response.into_inner())
        }
        SCHEDULER_HEARTBEAT => {
            let request: HeartbeatRequest = frame.decode_message(SCHEDULER_HEARTBEAT)?;
            let response = coordinator.heartbeat(tonic::Request::new(request)).await?;
            IrohFrame::message(SCHEDULER_HEARTBEAT, "", &response.into_inner())
        }
        SCHEDULER_GENERATE => {
            let request: GenerateRequest = frame.decode_message(SCHEDULER_GENERATE)?;
            let response = coordinator.generate(tonic::Request::new(request)).await?;
            let mut stream = response.into_inner();
            let mut tokens = Vec::new();
            while let Some(token) = futures::StreamExt::next(&mut stream).await {
                tokens.push(token?);
            }
            IrohFrame::message(SCHEDULER_GENERATE, "", &GenerateResponse { tokens })
        }
        SCHEDULER_CLUSTER_STATUS => {
            let request: ClusterStatusRequest = frame.decode_message(SCHEDULER_CLUSTER_STATUS)?;
            let response = coordinator
                .cluster_status(tonic::Request::new(request))
                .await?;
            IrohFrame::message(SCHEDULER_CLUSTER_STATUS, "", &response.into_inner())
        }
        _ => return Err(format!("unknown scheduler op {}", frame.op).into()),
    };
    iroh_transport::write_frame(&mut send, &response).await?;
    send.finish()?;
    Ok(())
}

async fn handle_worker_connection(
    connection: iroh::endpoint::Connection,
    worker: NetworkWorker<BitNetLayerExecutor>,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut send, mut recv) = connection.accept_bi().await?;
    let frame = iroh_transport::read_frame(&mut recv, DEFAULT_FRAME_LIMIT).await?;
    if frame.token != cluster_token {
        return Err("invalid cluster token".into());
    }
    let response = match frame.op {
        WORKER_FORWARD_ACTIVATION => {
            let request: ProtoActivationTensor = frame.decode_message(WORKER_FORWARD_ACTIVATION)?;
            let response = worker
                .forward_activation(tonic::Request::new(request))
                .await?;
            IrohFrame::message(WORKER_FORWARD_ACTIVATION, "", &response.into_inner())
        }
        WORKER_FINAL_LOGITS => {
            let request: ProtoActivationTensor = frame.decode_message(WORKER_FINAL_LOGITS)?;
            let response = worker.final_logits(tonic::Request::new(request)).await?;
            IrohFrame::message(WORKER_FINAL_LOGITS, "", &response.into_inner())
        }
        WORKER_APPLY_TOPOLOGY => {
            let request: TopologyUpdate = frame.decode_message(WORKER_APPLY_TOPOLOGY)?;
            let response = worker.apply_topology(tonic::Request::new(request)).await?;
            IrohFrame::message(WORKER_APPLY_TOPOLOGY, "", &response.into_inner())
        }
        WORKER_LOAD_SHARD => {
            let request: LoadShardRequest = frame.decode_message(WORKER_LOAD_SHARD)?;
            let response = worker.load_shard(tonic::Request::new(request)).await?;
            IrohFrame::message(WORKER_LOAD_SHARD, "", &response.into_inner())
        }
        WORKER_CLEANUP => {
            let request: CleanupRequest = frame.decode_message(WORKER_CLEANUP)?;
            let response = worker.cleanup(tonic::Request::new(request)).await?;
            IrohFrame::message(WORKER_CLEANUP, "", &response.into_inner())
        }
        _ => return Err(format!("unknown worker op {}", frame.op).into()),
    };
    iroh_transport::write_frame(&mut send, &response).await?;
    send.finish()?;
    Ok(())
}

struct IrohSchedulerClient {
    endpoint: Endpoint,
    remote: EndpointId,
    token: String,
}

impl IrohSchedulerClient {
    async fn request<M, R>(
        &self,
        op: u8,
        message: &M,
    ) -> Result<R, iroh_transport::IrohTransportError>
    where
        M: prost::Message,
        R: prost::Message + Default,
    {
        let response = iroh_transport::request(
            &self.endpoint,
            self.remote,
            BITTY_SCHEDULER_ALPN,
            IrohFrame::message(op, self.token.clone(), message),
            DEFAULT_FRAME_LIMIT,
        )
        .await?;
        response.decode_message(op)
    }
}

async fn resolve_scheduler_target(
    join: &str,
    iroh_node: Option<&IrohNode>,
    cluster_token: &str,
) -> Result<SchedulerTarget, Box<dyn std::error::Error>> {
    if let Some((endpoint_id, token)) = iroh_transport::parse_iroh_uri(join) {
        let endpoint_id: EndpointId = endpoint_id.parse()?;
        let Some(iroh_node) = iroh_node else {
            return Err("iroh join target requires Iroh to be enabled".into());
        };
        return Ok(SchedulerTarget::Iroh {
            endpoint: iroh_node.endpoint.clone(),
            remote: endpoint_id,
            token: token.unwrap_or(cluster_token).to_string(),
        });
    }
    Ok(SchedulerTarget::Tcp(parse_tcp_join_target(join)))
}

fn is_iroh_join_target(join: &str) -> bool {
    join.starts_with("iroh://")
}

fn token_from_join(join: Option<&str>) -> Option<String> {
    join.and_then(iroh_transport::parse_iroh_uri)
        .and_then(|(_, token)| token.map(str::to_string))
}

fn bitty_data_dir(configured: Option<&str>) -> PathBuf {
    configured
        .map(PathBuf::from)
        .or_else(|| std::env::var("BITTY_DATA_DIR").ok().map(PathBuf::from))
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| PathBuf::from(home).join(".bitty"))
        })
        .unwrap_or_else(|| PathBuf::from(".bitty"))
}

fn load_or_create_iroh_secret(data_dir: &PathBuf) -> Result<SecretKey, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(data_dir)?;
    let path = data_dir.join("iroh-secret.key");
    if path.exists() {
        let value = std::fs::read_to_string(path)?;
        let bytes = decode_hex_32(value.trim())?;
        return Ok(SecretKey::from_bytes(&bytes));
    }
    let secret = SecretKey::generate();
    std::fs::write(path, encode_hex(&secret.to_bytes()))?;
    Ok(secret)
}

fn load_or_create_cluster_token(data_dir: &PathBuf) -> String {
    let _ = std::fs::create_dir_all(data_dir);
    let path = data_dir.join("cluster-token");
    if let Ok(value) = std::fs::read_to_string(&path) {
        let token = value.trim();
        if !token.is_empty() {
            return token.to_string();
        }
    }
    let token = encode_hex(&SecretKey::generate().to_bytes());
    let _ = std::fs::write(path, &token);
    token
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_hex_32(value: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    if value.len() != 64 {
        return Err("expected 64 hex characters".into());
    }
    let mut bytes = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_value(chunk[0])?;
        let low = hex_value(chunk[1])?;
        bytes[index] = (high << 4) | low;
    }
    Ok(bytes)
}

fn hex_value(byte: u8) -> Result<u8, Box<dyn std::error::Error>> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err("invalid hex digit".into()),
    }
}

fn parse_tcp_join_target(join: &str) -> String {
    let Some(rest) = join.strip_prefix("bitty://") else {
        return join.into();
    };
    rest.split_once('?')
        .map(|(tcp, _)| tcp)
        .unwrap_or(rest)
        .to_string()
}

fn listen_as_connect_endpoint(listen: &str) -> String {
    if let Some(port) = listen.rsplit_once(':').map(|(_, port)| port) {
        format!("127.0.0.1:{port}")
    } else {
        listen.into()
    }
}

fn public_endpoint_from_listen(listen: &str) -> String {
    if let Some(port) = listen.rsplit_once(':').map(|(_, port)| port) {
        format!("127.0.0.1:{port}")
    } else {
        listen.into()
    }
}

fn default_node_id() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "bitty-node".into())
}

fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("bitty-{nanos:x}")
}

fn demo_layers(count: u32) -> Vec<LayerMetadata> {
    (0..count)
        .map(|layer_id| LayerMetadata {
            layer_id,
            weight_bytes: 512_000,
            activation_bytes: 4096,
            estimated_flops: 1e9,
            precision_critical: layer_id == 0 || layer_id + 1 == count,
        })
        .collect()
}

fn print_help() {
    println!("Usage:");
    println!("  bitty node --model PATH");
    println!("  bitty node --join 'iroh://LEADER_NODE_ID?token=TOKEN' --model PATH");
    println!("  bitty node --no-iroh --join HOST:PORT --model PATH");
    println!("  bitty generate --node 'iroh://LEADER_NODE_ID?token=TOKEN' --prompt TEXT");
    println!("  bitty chat --node 'iroh://LEADER_NODE_ID?token=TOKEN' [--prompt TEXT]");
    println!("  bitty status --node 'iroh://LEADER_NODE_ID?token=TOKEN'");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bootstrap_node() {
        let command = Cli::parse(vec![
            "node".into(),
            "--model".into(),
            "/m.gguf".into(),
            "--listen".into(),
            "0.0.0.0:50051".into(),
        ])
        .unwrap();
        match command {
            CliCommand::Node(config) => {
                assert_eq!(config.model, "/m.gguf");
                assert_eq!(config.join, None);
                assert_eq!(config.listen, "0.0.0.0:50051");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_join_node() {
        let command = Cli::parse(vec![
            "node".into(),
            "--join".into(),
            "10.0.0.1:50051".into(),
            "--model".into(),
            "/m.gguf".into(),
            "--listen".into(),
            "0.0.0.0:50061".into(),
        ])
        .unwrap();
        match command {
            CliCommand::Node(config) => {
                assert_eq!(config.join.as_deref(), Some("10.0.0.1:50051"));
                assert_eq!(config.listen, "0.0.0.0:50061");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_iroh_join_ticket_and_disable_flag() {
        let command = Cli::parse(vec![
            "node".into(),
            "--join".into(),
            "bitty://10.0.0.1:50051?iroh=node-id".into(),
            "--model".into(),
            "/m.gguf".into(),
            "--no-iroh".into(),
        ])
        .unwrap();
        match command {
            CliCommand::Node(config) => {
                assert_eq!(
                    config.join.as_deref(),
                    Some("bitty://10.0.0.1:50051?iroh=node-id")
                );
                assert!(!config.iroh);
                assert_eq!(
                    parse_tcp_join_target(config.join.as_deref().unwrap()),
                    "10.0.0.1:50051"
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_iroh_only_join_target() {
        let command = Cli::parse(vec![
            "node".into(),
            "--join".into(),
            "iroh://leader-node-id?token=secret".into(),
            "--model".into(),
            "/m.gguf".into(),
        ])
        .unwrap();
        match command {
            CliCommand::Node(config) => {
                assert_eq!(
                    config.join.as_deref(),
                    Some("iroh://leader-node-id?token=secret")
                );
                assert!(config.iroh);
                assert!(is_iroh_join_target(config.join.as_deref().unwrap()));
                assert_eq!(
                    token_from_join(config.join.as_deref()).as_deref(),
                    Some("secret")
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_generate() {
        let command = Cli::parse(vec![
            "generate".into(),
            "--node".into(),
            "127.0.0.1:50051".into(),
            "--prompt".into(),
            "Hello".into(),
        ])
        .unwrap();
        match command {
            CliCommand::Generate(config) => {
                assert_eq!(config.node, "127.0.0.1:50051");
                assert_eq!(config.prompt, "Hello");
                assert_eq!(config.max_tokens, 32);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn parses_status() {
        let command =
            Cli::parse(vec!["status".into(), "--node".into(), "node:50051".into()]).unwrap();
        match command {
            CliCommand::Status(config) => assert_eq!(config.node, "node:50051"),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn node_requires_model() {
        let err = Cli::parse(vec!["node".into()]).unwrap_err();
        assert!(err.contains("--model"));
    }

    #[test]
    fn same_machine_node_join_generate_smoke_commands_parse() {
        let leader = Cli::parse(vec![
            "node".into(),
            "--model".into(),
            "/m.gguf".into(),
            "--listen".into(),
            "127.0.0.1:50151".into(),
            "--worker-listen".into(),
            "127.0.0.1:50161".into(),
            "--public-endpoint".into(),
            "127.0.0.1:50161".into(),
        ])
        .unwrap();
        let joined = Cli::parse(vec![
            "node".into(),
            "--join".into(),
            "127.0.0.1:50151".into(),
            "--model".into(),
            "/m.gguf".into(),
            "--listen".into(),
            "127.0.0.1:50162".into(),
            "--public-endpoint".into(),
            "127.0.0.1:50162".into(),
        ])
        .unwrap();
        let generate = Cli::parse(vec![
            "generate".into(),
            "--node".into(),
            "127.0.0.1:50151".into(),
            "--prompt".into(),
            "Hello".into(),
            "--max-tokens".into(),
            "1".into(),
        ])
        .unwrap();

        assert!(matches!(leader, CliCommand::Node(_)));
        assert!(matches!(joined, CliCommand::Node(_)));
        assert!(matches!(generate, CliCommand::Generate(_)));
    }
}
