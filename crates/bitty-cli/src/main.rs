mod logger;
mod model_store;
mod modelfile;
mod server;
mod settings;

use bitty_bitnet_runtime::BitNetRuntime;
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
use iroh::{endpoint::presets, Endpoint, EndpointAddr, EndpointId, SecretKey};
use model_store::{copy_model, installed_models, pull_model, remove_model, resolve_model};
use settings::BittySettings;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, timeout, Duration};

const DEFAULT_TCP_CLUSTER: &str = "127.0.0.1:50051";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    logger::log_default(format!("bitty command: {}", raw_args.join(" ")));
    let result = match Cli::parse(raw_args) {
        Ok(CliCommand::Node(config)) => run_node(config).await,
        Ok(CliCommand::Run(config)) => run_model(config).await,
        Ok(CliCommand::Pull(config)) => run_pull(config).await,
        Ok(CliCommand::List(config)) => run_list(config).await,
        Ok(CliCommand::Show(config)) => run_show(config).await,
        Ok(CliCommand::Ps(config)) => run_ps(config).await,
        Ok(CliCommand::Stop(config)) => run_stop(config).await,
        Ok(CliCommand::Serve(config)) => run_serve(config).await,
        Ok(CliCommand::Create(config)) => run_create(config).await,
        Ok(CliCommand::Rm(config)) => run_rm(config).await,
        Ok(CliCommand::Cp(config)) => run_cp(config).await,
        Ok(CliCommand::Settings(config)) => run_settings(config).await,
        Ok(CliCommand::Logs(config)) => run_logs(config).await,
        Ok(CliCommand::Cluster(config)) => run_cluster(config).await,
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
    };
    if let Err(err) = &result {
        logger::log_default(format!("bitty error: {err}"));
    }
    result
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CliCommand {
    Node(NodeConfig),
    Run(RunConfig),
    Pull(ModelCommand),
    List(DataDirConfig),
    Show(ModelCommand),
    Ps(DataDirConfig),
    Stop(ModelCommand),
    Serve(ServeConfig),
    Create(CreateConfig),
    Rm(ModelCommand),
    Cp(CpConfig),
    Settings(SettingsCommand),
    Logs(LogsConfig),
    Cluster(ClusterCommand),
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
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChatConfig {
    model: Option<String>,
    node: Option<String>,
    prompt: Option<String>,
    max_tokens: u32,
    temperature: String,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StatusConfig {
    node: Option<String>,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RunConfig {
    model: String,
    prompt: Option<String>,
    node: Option<String>,
    max_tokens: u32,
    temperature: String,
    data_dir: Option<String>,
    auto_pull: bool,
    local: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelCommand {
    model: String,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DataDirConfig {
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ServeConfig {
    data_dir: Option<String>,
    host: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CreateConfig {
    name: String,
    file: String,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CpConfig {
    source: String,
    dest: String,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SettingsCommand {
    Get {
        key: Option<String>,
        data_dir: Option<String>,
    },
    Set {
        key: String,
        value: String,
        data_dir: Option<String>,
    },
    Path {
        data_dir: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LogsConfig {
    data_dir: Option<String>,
    lines: usize,
    path: bool,
    clear: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ClusterCommand {
    Status(ClusterConfig),
    Nodes(ClusterConfig),
    Check(ClusterConfig),
    Invite(DataDirConfig),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClusterConfig {
    node: Option<String>,
    data_dir: Option<String>,
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
            "run" => parse_run(&mut args).map(CliCommand::Run),
            "pull" => parse_model_command(&mut args, "pull").map(CliCommand::Pull),
            "ls" | "list" => parse_data_dir(&mut args).map(CliCommand::List),
            "show" => parse_model_command(&mut args, "show").map(CliCommand::Show),
            "ps" => parse_data_dir(&mut args).map(CliCommand::Ps),
            "stop" => parse_model_command(&mut args, "stop").map(CliCommand::Stop),
            "serve" => parse_serve(&mut args).map(CliCommand::Serve),
            "create" => parse_create(&mut args).map(CliCommand::Create),
            "rm" => parse_model_command(&mut args, "rm").map(CliCommand::Rm),
            "cp" => parse_cp(&mut args).map(CliCommand::Cp),
            "settings" => parse_settings(&mut args).map(CliCommand::Settings),
            "logs" => parse_logs(&mut args).map(CliCommand::Logs),
            "cluster" => parse_cluster(&mut args).map(CliCommand::Cluster),
            "generate" => parse_generate(&mut args).map(CliCommand::Generate),
            "chat" => parse_chat(&mut args).map(CliCommand::Chat),
            "status" => parse_status(&mut args).map(CliCommand::Status),
            "-h" | "--help" | "help" => Ok(CliCommand::Help),
            other => Err(format!("unknown command: {other}")),
        }
    }
}

async fn run_node(config: NodeConfig) -> Result<(), Box<dyn std::error::Error>> {
    logger::log_default(format!(
        "starting node id={} model={} join={}",
        config.node_id,
        config.model,
        config.join.as_deref().unwrap_or("leader")
    ));
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
    if let Some(join) = &config.join {
        remember_active_cluster(&data_dir, join)?;
    }

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
            let coordinator_server = coordinator.clone();
            tokio::spawn(async move {
                if let Err(err) = coordinator_server.serve(&leader_addr).await {
                    eprintln!("bitty node scheduler stopped: {err}");
                }
            });
        }
        match &iroh_node {
            Some(_) => SchedulerTarget::Local(coordinator.clone()),
            None => SchedulerTarget::Tcp(leader),
        }
    };

    if let Some(iroh_node) = &iroh_node {
        iroh_node.serve_protocols(scheduler_service, worker.clone(), cluster_token.clone());
        println!("iroh node id: {}", iroh_node.node_id);
        println!("iroh bound sockets: {}", iroh_node.bound_sockets.join(", "));
        if config.join.is_none() {
            let invite =
                iroh_transport::iroh_uri_for_addr(&iroh_node.endpoint.addr(), &cluster_token);
            remember_active_cluster(&data_dir, &invite)?;
            println!("bitty join: {}", invite);
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

async fn run_model(config: RunConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    logger::log(
        &settings.data_dir,
        format!("running model {}", config.model),
    )?;
    let mut model = resolve_model(&settings, &config.model);
    if model.is_none() && config.auto_pull && settings.auto_pull {
        model = Some(pull_model(&settings, &config.model)?);
    }
    let model = model.ok_or_else(|| format!("model not found: {}", config.model))?;
    let prompt = config.prompt.unwrap_or_default();
    let cluster_node = if config.local {
        None
    } else {
        config
            .node
            .clone()
            .or_else(|| active_cluster_target(&settings))
    };
    if let Some(node) = cluster_node {
        if config.node.is_none() {
            println!("using cluster: {node}");
        }
        if prompt.is_empty() {
            println!("Bitty cluster chat: {}. Type /exit to quit.", model.id());
            loop {
                print!("> ");
                io::stdout().flush()?;
                let mut line = String::new();
                if io::stdin().read_line(&mut line)? == 0 {
                    break;
                }
                let line = line.trim();
                if line == "/exit" || line == "/quit" {
                    break;
                }
                if line.is_empty() {
                    continue;
                }
                run_generate(GenerateConfig {
                    node: node.clone(),
                    prompt: line.into(),
                    max_tokens: config.max_tokens,
                    temperature: config.temperature.clone(),
                    data_dir: config.data_dir.clone(),
                })
                .await?;
            }
            return Ok(());
        }
        return run_generate(GenerateConfig {
            node,
            prompt,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            data_dir: config.data_dir,
        })
        .await;
    }
    if prompt.is_empty() {
        println!("Bitty chat: {}. Type /exit to quit.", model.id());
        loop {
            print!("> ");
            io::stdout().flush()?;
            let mut line = String::new();
            if io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim();
            if line == "/exit" || line == "/quit" {
                break;
            }
            if line.is_empty() {
                continue;
            }
            run_local_model(
                &settings,
                &model,
                line,
                config.max_tokens,
                &config.temperature,
            )
            .await?;
        }
        return Ok(());
    }
    run_local_model(
        &settings,
        &model,
        &prompt,
        config.max_tokens,
        &config.temperature,
    )
    .await
}

async fn run_local_model(
    settings: &BittySettings,
    model: &model_store::ModelSpec,
    prompt: &str,
    max_tokens: u32,
    temperature: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = model.model_path(settings);
    if !path.exists() {
        return Err(format!(
            "model file is missing: {}. Run `bitty pull {}` first.",
            path.display(),
            model.id()
        )
        .into());
    }
    record_running_model(settings, &model.id())?;
    let runtime = BitNetRuntime::load(&path).await?;
    let text = runtime
        .generate_full(
            prompt,
            max_tokens as usize,
            temperature.parse().unwrap_or(model.temperature),
        )
        .await?;
    println!("{text}");
    Ok(())
}

async fn run_pull(config: ModelCommand) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    logger::log(
        &settings.data_dir,
        format!("pulling model {}", config.model),
    )?;
    let model = pull_model(&settings, &config.model)?;
    println!(
        "pulled {} to {}",
        model.id(),
        model.model_path(&settings).display()
    );
    Ok(())
}

async fn run_list(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    println!("NAME\tBACKEND\tQUANTIZATION\tPATH");
    for model in installed_models(&settings) {
        println!(
            "{}\t{}\t{}\t{}",
            model.id(),
            model.backend,
            model.quantization,
            model.model_path(&settings).display()
        );
    }
    Ok(())
}

async fn run_show(config: ModelCommand) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let model = resolve_model(&settings, &config.model)
        .ok_or_else(|| format!("model not found: {}", config.model))?;
    println!("name: {}", model.id());
    println!("display_name: {}", model.display_name);
    println!("backend: {}", model.backend);
    println!("quantization: {}", model.quantization);
    println!("layers: {}", model.layers);
    println!("path: {}", model.model_path(&settings).display());
    println!("source: {}", model.source);
    println!("temperature: {}", model.temperature);
    println!("num_predict: {}", model.num_predict);
    println!("num_ctx: {}", model.num_ctx);
    Ok(())
}

async fn run_ps(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let path = running_models_path(&settings);
    let contents = std::fs::read_to_string(path).unwrap_or_default();
    println!("NAME\tSTATUS");
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        println!("{line}\tloaded");
    }
    Ok(())
}

async fn run_stop(config: ModelCommand) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let path = running_models_path(&settings);
    let contents = std::fs::read_to_string(&path).unwrap_or_default();
    let next = contents
        .lines()
        .filter(|line| *line != config.model)
        .collect::<Vec<_>>()
        .join("\n");
    settings::ensure_parent(&path)?;
    std::fs::write(path, next)?;
    println!("stopped {}", config.model);
    Ok(())
}

async fn run_serve(config: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = load_settings(config.data_dir.as_deref());
    if let Some(host) = config.host {
        settings.api_host = host;
    }
    logger::log(
        &settings.data_dir,
        format!("starting api server on {}", settings.api_host),
    )?;
    server::serve(settings)
}

async fn run_create(config: CreateConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let model = modelfile::create_profile(&settings, &config.name, &PathBuf::from(config.file))?;
    println!("created {}", model.id());
    Ok(())
}

async fn run_rm(config: ModelCommand) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    remove_model(&settings, &config.model)?;
    println!("removed {}", config.model);
    Ok(())
}

async fn run_cp(config: CpConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    copy_model(&settings, &config.source, &config.dest)?;
    println!("copied {} to {}", config.source, config.dest);
    Ok(())
}

async fn run_settings(config: SettingsCommand) -> Result<(), Box<dyn std::error::Error>> {
    match config {
        SettingsCommand::Get { key, data_dir } => {
            let settings = load_settings(data_dir.as_deref());
            if let Some(key) = key {
                println!("{}", settings.get(&key).unwrap_or_default());
            } else {
                print!("{}", settings.to_toml());
            }
        }
        SettingsCommand::Set {
            key,
            value,
            data_dir,
        } => {
            let mut settings = load_settings(data_dir.as_deref());
            if !settings.set_value(&key, &value) {
                return Err(format!("unknown setting: {key}").into());
            }
            settings.save()?;
            println!("{key} = {value}");
        }
        SettingsCommand::Path { data_dir } => {
            let settings = load_settings(data_dir.as_deref());
            println!("{}", settings.path().display());
        }
    }
    Ok(())
}

async fn run_logs(config: LogsConfig) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    let path = logger::log_path(&data_dir);
    if config.path {
        println!("{}", path.display());
        return Ok(());
    }
    if config.clear {
        settings::ensure_parent(&path)?;
        std::fs::write(&path, "")?;
        logger::log(&data_dir, "logs cleared")?;
        println!("cleared {}", path.display());
        return Ok(());
    }
    logger::log(&data_dir, "logs read")?;
    match logger::read_last_lines(&path, config.lines) {
        Ok(lines) if !lines.is_empty() => println!("{lines}"),
        Ok(_) => println!("log is empty: {}", path.display()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            println!("log file does not exist yet: {}", path.display())
        }
        Err(err) => return Err(err.into()),
    }
    Ok(())
}

async fn run_cluster(config: ClusterCommand) -> Result<(), Box<dyn std::error::Error>> {
    match config {
        ClusterCommand::Status(config) => {
            let status =
                fetch_cluster_status(config.node.as_deref(), config.data_dir.as_deref()).await?;
            print_cluster_status(status, true);
        }
        ClusterCommand::Nodes(config) => {
            let status =
                fetch_cluster_status(config.node.as_deref(), config.data_dir.as_deref()).await?;
            print_cluster_nodes(status);
        }
        ClusterCommand::Check(config) => {
            let status =
                fetch_cluster_status(config.node.as_deref(), config.data_dir.as_deref()).await?;
            print_cluster_check(&status);
            if status.active_workers == 0 || !status.model_ready {
                return Err("cluster is not ready".into());
            }
        }
        ClusterCommand::Invite(config) => {
            let data_dir = bitty_data_dir(config.data_dir.as_deref());
            let token = load_or_create_cluster_token(&data_dir);
            let settings = BittySettings::load(data_dir.clone());
            let invite = if let Some(saved) = active_cluster_target(&settings) {
                relay_only_invite(&saved).unwrap_or(saved)
            } else {
                let iroh_node = start_iroh_node(&data_dir).await?;
                iroh_transport::iroh_uri_for_relay_addr(&iroh_node.endpoint.addr(), &token)
            };
            remember_active_cluster(&data_dir, &invite)?;
            println!("{invite}");
            println!("Share this invite with another Bitty node:");
            println!(
                "bitty node --join '{}' --model ~/.bitty/models/bitnet-b1.58/latest/ggml-model-i2_s.gguf",
                invite
            );
            println!("Keep `bitty node` running while other machines join.");
        }
    }
    Ok(())
}

#[derive(Clone)]
enum SchedulerTarget {
    Local(NetworkCoordinator),
    Tcp(String),
    Iroh {
        endpoint: Endpoint,
        remote: EndpointAddr,
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
    profile.model_path = config.model.clone();
    profile.worker_endpoint = if let Some(iroh_node) = iroh_node {
        iroh_transport::iroh_uri_for_addr(&iroh_node.endpoint.addr(), cluster_token)
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
        SchedulerTarget::Local(coordinator) => {
            let registration = coordinator
                .register_worker(tonic::Request::new(RegisterWorkerRequest {
                    profile: Some((&profile).into()),
                }))
                .await?
                .into_inner();
            println!(
                "bitty node started local scheduler; topology_epoch={} assignments={}",
                registration.topology_epoch,
                registration.assignments.len()
            );
            loop {
                sleep(Duration::from_millis(config.heartbeat_interval_ms)).await;
                let response = coordinator
                    .heartbeat(tonic::Request::new(HeartbeatRequest {
                        node_id: profile.node_id.0.clone(),
                        observed_tokens_per_second: profile.effective_compute_score(),
                    }))
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
                remote: remote.clone(),
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
                remote.id,
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
    let node = resolve_cluster_node(config.node.as_str(), config.data_dir.as_deref())?;
    if let Some(target) = iroh_transport::parse_iroh_target(&node) {
        let endpoint = start_iroh_client().await?;
        let client = IrohSchedulerClient {
            endpoint,
            remote: target.endpoint_addr,
            token: target.token.unwrap_or_default(),
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

    let mut client = CoordinatorServiceClient::connect(normalize_endpoint(&node)).await?;
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
    if let Some(model) = config.model {
        return run_model(RunConfig {
            model,
            prompt: config.prompt,
            node: config.node,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            data_dir: config.data_dir,
            auto_pull: true,
            local: false,
        })
        .await;
    }

    let node = resolve_cluster_node(
        config.node.as_deref().unwrap_or(""),
        config.data_dir.as_deref(),
    )?;
    if let Some(prompt) = config.prompt {
        return run_generate(GenerateConfig {
            node,
            prompt,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            data_dir: config.data_dir,
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
            node: node.clone(),
            prompt: prompt.into(),
            max_tokens: config.max_tokens,
            temperature: config.temperature.clone(),
            data_dir: config.data_dir.clone(),
        })
        .await?;
    }
    Ok(())
}

async fn run_status(config: StatusConfig) -> Result<(), Box<dyn std::error::Error>> {
    let status = fetch_cluster_status(config.node.as_deref(), config.data_dir.as_deref()).await?;
    print_cluster_status(status, true);
    Ok(())
}

async fn fetch_cluster_status(
    node: Option<&str>,
    data_dir: Option<&str>,
) -> Result<ClusterStatusResponse, Box<dyn std::error::Error>> {
    let node = resolve_cluster_node(node.unwrap_or(""), data_dir)?;
    let status = if let Some(target) = iroh_transport::parse_iroh_target(&node) {
        let endpoint = start_iroh_client().await?;
        let client = IrohSchedulerClient {
            endpoint,
            remote: target.endpoint_addr,
            token: target.token.unwrap_or_default(),
        };
        client
            .request::<_, ClusterStatusResponse>(SCHEDULER_CLUSTER_STATUS, &ClusterStatusRequest {})
            .await?
    } else {
        let mut client = CoordinatorServiceClient::connect(normalize_endpoint(&node)).await?;
        client
            .cluster_status(ClusterStatusRequest {})
            .await?
            .into_inner()
    };
    Ok(status)
}

fn print_cluster_status(status: ClusterStatusResponse, include_assignments: bool) {
    println!("Bitty cluster");
    println!("-------------");
    println!("leader: {}", status.leader_id);
    println!("epoch: {}", status.topology_epoch);
    println!("workers: {}", status.active_workers);
    println!("model ready: {}", yes_no(status.model_ready));
    if !status.model_path.is_empty() {
        println!("model: {}", status.model_path);
    }
    println!("assignments: {}", status.assignments.len());
    if !include_assignments {
        return;
    }
    if !status.assignments.is_empty() {
        println!();
        println!("Assignments");
    }
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
}

fn print_cluster_nodes(status: ClusterStatusResponse) {
    println!("Bitty nodes");
    println!("-----------");
    println!("NODE\tLAYERS\tSTAGE\tNEXT");
    for assignment in status.assignments {
        if let Some(range) = assignment.range {
            println!(
                "{}\t{}..{}\t{}\t{}",
                assignment.node_id,
                range.start_layer,
                range.end_layer_exclusive,
                assignment.model_stage,
                assignment.next_node_id
            );
        }
    }
}

fn print_cluster_check(status: &ClusterStatusResponse) {
    println!("Bitty cluster check");
    println!("-------------------");
    println!(
        "status: {}",
        if status.active_workers > 0 && status.model_ready {
            "ready"
        } else {
            "not ready"
        }
    );
    println!("leader: {}", status.leader_id);
    println!("workers: {}", status.active_workers);
    println!("model ready: {}", yes_no(status.model_ready));
    println!("assignments: {}", status.assignments.len());
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
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
        node: String::new(),
        prompt: String::new(),
        max_tokens: 32,
        temperature: "0".into(),
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = required_next(args, "--node")?,
            "--prompt" => config.prompt = required_next(args, "--prompt")?,
            "--max-tokens" => config.max_tokens = parse_next(args, "--max-tokens")?,
            "--temperature" => config.temperature = required_next(args, "--temperature")?,
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
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
        model: None,
        node: None,
        prompt: None,
        max_tokens: 128,
        temperature: "0.7".into(),
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = Some(required_next(args, "--node")?),
            "--prompt" => config.prompt = Some(required_next(args, "--prompt")?),
            "--max-tokens" => config.max_tokens = parse_next(args, "--max-tokens")?,
            "--temperature" => config.temperature = required_next(args, "--temperature")?,
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.model.is_none() => config.model = Some(value.into()),
            other => return Err(format!("unknown chat argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_status(args: &mut impl Iterator<Item = String>) -> Result<StatusConfig, String> {
    let mut config = StatusConfig {
        node: None,
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = Some(required_next(args, "--node")?),
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            other => return Err(format!("unknown status argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_run(args: &mut impl Iterator<Item = String>) -> Result<RunConfig, String> {
    let mut config = RunConfig {
        model: String::new(),
        prompt: None,
        node: None,
        max_tokens: 128,
        temperature: "0.7".into(),
        data_dir: None,
        auto_pull: true,
        local: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = Some(required_next(args, "--node")?),
            "--prompt" => config.prompt = Some(required_next(args, "--prompt")?),
            "--max-tokens" | "--num-predict" => {
                config.max_tokens = parse_next(args, "--max-tokens")?
            }
            "--temperature" => config.temperature = required_next(args, "--temperature")?,
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            "--no-auto-pull" => config.auto_pull = false,
            "--local" => config.local = true,
            "--no-daemon" => {}
            "--join" => config.node = Some(required_next(args, "--join")?),
            "--num-ctx" | "--seed" | "--top-k" | "--top-p" | "--system" | "--template" => {
                let _ = required_next(args, arg.as_str()).ok();
            }
            value if config.model.is_empty() => config.model = value.into(),
            value => config.prompt = Some(value.into()),
        }
    }
    if config.model.is_empty() {
        return Err("bitty run requires MODEL".into());
    }
    Ok(config)
}

fn parse_model_command(
    args: &mut impl Iterator<Item = String>,
    command: &str,
) -> Result<ModelCommand, String> {
    let mut config = ModelCommand {
        model: String::new(),
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.model.is_empty() => config.model = value.into(),
            other => return Err(format!("unknown {command} argument: {other}")),
        }
    }
    if config.model.is_empty() {
        return Err(format!("bitty {command} requires MODEL"));
    }
    Ok(config)
}

fn parse_data_dir(args: &mut impl Iterator<Item = String>) -> Result<DataDirConfig, String> {
    let mut config = DataDirConfig { data_dir: None };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_serve(args: &mut impl Iterator<Item = String>) -> Result<ServeConfig, String> {
    let mut config = ServeConfig {
        data_dir: None,
        host: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            "--host" | "--api-host" => config.host = Some(required_next(args, "--host")?),
            other => return Err(format!("unknown serve argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_create(args: &mut impl Iterator<Item = String>) -> Result<CreateConfig, String> {
    let mut config = CreateConfig {
        name: String::new(),
        file: "Modelfile".into(),
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-f" | "--file" => config.file = required_next(args, "--file")?,
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.name.is_empty() => config.name = value.into(),
            other => return Err(format!("unknown create argument: {other}")),
        }
    }
    if config.name.is_empty() {
        return Err("bitty create requires NAME".into());
    }
    Ok(config)
}

fn parse_cp(args: &mut impl Iterator<Item = String>) -> Result<CpConfig, String> {
    let mut config = CpConfig {
        source: String::new(),
        dest: String::new(),
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.source.is_empty() => config.source = value.into(),
            value if config.dest.is_empty() => config.dest = value.into(),
            other => return Err(format!("unknown cp argument: {other}")),
        }
    }
    if config.source.is_empty() || config.dest.is_empty() {
        return Err("bitty cp requires SOURCE DEST".into());
    }
    Ok(config)
}

fn parse_settings(args: &mut impl Iterator<Item = String>) -> Result<SettingsCommand, String> {
    let action = args.next().unwrap_or_else(|| "get".into());
    let mut data_dir = None;
    match action.as_str() {
        "get" => {
            let mut key = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--data-dir" => data_dir = Some(required_next(args, "--data-dir")?),
                    value => key = Some(value.into()),
                }
            }
            Ok(SettingsCommand::Get { key, data_dir })
        }
        "set" => {
            let key = required_next(args, "key")?;
            let value = required_next(args, "value")?;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--data-dir" => data_dir = Some(required_next(args, "--data-dir")?),
                    other => return Err(format!("unknown settings set argument: {other}")),
                }
            }
            Ok(SettingsCommand::Set {
                key,
                value,
                data_dir,
            })
        }
        "path" => {
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--data-dir" => data_dir = Some(required_next(args, "--data-dir")?),
                    other => return Err(format!("unknown settings path argument: {other}")),
                }
            }
            Ok(SettingsCommand::Path { data_dir })
        }
        other => Err(format!("unknown settings command: {other}")),
    }
}

fn parse_logs(args: &mut impl Iterator<Item = String>) -> Result<LogsConfig, String> {
    let mut config = LogsConfig {
        data_dir: None,
        lines: 80,
        path: false,
        clear: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            "--lines" | "-n" => config.lines = parse_next(args, "--lines")?,
            "--path" => config.path = true,
            "--clear" => config.clear = true,
            other => return Err(format!("unknown logs argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_cluster(args: &mut impl Iterator<Item = String>) -> Result<ClusterCommand, String> {
    let action = args.next().unwrap_or_else(|| "status".into());
    match action.as_str() {
        "status" => parse_cluster_config(args).map(ClusterCommand::Status),
        "nodes" => parse_cluster_config(args).map(ClusterCommand::Nodes),
        "check" => parse_cluster_config(args).map(ClusterCommand::Check),
        "invite" => parse_data_dir(args).map(ClusterCommand::Invite),
        other => Err(format!("unknown cluster command: {other}")),
    }
}

fn parse_cluster_config(args: &mut impl Iterator<Item = String>) -> Result<ClusterConfig, String> {
    let mut config = ClusterConfig {
        node: None,
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = Some(required_next(args, "--node")?),
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            other => return Err(format!("unknown cluster argument: {other}")),
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
    let _ = timeout(Duration::from_secs(10), endpoint.online()).await;
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

async fn start_iroh_client() -> Result<Endpoint, Box<dyn std::error::Error>> {
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(SecretKey::generate())
        .alpns(vec![
            BITTY_SCHEDULER_ALPN.to_vec(),
            BITTY_WORKER_ALPN.to_vec(),
        ])
        .bind()
        .await?;
    let _ = timeout(Duration::from_secs(10), endpoint.online()).await;
    Ok(endpoint)
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
    let response = match handle_scheduler_frame(frame, remote_id, coordinator, &cluster_token).await
    {
        Ok(response) => response,
        Err(err) => iroh_transport::error_frame(err),
    };
    iroh_transport::write_frame(&mut send, &response).await?;
    send.finish()?;
    send.stopped().await?;
    Ok(())
}

async fn handle_scheduler_frame(
    frame: IrohFrame,
    remote_id: EndpointId,
    coordinator: NetworkCoordinator,
    cluster_token: &str,
) -> Result<IrohFrame, Box<dyn std::error::Error>> {
    let response = match frame.op {
        SCHEDULER_REGISTER_WORKER => {
            let mut request: RegisterWorkerRequest =
                frame.decode_message(SCHEDULER_REGISTER_WORKER)?;
            if let Some(profile) = request.profile.as_mut() {
                if profile.worker_endpoint.is_empty() {
                    profile.worker_endpoint = iroh_transport::iroh_uri(remote_id, &cluster_token);
                }
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
    Ok(response)
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
    let response = match handle_worker_frame(frame, worker).await {
        Ok(response) => response,
        Err(err) => iroh_transport::error_frame(err),
    };
    iroh_transport::write_frame(&mut send, &response).await?;
    send.finish()?;
    send.stopped().await?;
    Ok(())
}

async fn handle_worker_frame(
    frame: IrohFrame,
    worker: NetworkWorker<BitNetLayerExecutor>,
) -> Result<IrohFrame, Box<dyn std::error::Error>> {
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
    Ok(response)
}

struct IrohSchedulerClient {
    endpoint: Endpoint,
    remote: EndpointAddr,
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
        let response = iroh_transport::request_addr(
            &self.endpoint,
            self.remote.clone(),
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
    if let Some(target) = iroh_transport::parse_iroh_target(join) {
        let Some(iroh_node) = iroh_node else {
            return Err("iroh join target requires Iroh to be enabled".into());
        };
        return Ok(SchedulerTarget::Iroh {
            endpoint: iroh_node.endpoint.clone(),
            remote: target.endpoint_addr,
            token: target.token.unwrap_or_else(|| cluster_token.to_string()),
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

fn active_cluster_target(settings: &BittySettings) -> Option<String> {
    let target = settings.active_cluster.trim();
    (!target.is_empty()).then(|| target.to_string())
}

fn relay_only_invite(value: &str) -> Option<String> {
    let target = iroh_transport::parse_iroh_target(value)?;
    Some(iroh_transport::iroh_uri_for_relay_addr(
        &target.endpoint_addr,
        target.token.as_deref().unwrap_or_default(),
    ))
}

fn resolve_cluster_node(
    explicit: &str,
    data_dir: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    if !explicit.trim().is_empty() {
        return Ok(explicit.trim().to_string());
    }
    let settings = load_settings(data_dir);
    active_cluster_target(&settings).ok_or_else(|| {
        format!(
            "no active Bitty cluster saved. Start one with `bitty node --model PATH`, join one with `bitty node --join INVITE --model PATH`, or pass `--node {DEFAULT_TCP_CLUSTER}`."
        )
        .into()
    })
}

fn remember_active_cluster(
    data_dir: &PathBuf,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = BittySettings::load(data_dir.clone());
    settings.active_cluster = target.to_string();
    settings.save()?;
    Ok(())
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

fn load_settings(configured_data_dir: Option<&str>) -> BittySettings {
    let data_dir = bitty_data_dir(configured_data_dir);
    let settings = BittySettings::load(data_dir);
    let _ = settings.save();
    settings
}

fn running_models_path(settings: &BittySettings) -> PathBuf {
    settings.data_dir.join("state").join("running-models")
}

fn record_running_model(settings: &BittySettings, model: &str) -> std::io::Result<()> {
    let path = running_models_path(settings);
    settings::ensure_parent(&path)?;
    let contents = std::fs::read_to_string(&path).unwrap_or_default();
    if contents.lines().any(|line| line == model) {
        return Ok(());
    }
    let mut next = contents;
    if !next.is_empty() && !next.ends_with('\n') {
        next.push('\n');
    }
    next.push_str(model);
    next.push('\n');
    std::fs::write(path, next)
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
        .or_else(|_| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|hostname| hostname.trim().to_string())
                .filter(|hostname| !hostname.is_empty())
                .ok_or(std::env::VarError::NotPresent)
        })
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
    println!("  bitty run MODEL [PROMPT] [--local|--node TARGET]");
    println!("  bitty pull MODEL");
    println!("  bitty ls | bitty list");
    println!("  bitty show MODEL");
    println!("  bitty ps");
    println!("  bitty stop MODEL");
    println!("  bitty serve [--host 127.0.0.1:11435]");
    println!("  bitty create NAME -f Modelfile");
    println!("  bitty rm MODEL");
    println!("  bitty cp SOURCE DEST");
    println!("  bitty settings get|set|path");
    println!("  bitty logs [--lines N|--path|--clear]");
    println!("  bitty cluster status|nodes|check|invite [--node TARGET]");
    println!("  bitty node --model PATH");
    println!("  bitty node --join 'iroh://INVITE' --model PATH");
    println!("  bitty node --no-iroh --join HOST:PORT --model PATH");
    println!("  bitty generate --prompt TEXT [--node TARGET]");
    println!("  bitty chat [MODEL] [--prompt TEXT] [--node TARGET]");
    println!("  bitty status [--node TARGET]");
    println!();
    println!("Bitty remembers the active cluster after `bitty node` or `bitty cluster invite`.");
    println!("Use `--local` on `bitty run` to force local-only generation.");
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
    fn parses_ollama_style_commands() {
        assert!(matches!(
            Cli::parse(vec![
                "run".into(),
                "bitnet-b1.58".into(),
                "hello".into(),
                "--num-predict".into(),
                "4".into()
            ])
            .unwrap(),
            CliCommand::Run(RunConfig { max_tokens: 4, .. })
        ));
        assert!(matches!(
            Cli::parse(vec!["run".into(), "bitnet-b1.58".into(), "--local".into()]).unwrap(),
            CliCommand::Run(RunConfig { local: true, .. })
        ));
        assert!(matches!(
            Cli::parse(vec!["pull".into(), "bitnet-b1.58".into()]).unwrap(),
            CliCommand::Pull(ModelCommand { .. })
        ));
        assert!(matches!(
            Cli::parse(vec!["ls".into()]).unwrap(),
            CliCommand::List(DataDirConfig { .. })
        ));
        assert!(matches!(
            Cli::parse(vec![
                "serve".into(),
                "--host".into(),
                "127.0.0.1:11435".into()
            ])
            .unwrap(),
            CliCommand::Serve(ServeConfig { .. })
        ));
    }

    #[test]
    fn parses_settings_and_create_commands() {
        assert!(matches!(
            Cli::parse(vec![
                "settings".into(),
                "set".into(),
                "default_model".into(),
                "bitnet-b1.58".into()
            ])
            .unwrap(),
            CliCommand::Settings(SettingsCommand::Set { .. })
        ));
        assert!(matches!(
            Cli::parse(vec![
                "create".into(),
                "my-model".into(),
                "-f".into(),
                "Modelfile".into()
            ])
            .unwrap(),
            CliCommand::Create(CreateConfig { .. })
        ));
    }

    #[test]
    fn parses_logs_and_cluster_commands() {
        assert!(matches!(
            Cli::parse(vec!["logs".into(), "--lines".into(), "10".into()]).unwrap(),
            CliCommand::Logs(LogsConfig { lines: 10, .. })
        ));
        assert!(matches!(
            Cli::parse(vec![
                "cluster".into(),
                "nodes".into(),
                "--node".into(),
                "iroh://abc?token=secret".into()
            ])
            .unwrap(),
            CliCommand::Cluster(ClusterCommand::Nodes(ClusterConfig { .. }))
        ));
        assert!(matches!(
            Cli::parse(vec!["cluster".into(), "invite".into()]).unwrap(),
            CliCommand::Cluster(ClusterCommand::Invite(DataDirConfig { .. }))
        ));
    }

    #[test]
    fn parses_status() {
        let command =
            Cli::parse(vec!["status".into(), "--node".into(), "node:50051".into()]).unwrap();
        match command {
            CliCommand::Status(config) => assert_eq!(config.node.as_deref(), Some("node:50051")),
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
