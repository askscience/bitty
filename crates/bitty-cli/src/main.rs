mod cluster_store;
mod logger;
mod model_store;
mod modelfile;
mod secrets;
mod server;
mod settings;

use bitty_coordinator::network::NetworkCoordinator;
use bitty_inference::{StageExecutor, STAGE_BACKEND_ID};
use bitty_protocol::endpoint::normalize_endpoint;
use bitty_protocol::iroh_transport::{
    self, IrohFrame, BITTY_SCHEDULER_ALPN, BITTY_WORKER_ALPN, DEFAULT_FRAME_LIMIT,
    SCHEDULER_CLUSTER_STATUS, SCHEDULER_GENERATE, SCHEDULER_HEARTBEAT, SCHEDULER_REGISTER_WORKER,
    WORKER_APPLY_TOPOLOGY, WORKER_AWAIT_CHAIN_RESULT, WORKER_CLEANUP, WORKER_FINAL_LOGITS,
    WORKER_FORWARD_ACTIVATION, WORKER_FORWARD_CHAIN, WORKER_LOAD_SHARD, WORKER_SAMPLE_TOKEN,
};
use bitty_protocol::pb::coordinator_service_client::CoordinatorServiceClient;
use bitty_protocol::pb::coordinator_service_server::CoordinatorService;
use bitty_protocol::pb::worker_service_server::WorkerService;
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, ChainResultRequest, ChainStep, CleanupRequest,
    ClusterStatusRequest, ClusterStatusResponse, GenerateRequest, GenerateResponse,
    HeartbeatRequest, HeartbeatResponse, LoadShardRequest, RegisterWorkerRequest,
    RegisterWorkerResponse, SampleTokenRequest, TopologyUpdate,
};
use bitty_protocol::security::{
    constant_time_eq, validate_cluster_token, AuthMode, BITTY_TOKEN_HEADER,
};
use bitty_protocol::{HardwareProfile, LayerMetadata, NodeId, BITTY_PROTOCOL_VERSION};
use bitty_worker::{
    network::{NetworkWorker, RuntimeStats},
    HardwareProfiler,
};
use iroh::{endpoint::presets, Endpoint, EndpointAddr, EndpointId, SecretKey};
use model_store::{
    copy_model, installed_models, pull_model, pull_stage, remove_model, resolve_model,
};
use settings::BittySettings;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, timeout, Duration};

const DEFAULT_TCP_CLUSTER: &str = "127.0.0.1:50051";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    load_env_file(&bitty_data_dir(None).join("env"));
    logger::log_default(format!(
        "bitty command: {}",
        secrets::redact_secret_args(&raw_args).join(" ")
    ));
    let result = match Cli::parse(raw_args) {
        Ok(CliCommand::Node(config)) => run_node(config).await,
        Ok(CliCommand::Run(config)) => run_model(config).await,
        Ok(CliCommand::Pull(config)) => run_pull(config).await,
        Ok(CliCommand::List(config)) => run_list(config).await,
        Ok(CliCommand::Show(config)) => run_show(config).await,
        Ok(CliCommand::Ps(config)) => run_ps(config).await,
        Ok(CliCommand::Stop(config)) => run_stop(config).await,
        Ok(CliCommand::Start(config)) => run_start(config).await,
        Ok(CliCommand::Restart(config)) => run_restart(config).await,
        Ok(CliCommand::Serve(config)) => run_serve(config).await,
        Ok(CliCommand::Setup(config)) => run_setup(config).await,
        Ok(CliCommand::Create(config)) => run_create(config).await,
        Ok(CliCommand::Rm(config)) => run_rm(config).await,
        Ok(CliCommand::Cp(config)) => run_cp(config).await,
        Ok(CliCommand::Settings(config)) => run_settings(config).await,
        Ok(CliCommand::Logs(config)) => run_logs(config).await,
        Ok(CliCommand::Cluster(config)) => run_cluster(config).await,
        Ok(CliCommand::Invite(config)) => run_invite(config).await,
        Ok(CliCommand::Share(config)) => run_share(config).await,
        Ok(CliCommand::Join(config)) => run_join(config).await,
        Ok(CliCommand::Connect(config)) => run_connect(config).await,
        Ok(CliCommand::Use(config)) => run_use(config).await,
        Ok(CliCommand::Clusters(config)) => run_clusters(config).await,
        Ok(CliCommand::Generate(config)) => run_generate(config).await,
        Ok(CliCommand::Chat(config)) => run_chat(config).await,
        Ok(CliCommand::Status(config)) => run_status(config).await,
        Ok(CliCommand::Clean(config)) => run_clean(config).await,
        Ok(CliCommand::Reset(config)) => run_reset(config).await,
        Ok(CliCommand::Help) => {
            print_help();
            Ok(())
        }
        Ok(CliCommand::Version) => {
            print_version();
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
    Stop(StopConfig),
    Start(StartConfig),
    Restart(StartConfig),
    Serve(ServeConfig),
    Setup(DataDirConfig),
    Create(CreateConfig),
    Rm(ModelCommand),
    Cp(CpConfig),
    Settings(SettingsCommand),
    Logs(LogsConfig),
    Cluster(ClusterCommand),
    Invite(InviteConfig),
    Share(InviteConfig),
    Join(JoinConfig),
    Connect(JoinConfig),
    Use(UseConfig),
    Clusters(DataDirConfig),
    Generate(GenerateConfig),
    Chat(ChatConfig),
    Status(StatusConfig),
    Clean(DataDirConfig),
    Reset(DataDirConfig),
    Help,
    Version,
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
    options: GenerateOptions,
}

/// Optional settings of a request; zero and empty values leave the coordinator's defaults.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GenerateOptions {
    top_k: u32,
    top_p: String,
    seed: u64,
    /// Send the prompt as it is, without the chat template.
    raw: bool,
    /// Let the model reason before it answers.
    thinking: bool,
    /// Token of the cluster, for a scheduler reached over plain TCP.
    cluster_token: Option<String>,
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
    /// `pull --layers A..B`: fetch only that layer range.
    layers: Option<(u32, u32)>,
    output: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DataDirConfig {
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StopConfig {
    model: Option<String>,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StartConfig {
    model: Option<String>,
    join: Option<String>,
    data_dir: Option<String>,
    /// Do nothing unless a node was started before and never stopped by hand.
    if_started: bool,
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
    Benchmark(ClusterConfig),
    Invite(DataDirConfig),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClusterConfig {
    node: Option<String>,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InviteConfig {
    name: Option<String>,
    replace: bool,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct JoinConfig {
    target: String,
    name: Option<String>,
    replace: bool,
    model: Option<String>,
    node_id: Option<String>,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct UseConfig {
    target: String,
    name: Option<String>,
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
            "stop" => parse_stop(&mut args).map(CliCommand::Stop),
            "start" => parse_start(&mut args).map(CliCommand::Start),
            "restart" => parse_start(&mut args).map(CliCommand::Restart),
            "serve" => parse_serve(&mut args).map(CliCommand::Serve),
            "setup" => parse_data_dir(&mut args).map(CliCommand::Setup),
            "create" => parse_create(&mut args).map(CliCommand::Create),
            "rm" => parse_model_command(&mut args, "rm").map(CliCommand::Rm),
            "cp" => parse_cp(&mut args).map(CliCommand::Cp),
            "settings" => parse_settings(&mut args).map(CliCommand::Settings),
            "logs" => parse_logs(&mut args).map(CliCommand::Logs),
            "cluster" => parse_cluster(&mut args).map(CliCommand::Cluster),
            "invite" => parse_invite(&mut args).map(CliCommand::Invite),
            "share" => parse_share(&mut args).map(CliCommand::Share),
            "join" => parse_join(&mut args).map(CliCommand::Join),
            "connect" => parse_join(&mut args).map(CliCommand::Connect),
            "use" => parse_use(&mut args).map(CliCommand::Use),
            "clusters" => parse_data_dir(&mut args).map(CliCommand::Clusters),
            "generate" => parse_generate(&mut args).map(CliCommand::Generate),
            "chat" => parse_chat(&mut args).map(CliCommand::Chat),
            "status" => parse_status(&mut args).map(CliCommand::Status),
            "clean" => parse_data_dir(&mut args).map(CliCommand::Clean),
            "reset" => parse_data_dir(&mut args).map(CliCommand::Reset),
            "-h" | "--help" | "help" => Ok(CliCommand::Help),
            "-V" | "--version" | "version" => Ok(CliCommand::Version),
            other => Err(format!("unknown command: {other}")),
        }
    }
}

async fn run_node(mut config: NodeConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.node_id.is_empty() {
        config.node_id = stored_node_id(&bitty_data_dir(config.data_dir.as_deref()));
    }
    logger::log_default(format!(
        "starting node id={} model={} join={}",
        config.node_id,
        config.model,
        secrets::redact_iroh_token(config.join.as_deref().unwrap_or("leader"))
    ));
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
    validate_cluster_token(&cluster_token)?;
    // A node that holds the whole model opens it directly. Any other node keeps only the
    // vocabulary file now and fetches the stage file of its layers once they are assigned.
    let node_settings = BittySettings::load(data_dir.clone());
    let (model_file, stage_source) =
        match model_store::whole_model_path(&node_settings, &config.model) {
            Some(path) => (path, None),
            None => {
                let settings = node_settings.clone();
                let model = config.model.clone();
                let vocabulary = tokio::task::spawn_blocking(move || {
                    model_store::ensure_stage(&settings, &model, 0, 0, None)
                })
                .await??;
                let settings = node_settings.clone();
                let model = config.model.clone();
                let source: bitty_inference::StageSource = Arc::new(move |start, end, stale| {
                    let path =
                        model_store::ensure_stage(&settings, &model, start, end, Some(stale))?;
                    refuse_stage_past_memory(&path)?;
                    Ok(path)
                });
                (vocabulary, Some(source))
            }
        };
    let mut executor = StageExecutor::load(&model_file).await?;
    if let Some(source) = stage_source {
        executor = executor.with_stage_source(source);
    }
    let executor = Arc::new(executor);
    let mut worker = NetworkWorker::new(NodeId::new(config.node_id.clone()), executor)
        .with_auth_mode(AuthMode::PreSharedToken(cluster_token.clone()));
    if let Some(iroh_node) = &iroh_node {
        worker = worker.with_iroh_endpoint(iroh_node.endpoint.clone());
    }
    let worker_stats = worker.runtime_stats();
    if let Some(join) = &config.join {
        remember_active_cluster(&data_dir, join)?;
    }

    let local_worker_endpoint = if use_iroh && config.join.is_none() {
        Some(
            config
                .worker_listen
                .clone()
                .unwrap_or_else(|| "127.0.0.1:50061".into()),
        )
    } else if !use_iroh {
        Some(config.worker_listen.clone().unwrap_or_else(|| {
            if config.join.is_some() {
                config.listen.clone()
            } else {
                "0.0.0.0:50061".into()
            }
        }))
    } else {
        None
    };

    if let Some(worker_listen) = &local_worker_endpoint {
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
        let mut coordinator = NetworkCoordinator::new(leader_layers(&model_file, config.layers))
            .with_model_path(&model_file)
            .with_auth_mode(AuthMode::PreSharedToken(cluster_token.clone()));
        if let Some(iroh_node) = &iroh_node {
            coordinator =
                coordinator.with_iroh_endpoint(iroh_node.endpoint.clone(), cluster_token.clone());
            coordinator.spawn_background_loops();
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
        local_worker_endpoint.as_deref(),
        &cluster_token,
        worker_stats,
    )
    .await
}

async fn run_model(config: RunConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = load_settings(config.data_dir.as_deref());
    logger::log(
        &settings.data_dir,
        format!("running model {}", config.model),
    )?;
    let mut model = resolve_model(&settings, &config.model);
    if model.is_none() && config.auto_pull && settings.auto_pull {
        model = Some(pull_model(&settings, &config.model)?);
    }
    let model = model.ok_or_else(|| format!("model not found: {}", config.model))?;
    if !config.local && config.node.is_none() && settings.auto_start_node {
        // the node decides what it needs on disk: the whole file when it is already here,
        // otherwise only the stage of its own layers. Nothing is pulled for the client.
        ensure_background_runtime(&settings, Some(model.id()), None).await?;
        settings = BittySettings::load(settings.data_dir.clone());
    }
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
                    options: GenerateOptions::default(),
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
            options: GenerateOptions::default(),
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
    let executor = StageExecutor::load(&path).await?;
    if !executor.engine_available() {
        return Err(format!(
            "local inference is not available yet: this build has no llama.cpp stage engine. \
             {} was read correctly ({} blocks, architecture {}).",
            path.display(),
            executor.metadata().layer_count,
            executor.metadata().architecture
        )
        .into());
    }
    // Reached only once the stage engine is linked in.
    let _ = (prompt, max_tokens, temperature);
    record_running_model(settings, &model.id())?;
    Ok(())
}

async fn run_pull(config: ModelCommand) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    logger::log(
        &settings.data_dir,
        format!("pulling model {}", config.model),
    )?;
    if let Some((start, end)) = config.layers {
        let output = config.output.as_deref().map(std::path::Path::new);
        let (path, report) = pull_stage(&settings, &config.model, start, end, output, None)?;
        const MIB: f64 = 1024.0 * 1024.0;
        println!(
            "pulled layers {start}..{end} of {} to {}",
            config.model,
            path.display()
        );
        println!(
            "{} of {} tensors, {:.0} MiB of {:.0} MiB ({:.0} MiB read from the source)",
            report.tensors,
            report.source_tensors,
            report.bytes as f64 / MIB,
            report.source_bytes as f64 / MIB,
            report.fetched_bytes as f64 / MIB
        );
        return Ok(());
    }
    if config.output.is_some() {
        return Err("--output needs --layers".into());
    }
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
    print_runtime_summary(&settings);
    let path = running_models_path(&settings);
    let contents = std::fs::read_to_string(path).unwrap_or_default();
    println!("NAME\tSTATUS");
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        println!("{line}\tloaded");
    }
    Ok(())
}

async fn run_stop(config: StopConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let Some(model) = config.model else {
        stop_background_runtime(&settings)?;
        return Ok(());
    };
    let path = running_models_path(&settings);
    let contents = std::fs::read_to_string(&path).unwrap_or_default();
    let next = contents
        .lines()
        .filter(|line| *line != model)
        .collect::<Vec<_>>()
        .join("\n");
    settings::ensure_parent(&path)?;
    std::fs::write(path, next)?;
    println!("stopped {model}");
    Ok(())
}

async fn run_start(config: StartConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    if config.if_started && read_runtime_state(&settings.data_dir).is_none() {
        return Ok(());
    }
    // after `bitty stop` a plain start goes back to what the node was doing: a worker joins
    // its cluster again and does not come back as the leader of a cluster of its own
    let stopped = match read_runtime_state(&settings.data_dir) {
        Some(_) => None,
        None => read_state_file(&stopped_state_path(&settings.data_dir)),
    };
    let join = config.join.or_else(|| {
        stopped
            .as_ref()
            .filter(|state| state.mode == "worker" && !state.target.is_empty())
            .map(|state| state.target.clone())
    });
    let model = config.model.or_else(|| {
        stopped
            .map(|state| state.model)
            .filter(|model| !model.is_empty())
    });
    ensure_background_runtime(&settings, model, join).await?;
    print_runtime_summary(&settings);
    Ok(())
}

async fn run_restart(config: StartConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    // stopping forgets the saved state, so keep what the node was doing: without this a worker
    // would come back as the leader of a cluster of its own
    let saved = read_runtime_state(&settings.data_dir);
    if config.if_started && saved.is_none() {
        return Ok(());
    }
    let _ = stop_background_runtime(&settings);
    let join = config.join.or_else(|| {
        saved
            .as_ref()
            .filter(|state| state.mode == "worker" && !state.target.is_empty())
            .map(|state| state.target.clone())
    });
    let model = config.model.or_else(|| {
        saved
            .map(|state| state.model)
            .filter(|model| !model.is_empty())
    });
    ensure_background_runtime(&settings, model, join).await?;
    print_runtime_summary(&settings);
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

async fn run_setup(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    println!("Bitty {}", bitty_version());
    println!("data dir: {}", settings.data_dir.display());
    let model = resolve_model(&settings, &settings.default_model)
        .unwrap_or_else(|| model_store::find_registry_model(&settings.default_model).unwrap());
    let path = model.model_path(&settings);
    if path.exists() {
        println!("model ready: {} ({})", model.id(), path.display());
    } else if settings.auto_pull {
        println!("pulling default model: {}", settings.default_model);
        let model = pull_model(&settings, &settings.default_model)?;
        println!(
            "model ready: {} ({})",
            model.id(),
            model.model_path(&settings).display()
        );
    } else {
        println!(
            "model missing: {}. Run `bitty pull {}`.",
            path.display(),
            settings.default_model
        );
    }
    println!("next: `bitty share home` or `bitty connect INVITE --name home`");
    Ok(())
}

async fn run_clean(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    assert_destructive_data_dir(&settings.data_dir)?;
    if !confirm_yes(
        "This removes local models, saved cluster aliases, cluster token, Iroh identity key, logs, and runtime state.\n\
         Your config.toml is kept (API host and defaults), but the active cluster is cleared.",
    )? {
        println!("Aborted.");
        return Ok(());
    }
    let mut settings = settings;
    stop_background_runtime(&settings)?;
    clean_local_state(&settings)?;
    settings.active_cluster.clear();
    settings.save()?;
    println!(
        "Cleaned Bitty data under {}. Run `bitty setup` or `bitty pull` to restore models.",
        settings.data_dir.display()
    );
    Ok(())
}

async fn run_reset(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    assert_destructive_data_dir(&settings.data_dir)?;
    if !confirm_yes(
        "This permanently deletes the entire Bitty data directory (models, config, clusters, tokens, keys, logs).\n\
         Afterward you get a fresh default config, as after a first install.",
    )? {
        println!("Aborted.");
        return Ok(());
    }
    let data_dir = settings.data_dir.clone();
    stop_background_runtime(&settings)?;
    std::fs::remove_dir_all(&data_dir)?;
    BittySettings::defaults(data_dir.clone()).save()?;
    println!(
        "Reset complete. Data directory: {}. Run `bitty setup` when you are ready.",
        data_dir.display()
    );
    Ok(())
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
        ClusterCommand::Benchmark(config) => {
            let status =
                fetch_cluster_status(config.node.as_deref(), config.data_dir.as_deref()).await?;
            print_cluster_benchmark(&status);
        }
        ClusterCommand::Invite(config) => {
            run_invite(InviteConfig {
                name: None,
                replace: false,
                data_dir: config.data_dir,
            })
            .await?;
        }
    }
    Ok(())
}

async fn run_invite(config: InviteConfig) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    let settings = BittySettings::load(data_dir.clone());
    ensure_background_runtime(&settings, None, None).await?;
    let invite = wait_for_active_cluster(&data_dir, Duration::from_secs(5))
        .await?
        .ok_or(
            "Bitty started, but no invite was published yet. Try `bitty invite` again in a moment.",
        )?;
    let invite = relay_only_invite(&invite).unwrap_or(invite);
    let name = remember_cluster_alias(&data_dir, config.name.as_deref(), &invite, config.replace)?;
    remember_active_cluster(&data_dir, &invite)?;
    println!("Invite name: {name}");
    println!("Invite:");
    println!("  {invite}");
    println!();
    println!("On another machine:");
    println!("  bitty connect '{}' --name {name}", invite);
    println!("Bitty is running in the background. Use `bitty stop` to stop it.");
    Ok(())
}

async fn run_share(config: InviteConfig) -> Result<(), Box<dyn std::error::Error>> {
    run_invite(config).await
}

async fn run_join(config: JoinConfig) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    let invite = resolve_cluster_alias_or_invite(&data_dir, &config.target)?;
    let name = remember_cluster_alias(&data_dir, config.name.as_deref(), &invite, config.replace)?;
    remember_active_cluster(&data_dir, &invite)?;
    println!("using cluster `{name}`");
    run_node(NodeConfig {
        model: config
            .model
            .unwrap_or_else(|| default_model_path(&data_dir)),
        node_id: config.node_id.unwrap_or_default(),
        listen: "0.0.0.0:50051".into(),
        worker_listen: None,
        public_endpoint: None,
        join: Some(invite),
        layers: 40,
        heartbeat_interval_ms: 1000,
        iroh: true,
        data_dir: config.data_dir,
        cluster_token: None,
    })
    .await
}

async fn run_connect(config: JoinConfig) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    let invite = resolve_cluster_alias_or_invite(&data_dir, &config.target)?;
    let settings = BittySettings::load(data_dir.clone());
    let name = remember_cluster_alias(&data_dir, config.name.as_deref(), &invite, config.replace)?;
    remember_active_cluster(&data_dir, &invite)?;
    ensure_background_runtime(&settings, config.model, Some(invite)).await?;
    println!("connected to cluster `{name}`");
    println!("You can now run: bitty run qwen3.5-35b-a3b \"hello\"");
    Ok(())
}

async fn run_use(config: UseConfig) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    let invite = resolve_cluster_alias_or_invite(&data_dir, &config.target)?;
    if let Some(name) = config.name.as_deref() {
        remember_cluster_alias(&data_dir, Some(name), &invite, false)?;
    }
    remember_active_cluster(&data_dir, &invite)?;
    println!(
        "active cluster: {}",
        config.name.as_deref().unwrap_or(&config.target)
    );
    Ok(())
}

async fn run_clusters(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    let settings = BittySettings::load(data_dir.clone());
    let active = active_cluster_target(&settings);
    let store = cluster_store::ClusterStore::load(&data_dir);
    println!("NAME\tACTIVE\tTARGET");
    for (name, invite) in store.aliases() {
        let marker = if Some(invite) == active.as_deref() {
            "yes"
        } else {
            "no"
        };
        println!("{name}\t{marker}\t{}", compact_invite(invite));
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
    local_worker_endpoint: Option<&str>,
    cluster_token: &str,
    worker_stats: RuntimeStats,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut profile = HardwareProfiler::new(config.node_id.clone()).profile();
    profile.network_rtt_ms = estimate_scheduler_rtt_ms(target).await;
    profile.model_path = config.model.clone();
    profile.build = option_env!("BITTY_GIT_SHA").unwrap_or("local").to_string();
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    profile.free_disk_mb =
        model_store::disk_for_stages_mb(&BittySettings::load(data_dir), &config.model).unwrap_or(0);
    profile.worker_endpoint = match target {
        SchedulerTarget::Local(_) => local_worker_endpoint
            .map(public_endpoint_from_listen)
            .unwrap_or_else(|| "127.0.0.1:50061".into()),
        SchedulerTarget::Iroh { .. } => iroh_node
            .map(|iroh_node| {
                iroh_transport::iroh_uri_for_addr(&iroh_node.endpoint.addr(), cluster_token)
            })
            .unwrap_or_default(),
        SchedulerTarget::Tcp(_) => {
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
        }
    };

    match target {
        SchedulerTarget::Local(coordinator) => {
            let registration = coordinator
                .register_worker(request_with_token(
                    RegisterWorkerRequest {
                        profile: Some((&profile).into()),
                        protocol_version: BITTY_PROTOCOL_VERSION,
                        inference_backend_id: STAGE_BACKEND_ID.into(),
                    },
                    cluster_token,
                ))
                .await?
                .into_inner();
            println!(
                "bitty node started local scheduler; topology_epoch={} assignments={}",
                registration.topology_epoch,
                registration.assignments.len()
            );
            loop {
                sleep(Duration::from_millis(config.heartbeat_interval_ms)).await;
                let snapshot = worker_stats.snapshot().await;
                let response = coordinator
                    .heartbeat(request_with_token(
                        HeartbeatRequest {
                            node_id: profile.node_id.0.clone(),
                            observed_tokens_per_second: observed_tokens_per_second(
                                &profile,
                                snapshot.observed_tokens_per_second,
                            ),
                            avg_forward_latency_ms: snapshot.avg_forward_latency_ms,
                            activation_bytes_per_second: snapshot.activation_bytes_per_second,
                            backend_type: profile.backend_type.clone(),
                        },
                        cluster_token,
                    ))
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
            let scheduler = RemoteScheduler::Tcp {
                endpoint: normalize_endpoint(leader),
                client: None,
                token: cluster_token.to_string(),
            };
            stay_registered(scheduler, config, &profile, worker_stats).await
        }
        SchedulerTarget::Iroh {
            endpoint,
            remote,
            token,
        } => {
            let scheduler = RemoteScheduler::Iroh {
                client: IrohSchedulerClient {
                    endpoint: endpoint.clone(),
                    remote: remote.clone(),
                    token: token.clone(),
                },
                invite: remote.clone(),
            };
            stay_registered(scheduler, config, &profile, worker_stats).await
        }
    }
}

/// A scheduler on another machine, over gRPC or over iroh.
enum RemoteScheduler {
    Tcp {
        endpoint: String,
        client: Option<CoordinatorServiceClient<tonic::transport::Channel>>,
        token: String,
    },
    Iroh {
        client: IrohSchedulerClient,
        /// The addresses written in the invite. They go stale when the leader restarts on
        /// another port or relay, so after a failure the leader is looked up by its id.
        invite: EndpointAddr,
    },
}

/// A leader that is gone must not hold a worker forever in one call.
const IROH_SCHEDULER_TIMEOUT: Duration = Duration::from_secs(20);

impl RemoteScheduler {
    fn label(&self) -> String {
        match self {
            Self::Tcp { endpoint, .. } => endpoint.clone(),
            Self::Iroh { client, .. } => format!("iroh {}", client.remote.id),
        }
    }

    /// The gRPC channel is opened on first use and dropped after an error, so the next call
    /// connects again: that is what brings a worker back once the leader has restarted.
    async fn tcp_client<'a>(
        endpoint: &str,
        client: &'a mut Option<CoordinatorServiceClient<tonic::transport::Channel>>,
    ) -> Result<&'a mut CoordinatorServiceClient<tonic::transport::Channel>, String> {
        if client.is_none() {
            let connected = CoordinatorServiceClient::connect(endpoint.to_string())
                .await
                .map_err(|err| err.to_string())?;
            *client = Some(connected);
        }
        Ok(client.as_mut().expect("just connected"))
    }

    /// One call to the leader over iroh. After a failure the next call reaches the leader
    /// another way: by id alone (its current addresses come from the lookup service), then
    /// with the invite's addresses again.
    async fn iroh_request<M, R>(
        client: &mut IrohSchedulerClient,
        invite: &EndpointAddr,
        op: u8,
        message: &M,
    ) -> Result<R, String>
    where
        M: prost::Message,
        R: prost::Message + Default,
    {
        let result = match timeout(IROH_SCHEDULER_TIMEOUT, client.request(op, message)).await {
            Ok(result) => result.map_err(|err| err.to_string()),
            Err(_) => Err(format!(
                "no answer in {}s",
                IROH_SCHEDULER_TIMEOUT.as_secs()
            )),
        };
        if result.is_err() {
            client.remote = if client.remote.is_empty() {
                invite.clone()
            } else {
                EndpointAddr::from_parts(invite.id, Vec::<iroh::TransportAddr>::new())
            };
        }
        result
    }

    async fn register(
        &mut self,
        request: RegisterWorkerRequest,
    ) -> Result<RegisterWorkerResponse, String> {
        match self {
            Self::Tcp {
                endpoint,
                client,
                token,
            } => {
                let result = Self::tcp_client(endpoint, client)
                    .await?
                    .register_worker(request_with_token(request, token))
                    .await;
                if result.is_err() {
                    *client = None;
                }
                result
                    .map(tonic::Response::into_inner)
                    .map_err(|status| bitty_protocol::worker_client::status_text(&status))
            }
            Self::Iroh { client, invite } => {
                Self::iroh_request(client, invite, SCHEDULER_REGISTER_WORKER, &request).await
            }
        }
    }

    async fn heartbeat(&mut self, request: HeartbeatRequest) -> Result<HeartbeatResponse, String> {
        match self {
            Self::Tcp {
                endpoint,
                client,
                token,
            } => {
                let result = Self::tcp_client(endpoint, client)
                    .await?
                    .heartbeat(request_with_token(request, token))
                    .await;
                if result.is_err() {
                    *client = None;
                }
                result
                    .map(tonic::Response::into_inner)
                    .map_err(|status| bitty_protocol::worker_client::status_text(&status))
            }
            Self::Iroh { client, invite } => {
                Self::iroh_request(client, invite, SCHEDULER_HEARTBEAT, &request).await
            }
        }
    }
}

/// Heartbeats that may fail in a row before the worker registers again.
const HEARTBEAT_FAILURES_BEFORE_REJOIN: u32 = 5;

/// Registers with a remote scheduler and keeps the registration alive for as long as the node
/// runs. A scheduler that restarts (an update, a reboot) has forgotten its workers: it answers
/// the next heartbeat with "not accepted", or does not answer at all for a while. In both cases
/// the worker registers again by itself, so nobody has to restart the nodes by hand.
async fn stay_registered(
    mut scheduler: RemoteScheduler,
    config: &NodeConfig,
    profile: &HardwareProfile,
    worker_stats: RuntimeStats,
) -> Result<(), Box<dyn std::error::Error>> {
    let label = scheduler.label();
    let mut retry = Duration::from_secs(1);
    loop {
        let registration = scheduler
            .register(RegisterWorkerRequest {
                profile: Some(profile.into()),
                protocol_version: BITTY_PROTOCOL_VERSION,
                inference_backend_id: STAGE_BACKEND_ID.into(),
            })
            .await;
        match registration {
            Ok(registration) => {
                println!(
                    "bitty node joined scheduler {label}; topology_epoch={} assignments={}",
                    registration.topology_epoch,
                    registration.assignments.len()
                );
                retry = Duration::from_secs(1);
            }
            Err(err) => {
                eprintln!(
                    "cannot register with scheduler {label}: {err}; trying again in {}s",
                    retry.as_secs()
                );
                sleep(retry).await;
                retry = (retry * 2).min(Duration::from_secs(30));
                continue;
            }
        }

        let mut failures = 0;
        loop {
            sleep(Duration::from_millis(config.heartbeat_interval_ms)).await;
            let snapshot = worker_stats.snapshot().await;
            let response = scheduler
                .heartbeat(HeartbeatRequest {
                    node_id: profile.node_id.0.clone(),
                    observed_tokens_per_second: observed_tokens_per_second(
                        profile,
                        snapshot.observed_tokens_per_second,
                    ),
                    avg_forward_latency_ms: snapshot.avg_forward_latency_ms,
                    activation_bytes_per_second: snapshot.activation_bytes_per_second,
                    backend_type: profile.backend_type.clone(),
                })
                .await;
            match response {
                Ok(response) if response.accepted => failures = 0,
                Ok(_) => {
                    eprintln!(
                        "scheduler {label} no longer knows node {}; registering again",
                        profile.node_id.0
                    );
                    break;
                }
                Err(err) => {
                    failures += 1;
                    eprintln!("heartbeat failed: {err}");
                    if failures >= HEARTBEAT_FAILURES_BEFORE_REJOIN {
                        eprintln!("scheduler {label} does not answer; registering again");
                        break;
                    }
                }
            }
        }
    }
}

async fn estimate_scheduler_rtt_ms(target: &SchedulerTarget) -> f64 {
    match target {
        SchedulerTarget::Local(_) => 0.5,
        SchedulerTarget::Iroh { .. } => std::env::var("BITTY_NETWORK_RTT_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(25.0),
        SchedulerTarget::Tcp(endpoint) => {
            let endpoint = endpoint
                .trim_start_matches("http://")
                .trim_start_matches("https://");
            let started = std::time::Instant::now();
            let address = endpoint.to_string();
            let result = tokio::task::spawn_blocking(move || {
                let Ok(socket) = address.parse::<std::net::SocketAddr>() else {
                    return false;
                };
                std::net::TcpStream::connect_timeout(&socket, std::time::Duration::from_millis(750))
                    .is_ok()
            })
            .await
            .unwrap_or(false);
            if result {
                started.elapsed().as_secs_f64() * 1000.0
            } else {
                std::env::var("BITTY_NETWORK_RTT_MS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(50.0)
            }
        }
    }
}

fn observed_tokens_per_second(profile: &HardwareProfile, measured: f64) -> f64 {
    if measured > 0.0 {
        measured
    } else {
        profile.effective_compute_score()
    }
}

fn request_with_token<T>(message: T, token: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    if let Ok(value) = token.parse() {
        request.metadata_mut().insert(BITTY_TOKEN_HEADER, value);
    }
    request
}

async fn run_generate(config: GenerateConfig) -> Result<(), Box<dyn std::error::Error>> {
    let node = resolve_cluster_node(config.node.as_str(), config.data_dir.as_deref())?;
    if let Some(target) = iroh_transport::parse_iroh_target(&node) {
        let endpoint = start_iroh_client(target.endpoint_addr.is_empty()).await?;
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
                    top_k: config.options.top_k,
                    top_p: config.options.top_p.parse().unwrap_or(0.0),
                    seed: config.options.seed,
                    raw: config.options.raw,
                    thinking: config.options.thinking,
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
        .generate(request_with_token(
            GenerateRequest {
                request_id: request_id(),
                prompt_tokens: Vec::new(),
                prompt: config.prompt,
                max_new_tokens: config.max_tokens,
                temperature: config.temperature.parse().unwrap_or(0.0),
                top_k: config.options.top_k,
                top_p: config.options.top_p.parse().unwrap_or(0.0),
                seed: config.options.seed,
                raw: config.options.raw,
                thinking: config.options.thinking,
            },
            config.options.cluster_token.as_deref().unwrap_or_default(),
        ))
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
            options: GenerateOptions::default(),
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
            options: GenerateOptions::default(),
            data_dir: config.data_dir.clone(),
        })
        .await?;
    }
    Ok(())
}

async fn run_status(config: StatusConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = load_settings(config.data_dir.as_deref());
    if config.node.is_none() && settings.auto_start_node {
        let _ = ensure_background_runtime(&settings, None, None).await;
        settings = BittySettings::load(settings.data_dir.clone());
    }
    print_runtime_summary(&settings);
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
        let endpoint = start_iroh_client(target.endpoint_addr.is_empty()).await?;
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
    if !status.profiles.is_empty() {
        println!();
        print_cluster_benchmark(status);
    }
}

fn print_cluster_benchmark(status: &ClusterStatusResponse) {
    println!("Hardware profile");
    println!("----------------");
    println!("NODE\tBACKEND\tELIGIBLE\tRAM\tVRAM\tCPU TFLOPS\tGPU TFLOPS\tRTT\tUPLINK");
    for profile in &status.profiles {
        println!(
            "{}\t{}\t{}\t{} MB\t{} MB\t{:.2}\t{:.2}\t{:.1} ms\t{:.0} Mbps",
            profile.node_id,
            if profile.backend_type.is_empty() {
                if profile.gpu_tflops > 0.0 {
                    "gpu"
                } else {
                    "cpu"
                }
            } else {
                profile.backend_type.as_str()
            },
            yes_no(profile.layer_eligible),
            profile.ram_mb,
            profile.vram_mb,
            profile.cpu_tflops,
            profile.gpu_tflops,
            profile.network_rtt_ms,
            profile.uplink_mbps
        );
    }
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
        node_id: String::new(),
        listen: "0.0.0.0:50051".into(),
        worker_listen: None,
        public_endpoint: None,
        join: None,
        layers: 40,
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
        options: GenerateOptions::default(),
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = required_next(args, "--node")?,
            "--prompt" => config.prompt = required_next(args, "--prompt")?,
            "--max-tokens" => config.max_tokens = parse_next(args, "--max-tokens")?,
            "--temperature" => config.temperature = required_next(args, "--temperature")?,
            "--top-k" => config.options.top_k = parse_next(args, "--top-k")?,
            "--top-p" => config.options.top_p = required_next(args, "--top-p")?,
            "--seed" => config.options.seed = parse_next(args, "--seed")?,
            "--raw" => config.options.raw = true,
            "--think" => config.options.thinking = true,
            "--cluster-token" => {
                config.options.cluster_token = Some(required_next(args, "--cluster-token")?)
            }
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

/// `A..B`: layers from A up to, not including, B.
fn parse_layer_range(value: &str) -> Result<(u32, u32), String> {
    let invalid = || format!("--layers expects START..END (end not included), got {value}");
    let (start, end) = value.split_once("..").ok_or_else(invalid)?;
    let start: u32 = start.trim().parse().map_err(|_| invalid())?;
    let end: u32 = end.trim().parse().map_err(|_| invalid())?;
    if start >= end {
        return Err(invalid());
    }
    Ok((start, end))
}

fn parse_model_command(
    args: &mut impl Iterator<Item = String>,
    command: &str,
) -> Result<ModelCommand, String> {
    let mut config = ModelCommand {
        model: String::new(),
        data_dir: None,
        layers: None,
        output: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            "--layers" if command == "pull" => {
                config.layers = Some(parse_layer_range(&required_next(args, "--layers")?)?)
            }
            "--output" if command == "pull" => {
                config.output = Some(required_next(args, "--output")?)
            }
            value if config.model.is_empty() => config.model = value.into(),
            other => return Err(format!("unknown {command} argument: {other}")),
        }
    }
    if config.model.is_empty() {
        return Err(format!("bitty {command} requires MODEL"));
    }
    Ok(config)
}

fn parse_stop(args: &mut impl Iterator<Item = String>) -> Result<StopConfig, String> {
    let mut config = StopConfig {
        model: None,
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.model.is_none() => config.model = Some(value.into()),
            other => return Err(format!("unknown stop argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_start(args: &mut impl Iterator<Item = String>) -> Result<StartConfig, String> {
    let mut config = StartConfig {
        model: None,
        join: None,
        data_dir: None,
        if_started: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--if-started" => config.if_started = true,
            "--model" => config.model = Some(required_next(args, "--model")?),
            "--join" => config.join = Some(required_next(args, "--join")?),
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            other => return Err(format!("unknown start argument: {other}")),
        }
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
        "benchmark" | "bench" => parse_cluster_config(args).map(ClusterCommand::Benchmark),
        "invite" => parse_data_dir(args).map(ClusterCommand::Invite),
        other => Err(format!("unknown cluster command: {other}")),
    }
}

fn parse_invite(args: &mut impl Iterator<Item = String>) -> Result<InviteConfig, String> {
    let mut config = InviteConfig {
        name: None,
        replace: false,
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--name" | "-n" => config.name = Some(required_next(args, "--name")?),
            "--replace" => config.replace = true,
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            other => return Err(format!("unknown invite argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_share(args: &mut impl Iterator<Item = String>) -> Result<InviteConfig, String> {
    let mut config = InviteConfig {
        name: None,
        replace: false,
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--name" | "-n" => config.name = Some(required_next(args, "--name")?),
            "--replace" => config.replace = true,
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.name.is_none() => config.name = Some(value.into()),
            other => return Err(format!("unknown share argument: {other}")),
        }
    }
    Ok(config)
}

fn parse_join(args: &mut impl Iterator<Item = String>) -> Result<JoinConfig, String> {
    let mut config = JoinConfig {
        target: String::new(),
        name: None,
        replace: false,
        model: None,
        node_id: None,
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--name" | "-n" => config.name = Some(required_next(args, "--name")?),
            "--replace" => config.replace = true,
            "--model" => config.model = Some(required_next(args, "--model")?),
            "--node-id" => config.node_id = Some(required_next(args, "--node-id")?),
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.target.is_empty() => config.target = value.into(),
            other => return Err(format!("unknown join argument: {other}")),
        }
    }
    if config.target.is_empty() {
        return Err("bitty join requires INVITE_OR_NAME".into());
    }
    Ok(config)
}

fn parse_use(args: &mut impl Iterator<Item = String>) -> Result<UseConfig, String> {
    let mut config = UseConfig {
        target: String::new(),
        name: None,
        data_dir: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--name" | "-n" => config.name = Some(required_next(args, "--name")?),
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            value if config.target.is_empty() => config.target = value.into(),
            other => return Err(format!("unknown use argument: {other}")),
        }
    }
    if config.target.is_empty() {
        return Err("bitty use requires NAME_OR_INVITE".into());
    }
    Ok(config)
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
    bitty_protocol::cli::required_next(args, name)
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    bitty_protocol::cli::parse_next(args, name)
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
        worker: NetworkWorker<StageExecutor>,
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

/// A client that only dials out. `wait_until_online` holds it until its own relay connection is
/// up, which takes a few seconds: only needed when the target comes without any address and has
/// to be looked up.
async fn start_iroh_client(
    wait_until_online: bool,
) -> Result<Endpoint, Box<dyn std::error::Error>> {
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(SecretKey::generate())
        .alpns(vec![
            BITTY_SCHEDULER_ALPN.to_vec(),
            BITTY_WORKER_ALPN.to_vec(),
        ])
        .bind()
        .await?;
    if wait_until_online {
        let _ = timeout(Duration::from_secs(10), endpoint.online()).await;
    }
    Ok(endpoint)
}

async fn handle_iroh_request(
    incoming: iroh::endpoint::Incoming,
    coordinator: Option<NetworkCoordinator>,
    worker: NetworkWorker<StageExecutor>,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>> {
    // the address goes into the error: a peer that keeps failing has to be told from the others
    let from = incoming.remote_addr();
    let connection = incoming
        .accept()
        .map_err(|err| format!("from {from:?}: {err}"))?
        .await
        .map_err(|err| format!("handshake from {from:?}: {err}"))?;
    let alpn = connection.alpn().to_vec();
    let remote_id = connection.remote_id();
    let peer = remote_id.fmt_short().to_string();
    if alpn == BITTY_WORKER_ALPN {
        return handle_worker_connection(connection, worker, cluster_token)
            .await
            .map_err(|err| format!("worker call from {peer}: {err}").into());
    }
    let Some(coordinator) = coordinator else {
        return Err("this node is not a scheduler leader".into());
    };
    handle_scheduler_connection(connection, remote_id, coordinator, cluster_token)
        .await
        .map_err(|err| format!("leader call from {peer}: {err}").into())
}

async fn handle_scheduler_connection(
    connection: iroh::endpoint::Connection,
    remote_id: EndpointId,
    coordinator: NetworkCoordinator,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut send, mut recv) = connection.accept_bi().await?;
    let frame = iroh_transport::read_frame(&mut recv, DEFAULT_FRAME_LIMIT).await?;
    if !constant_time_eq(frame.token.as_bytes(), cluster_token.as_bytes()) {
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
                    profile.worker_endpoint = iroh_transport::iroh_uri(remote_id, cluster_token);
                }
            }
            let response = coordinator
                .register_worker(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(SCHEDULER_REGISTER_WORKER, "", &response.into_inner())
        }
        SCHEDULER_HEARTBEAT => {
            let request: HeartbeatRequest = frame.decode_message(SCHEDULER_HEARTBEAT)?;
            let response = coordinator
                .heartbeat(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(SCHEDULER_HEARTBEAT, "", &response.into_inner())
        }
        SCHEDULER_GENERATE => {
            let request: GenerateRequest = frame.decode_message(SCHEDULER_GENERATE)?;
            let response = coordinator
                .generate(request_with_token(request, cluster_token))
                .await?;
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
                .cluster_status(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(SCHEDULER_CLUSTER_STATUS, "", &response.into_inner())
        }
        _ => return Err(format!("unknown scheduler op {}", frame.op).into()),
    };
    Ok(response)
}

async fn handle_worker_connection(
    connection: iroh::endpoint::Connection,
    worker: NetworkWorker<StageExecutor>,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>> {
    // the coordinator keeps one connection and opens a stream for every request
    let mut served = 0u64;
    loop {
        let (mut send, mut recv) = match connection.accept_bi().await {
            Ok(streams) => streams,
            Err(_) if served > 0 => return Ok(()),
            Err(err) => return Err(err.into()),
        };
        served += 1;
        let frame = iroh_transport::read_frame(&mut recv, DEFAULT_FRAME_LIMIT).await?;
        if !constant_time_eq(frame.token.as_bytes(), cluster_token.as_bytes()) {
            return Err("invalid cluster token".into());
        }
        let response = match handle_worker_frame(frame, worker.clone(), &cluster_token).await {
            Ok(response) => response,
            Err(err) => iroh_transport::error_frame(err),
        };
        iroh_transport::write_frame(&mut send, &response).await?;
        send.finish()?;
        send.stopped().await?;
    }
}

async fn handle_worker_frame(
    frame: IrohFrame,
    worker: NetworkWorker<StageExecutor>,
    cluster_token: &str,
) -> Result<IrohFrame, Box<dyn std::error::Error>> {
    let response = match frame.op {
        WORKER_FORWARD_ACTIVATION => {
            let request: ProtoActivationTensor = frame.decode_message(WORKER_FORWARD_ACTIVATION)?;
            let response = worker
                .forward_activation(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_FORWARD_ACTIVATION, "", &response.into_inner())
        }
        WORKER_FINAL_LOGITS => {
            let request: ProtoActivationTensor = frame.decode_message(WORKER_FINAL_LOGITS)?;
            let response = worker
                .final_logits(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_FINAL_LOGITS, "", &response.into_inner())
        }
        WORKER_SAMPLE_TOKEN => {
            let request: SampleTokenRequest = frame.decode_message(WORKER_SAMPLE_TOKEN)?;
            let response = worker
                .sample_token(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_SAMPLE_TOKEN, "", &response.into_inner())
        }
        WORKER_APPLY_TOPOLOGY => {
            let request: TopologyUpdate = frame.decode_message(WORKER_APPLY_TOPOLOGY)?;
            let response = worker
                .apply_topology(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_APPLY_TOPOLOGY, "", &response.into_inner())
        }
        WORKER_LOAD_SHARD => {
            let request: LoadShardRequest = frame.decode_message(WORKER_LOAD_SHARD)?;
            let response = worker
                .load_shard(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_LOAD_SHARD, "", &response.into_inner())
        }
        WORKER_CLEANUP => {
            let request: CleanupRequest = frame.decode_message(WORKER_CLEANUP)?;
            let response = worker
                .cleanup(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_CLEANUP, "", &response.into_inner())
        }
        WORKER_FORWARD_CHAIN => {
            let request: ChainStep = frame.decode_message(WORKER_FORWARD_CHAIN)?;
            let response = worker
                .forward_chain(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_FORWARD_CHAIN, "", &response.into_inner())
        }
        WORKER_AWAIT_CHAIN_RESULT => {
            let request: ChainResultRequest = frame.decode_message(WORKER_AWAIT_CHAIN_RESULT)?;
            let response = worker
                .await_chain_result(request_with_token(request, cluster_token))
                .await?;
            IrohFrame::message(WORKER_AWAIT_CHAIN_RESULT, "", &response.into_inner())
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
        let data_dir = bitty_data_dir(data_dir);
        return resolve_cluster_alias_or_invite(&data_dir, explicit.trim());
    }
    let settings = load_settings(data_dir);
    active_cluster_target(&settings).ok_or_else(|| {
        format!(
            "no active Bitty cluster saved. Start one with `bitty node --model PATH`, join one with `bitty node --join INVITE --model PATH`, or pass `--node {DEFAULT_TCP_CLUSTER}`."
        )
        .into()
    })
}

fn resolve_cluster_alias_or_invite(
    data_dir: &Path,
    value: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if cluster_store::looks_like_invite(value) {
        return Ok(value.to_string());
    }
    let store = cluster_store::ClusterStore::load(data_dir);
    if let Some(invite) = store.get(value) {
        return Ok(invite.to_string());
    }
    if value.contains("://") || value.contains(':') {
        return Ok(value.to_string());
    }
    Err(format!("unknown cluster `{value}`. Use `bitty clusters` to list saved names.").into())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RuntimeState {
    pid: u32,
    mode: String,
    model: String,
    target: String,
}

async fn ensure_background_runtime(
    settings: &BittySettings,
    model_override: Option<String>,
    join_override: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(state) = read_runtime_state(&settings.data_dir) {
        if is_pid_running(state.pid) {
            return Ok(());
        }
    }

    let saved = read_runtime_state(&settings.data_dir);
    let join = join_override.or_else(|| {
        saved.as_ref().and_then(|state| {
            if state.mode == "worker" && !state.target.is_empty() {
                Some(state.target.clone())
            } else {
                None
            }
        })
    });
    let model = model_override
        .or_else(|| {
            saved
                .as_ref()
                .map(|state| state.model.clone())
                .filter(|model| !model.is_empty())
        })
        .unwrap_or_else(|| settings.default_model.clone());
    let model_path = match model_store::whole_model_path(settings, &model) {
        Some(path) => path.display().to_string(),
        // without the whole file the node fetches the vocabulary and the stage of its own layers
        None if model_store::has_stage_source(settings, &model) => model,
        None => model_path_for_runtime(settings, &model)?,
    };
    spawn_background_node(settings, &model_path, join).await
}

async fn spawn_background_node(
    settings: &BittySettings,
    model_path: &str,
    join: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    settings::ensure_parent(&runtime_state_path(&settings.data_dir))?;
    settings::ensure_parent(&logger::log_path(&settings.data_dir))?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(logger::log_path(&settings.data_dir))?;
    let err_log = log.try_clone()?;
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("node")
        .arg("--model")
        .arg(model_path)
        .arg("--data-dir")
        .arg(settings.data_dir.display().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log));
    if let Some(join) = &join {
        command.arg("--join").arg(join);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP: the node outlives the terminal that
        // started it, and neither it nor its stage engines open a console window
        command.creation_flags(0x0800_0000 | 0x0000_0200);
    }
    let mode = if join.is_some() { "worker" } else { "leader" };
    #[cfg(windows)]
    let child = {
        use std::os::windows::process::CommandExt;
        // CREATE_BREAKAWAY_FROM_JOB: a node started by the scheduled updater must not die with
        // the task that started it. A job that forbids leaving refuses the flag, so try without.
        command.creation_flags(0x0800_0000 | 0x0000_0200 | 0x0100_0000);
        match command.spawn() {
            Ok(child) => child,
            Err(_) => {
                command.creation_flags(0x0800_0000 | 0x0000_0200);
                command.spawn()?
            }
        }
    };
    #[cfg(not(windows))]
    let child = command.spawn()?;
    let state = RuntimeState {
        pid: child.id(),
        mode: mode.into(),
        model: model_path.into(),
        target: join.unwrap_or_default(),
    };
    write_runtime_state(&settings.data_dir, &state)?;
    logger::log(
        &settings.data_dir,
        format!("started background runtime pid={}", state.pid),
    )?;
    sleep(Duration::from_millis(800)).await;
    if mode == "leader" {
        let _ = wait_for_active_cluster(&settings.data_dir, Duration::from_secs(5)).await?;
    }
    println!("Bitty is running in the background (pid {}).", state.pid);
    Ok(())
}

fn stop_background_runtime(settings: &BittySettings) -> Result<(), Box<dyn std::error::Error>> {
    let Some(state) = read_runtime_state(&settings.data_dir) else {
        println!("Bitty is not running.");
        return Ok(());
    };
    if is_pid_running(state.pid) {
        if !kill_pid(state.pid) {
            return Err(format!("failed to stop Bitty runtime pid {}", state.pid).into());
        }
        // a node started right after this one needs the same ports
        let asked = std::time::Instant::now();
        while is_pid_running(state.pid) && asked.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(100));
        }
        println!("stopped Bitty runtime pid {}", state.pid);
    } else {
        println!("Bitty runtime pid {} was not running.", state.pid);
    }
    // kept aside, so that `bitty start` knows what to go back to
    let state_path = runtime_state_path(&settings.data_dir);
    if std::fs::rename(&state_path, stopped_state_path(&settings.data_dir)).is_err() {
        let _ = std::fs::remove_file(state_path);
    }
    Ok(())
}

fn print_runtime_summary(settings: &BittySettings) {
    if let Some(state) = read_runtime_state(&settings.data_dir) {
        let status = if is_pid_running(state.pid) {
            "running"
        } else {
            "stopped"
        };
        println!(
            "Bitty runtime: {status} (pid {}, mode {})",
            state.pid, state.mode
        );
        if !state.target.is_empty() {
            println!("Runtime cluster: {}", compact_invite(&state.target));
        }
    } else {
        println!("Bitty runtime: stopped");
    }
    if let Some(active) = active_cluster_target(settings) {
        println!("Active cluster: {}", compact_invite(&active));
    }
}

async fn wait_for_active_cluster(
    data_dir: &Path,
    wait: Duration,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let started = std::time::Instant::now();
    loop {
        let settings = BittySettings::load(data_dir.to_path_buf());
        if let Some(target) = active_cluster_target(&settings) {
            return Ok(Some(target));
        }
        if started.elapsed() >= wait {
            return Ok(None);
        }
        sleep(Duration::from_millis(200)).await;
    }
}

fn model_path_for_runtime(
    settings: &BittySettings,
    model: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut spec = resolve_model(settings, model);
    if let Some(candidate) = spec.clone() {
        if !candidate.model_path(settings).exists() && settings.auto_pull {
            spec = Some(pull_model(settings, &candidate.id())?);
        }
    }
    let spec = spec.ok_or_else(|| format!("model not found: {model}"))?;
    let path = spec.model_path(settings);
    if !path.exists() {
        return Err(format!(
            "model file is missing: {}. Run `bitty pull {}` or `bitty setup` first.",
            path.display(),
            spec.id()
        )
        .into());
    }
    Ok(path.display().to_string())
}

fn read_runtime_state(data_dir: &Path) -> Option<RuntimeState> {
    read_state_file(&runtime_state_path(data_dir))
}

fn read_state_file(path: &Path) -> Option<RuntimeState> {
    let contents = std::fs::read_to_string(path).ok()?;
    let mut state = RuntimeState::default();
    for line in contents.lines() {
        let Some((key, value)) = settings::parse_assignment(line) else {
            continue;
        };
        match key {
            "pid" => state.pid = value.parse().unwrap_or_default(),
            "mode" => state.mode = value.into(),
            "model" => state.model = value.into(),
            "target" => state.target = value.into(),
            _ => {}
        }
    }
    (state.pid > 0).then_some(state)
}

fn write_runtime_state(data_dir: &Path, state: &RuntimeState) -> std::io::Result<()> {
    let contents = format!(
        "pid = {}\nmode = \"{}\"\nmodel = \"{}\"\ntarget = \"{}\"\n",
        state.pid,
        escape_toml(&state.mode),
        escape_toml(&state.model),
        escape_toml(&state.target)
    );
    std::fs::write(runtime_state_path(data_dir), contents)
}

fn runtime_state_path(data_dir: &Path) -> PathBuf {
    data_dir.join("runtime").join("runtime.toml")
}

/// What the node was doing when `bitty stop` stopped it.
fn stopped_state_path(data_dir: &Path) -> PathBuf {
    data_dir.join("runtime").join("stopped.toml")
}

/// Whether the saved pid is still a bitty process. The number alone is not enough: after a
/// reboot another program may have it.
#[cfg(target_os = "linux")]
fn is_pid_running(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|name| name.trim().starts_with("bitty"))
        .unwrap_or(false)
}

#[cfg(windows)]
fn is_pid_running(pid: u32) -> bool {
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .stderr(Stdio::null())
        .output()
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .to_lowercase()
                .contains("\"bitty")
        })
        .unwrap_or(false)
}

#[cfg(not(any(target_os = "linux", windows)))]
fn is_pid_running(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(windows)]
fn kill_pid(pid: u32) -> bool {
    // /T takes the stage engines down with the node
    Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(not(windows))]
fn kill_pid(pid: u32) -> bool {
    Command::new("kill")
        .arg(pid.to_string())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn escape_toml(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn remember_cluster_alias(
    data_dir: &Path,
    name: Option<&str>,
    invite: &str,
    replace: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut store = cluster_store::ClusterStore::load(data_dir);
    let name = store.insert(name, invite, replace)?;
    store.save(data_dir)?;
    Ok(name)
}

fn remember_active_cluster(
    data_dir: &Path,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = BittySettings::load(data_dir.to_path_buf());
    settings.active_cluster = target.to_string();
    settings.save()?;
    Ok(())
}

fn default_model_path(data_dir: &Path) -> String {
    let settings = BittySettings::load(data_dir.to_path_buf());
    resolve_model(&settings, &settings.default_model)
        .map(|model| model.model_path(&settings).display().to_string())
        .unwrap_or_else(|| {
            settings
                .models_dir
                .join("qwen3.5-35b-a3b/latest/Qwen3.5-35B-A3B-Q4_K_M.gguf")
                .display()
                .to_string()
        })
}

fn compact_invite(invite: &str) -> String {
    let Some((head, _)) = invite.split_once("?token=") else {
        return invite.to_string();
    };
    format!("{head}?token=...")
}

fn confirm_yes(details: &str) -> Result<bool, std::io::Error> {
    println!("{details}");
    println!();
    println!("Type yes to confirm (anything else aborts):");
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line.trim() == "yes")
}

fn assert_destructive_data_dir(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.as_os_str().is_empty() {
        return Err("refusing empty data directory path".into());
    }
    if path == Path::new("/") {
        return Err("refusing destructive operation on filesystem root".into());
    }
    let allowed = std::env::var("BITTY_ALLOW_ANY_DATA_DIR_RESET")
        .map_or(false, |v| matches!(v.as_str(), "1" | "true" | "yes"))
        || path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == ".bitty");
    if !allowed {
        return Err(format!(
            "refusing to modify `{}`: directory name must be `.bitty`, or set BITTY_ALLOW_ANY_DATA_DIR_RESET=1",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn remove_path_if_exists(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

fn clean_local_state(settings: &BittySettings) -> std::io::Result<()> {
    let data_dir = &settings.data_dir;
    remove_path_if_exists(&settings.data_dir.join("models"))?;
    if settings.models_dir != settings.data_dir.join("models")
        && settings.models_dir.starts_with(data_dir)
    {
        remove_path_if_exists(&settings.models_dir)?;
    }
    remove_path_if_exists(&cluster_store::path(data_dir))?;
    remove_path_if_exists(&data_dir.join("cluster-token"))?;
    remove_path_if_exists(&data_dir.join("iroh-secret.key"))?;
    remove_path_if_exists(&data_dir.join("logs"))?;
    remove_path_if_exists(&data_dir.join("state"))?;
    remove_path_if_exists(&data_dir.join("runtime"))?;
    Ok(())
}

/// Reads `NAME=value` lines from `<data dir>/env` into the environment. This is where the
/// settings of a machine live (threads, GPU layers, ...): a node restarted by the updater or
/// after a reboot does not get the variables of the shell that started it the first time.
/// A variable that is already set wins over the file.
fn load_env_file(path: &Path) {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    for (name, value) in parse_env_file(&contents) {
        if std::env::var_os(&name).is_none() {
            std::env::set_var(name, value);
        }
    }
}

fn parse_env_file(contents: &str) -> Vec<(String, String)> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| line.strip_prefix("export ").unwrap_or(line).split_once('='))
        .map(|(name, value)| {
            let value = value.trim();
            let value = value
                .strip_prefix('"')
                .and_then(|inner| inner.strip_suffix('"'))
                .unwrap_or(value);
            (name.trim().to_string(), value.to_string())
        })
        .filter(|(name, _)| !name.is_empty())
        .collect()
}

fn bitty_data_dir(configured: Option<&str>) -> PathBuf {
    configured
        .map(PathBuf::from)
        .or_else(|| std::env::var("BITTY_DATA_DIR").ok().map(PathBuf::from))
        .or_else(|| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
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
        if validate_cluster_token(token).is_ok() {
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

/// The name of this machine in a network. It is made once and kept in the data directory, so a
/// node that restarts comes back as the same node and takes its old place in the plan.
fn stored_node_id(data_dir: &Path) -> String {
    let path = data_dir.join("node-id");
    let stored = std::fs::read_to_string(&path).unwrap_or_default();
    let stored = stored.trim();
    let valid = !stored.is_empty()
        && stored.len() <= 64
        && stored
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if valid {
        return stored.to_string();
    }
    let node_id = default_node_id();
    let _ = std::fs::create_dir_all(data_dir);
    let _ = std::fs::write(&path, format!("{node_id}\n"));
    node_id
}

/// The last check before layers are loaded: a stage larger than this machine can take now is
/// refused, whatever the leader planned. The reply carries the free memory, so the leader can
/// plan again with it.
fn refuse_stage_past_memory(stage: &Path) -> Result<(), String> {
    let mut profile = HardwareProfiler::new("check").profile();
    if profile.loads_into_gpu() {
        return Ok(());
    }
    if std::env::var("BITTY_AVAILABLE_RAM_MB").is_err() {
        profile.available_ram_mb =
            bitty_worker::profiler::available_ram_mb_now().min(profile.ram_mb);
    }
    let needed = std::fs::metadata(stage).map(|file| file.len()).unwrap_or(0)
        + bitty_protocol::stage_overhead_bytes();
    let limit = profile.hard_memory_bytes(bitty_protocol::MEMORY_RESERVE_FRACTION);
    if needed <= limit {
        return Ok(());
    }
    Err(format!(
        "these layers need {} MB and this machine can give {} MB now: free_memory_mb={}",
        needed / (1024 * 1024),
        limit / (1024 * 1024),
        profile.available_ram_mb
    ))
}

fn default_node_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("bitty-{:08x}", (nanos & 0xffff_ffff) as u32)
}

fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("bitty-{nanos:x}")
}

/// The layers the leader plans with, at the size they have in the model file. The planner
/// shares them by the memory of each machine, so the sizes have to be the real ones.
fn leader_layers(model_file: &Path, fallback_count: u32) -> Vec<LayerMetadata> {
    match bitty_model::ModelMetadata::from_gguf_path(model_file) {
        Ok(metadata) => {
            let layers = metadata.layer_metadata();
            if !layers.is_empty() && layers.iter().all(|layer| layer.weight_bytes > 0) {
                let bytes: u64 = layers.iter().map(|layer| layer.weight_bytes).sum();
                println!(
                    "planning with {} layers of about {} MB each",
                    layers.len(),
                    bytes / layers.len() as u64 / (1024 * 1024)
                );
                return layers;
            }
            eprintln!(
                "{} does not hold every layer: the planner does not know their size and will not limit by memory",
                model_file.display()
            );
            demo_layers(metadata.layer_count.max(1))
        }
        Err(err) => {
            eprintln!(
                "could not read the layers of {}: {err}; the planner will not limit by memory",
                model_file.display()
            );
            demo_layers(fallback_count)
        }
    }
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

fn bitty_version() -> String {
    option_env!("BITTY_GIT_SHA")
        .map(|sha| format!("{} ({sha})", env!("CARGO_PKG_VERSION")))
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

fn print_version() {
    println!("bitty {}", bitty_version());
    println!("repository: {}", env!("CARGO_PKG_REPOSITORY"));
}

fn print_help() {
    println!("Usage:");
    println!("  bitty setup");
    println!("  bitty share NAME");
    println!("  bitty connect INVITE_OR_NAME [--name NAME]");
    println!("  bitty start [--model MODEL_OR_PATH] [--if-started]");
    println!("  bitty stop [MODEL]");
    println!("  bitty restart [--if-started]");
    println!("  bitty run MODEL [PROMPT] [--local|--node TARGET]");
    println!("  bitty pull MODEL");
    println!("  bitty pull MODEL --layers START..END [--output FILE]");
    println!("  bitty ls | bitty list");
    println!("  bitty show MODEL");
    println!("  bitty ps");
    println!("  bitty serve [--host 127.0.0.1:11435]");
    println!("  bitty create NAME -f Modelfile");
    println!("  bitty rm MODEL");
    println!("  bitty cp SOURCE DEST");
    println!("  bitty settings get|set|path");
    println!("  bitty logs [--lines N|--path|--clear]");
    println!("  bitty invite [--name NAME] [--replace]");
    println!("  bitty join INVITE_OR_NAME [--name NAME] [--model PATH]");
    println!("  bitty use NAME_OR_INVITE [--name NAME]");
    println!("  bitty clusters");
    println!("  bitty cluster status|nodes|check|benchmark|invite [--node TARGET]");
    println!("  bitty node --model PATH");
    println!("  bitty node --join 'iroh://INVITE' --model PATH");
    println!("  bitty node --no-iroh --join HOST:PORT --model PATH");
    println!("  bitty generate --prompt TEXT [--node TARGET] [--max-tokens N] [--temperature T]");
    println!("                 [--top-k K] [--top-p P] [--seed N] [--think] [--raw] [--cluster-token TOKEN]");
    println!("  bitty chat [MODEL] [--prompt TEXT] [--node TARGET]");
    println!("  bitty status [--node TARGET]");
    println!(
        "  bitty clean [--data-dir DIR]   # models + cluster state + secrets; keeps config.toml"
    );
    println!("  bitty reset [--data-dir DIR]   # delete entire data dir; fresh default config");
    println!("  bitty version");
    println!();
    println!("Simple flow: `bitty setup`, then `bitty share home` or `bitty connect INVITE --name home`.");
    println!("Bitty starts the node runtime in the background for simple commands.");
    println!("Cluster names are local aliases. A reused name must match the same invite unless --replace is set.");
    println!("Use `--local` on `bitty run` to force local-only generation.");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs against a real model when `BITTY_TEST_MODEL` points at one.
    #[test]
    fn the_leader_plans_with_the_real_size_of_the_layers() {
        let missing = leader_layers(Path::new("/nonexistent/model.gguf"), 7);
        assert_eq!(missing.len(), 7);

        let Ok(model) = std::env::var("BITTY_TEST_MODEL") else {
            return;
        };
        let layers = leader_layers(Path::new(&model), 1);
        assert!(layers.len() > 1);
        assert!(layers.iter().all(|layer| layer.weight_bytes > 1024 * 1024));
        eprintln!(
            "{} layers, first {} MB, last {} MB",
            layers.len(),
            layers[0].weight_bytes >> 20,
            layers[layers.len() - 1].weight_bytes >> 20
        );
    }

    #[test]
    fn a_stopped_worker_is_remembered() {
        let dir = std::env::temp_dir().join(format!("bitty-stopped-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("runtime")).unwrap();
        let state = RuntimeState {
            pid: u32::MAX - 7,
            mode: "worker".into(),
            model: "qwen3.5-35b-a3b".into(),
            target: "iroh://leader?token=abc".into(),
        };
        write_runtime_state(&dir, &state).unwrap();
        let settings = BittySettings::defaults(dir.clone());
        stop_background_runtime(&settings).unwrap();
        assert!(read_runtime_state(&dir).is_none());
        assert_eq!(read_state_file(&stopped_state_path(&dir)), Some(state));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_node_keeps_its_id_across_restarts() {
        let dir = std::env::temp_dir().join(format!("bitty-node-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = stored_node_id(&dir);
        assert!(first.starts_with("bitty-"));
        assert_eq!(stored_node_id(&dir), first);

        // A file that is not an id is replaced.
        std::fs::write(dir.join("node-id"), "not an id / at all\n").unwrap();
        let replaced = stored_node_id(&dir);
        assert!(replaced.starts_with("bitty-"));
        assert_eq!(stored_node_id(&dir), replaced);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_clean_and_reset() {
        let clean = Cli::parse(vec![
            "clean".into(),
            "--data-dir".into(),
            "/tmp/.bitty".into(),
        ])
        .unwrap();
        match clean {
            CliCommand::Clean(cfg) => assert_eq!(cfg.data_dir.as_deref(), Some("/tmp/.bitty")),
            other => panic!("unexpected: {other:?}"),
        }
        let reset = Cli::parse(vec!["reset".into()]).unwrap();
        match reset {
            CliCommand::Reset(cfg) => assert!(cfg.data_dir.is_none()),
            other => panic!("unexpected: {other:?}"),
        }
    }

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
                "qwen3.5-35b-a3b".into(),
                "hello".into(),
                "--num-predict".into(),
                "4".into()
            ])
            .unwrap(),
            CliCommand::Run(RunConfig { max_tokens: 4, .. })
        ));
        assert!(matches!(
            Cli::parse(vec![
                "run".into(),
                "qwen3.5-35b-a3b".into(),
                "--local".into()
            ])
            .unwrap(),
            CliCommand::Run(RunConfig { local: true, .. })
        ));
        assert!(matches!(
            Cli::parse(vec!["pull".into(), "qwen3.5-35b-a3b".into()]).unwrap(),
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
                "qwen3.5-35b-a3b".into()
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
    fn parses_simple_cluster_alias_commands() {
        assert!(matches!(
            Cli::parse(vec!["setup".into()]).unwrap(),
            CliCommand::Setup(DataDirConfig { .. })
        ));
        assert!(matches!(
            Cli::parse(vec!["version".into()]).unwrap(),
            CliCommand::Version
        ));
        assert!(matches!(
            Cli::parse(vec![
                "start".into(),
                "--model".into(),
                "qwen3.5-35b-a3b".into()
            ])
            .unwrap(),
            CliCommand::Start(StartConfig { model: Some(_), .. })
        ));
        assert!(matches!(
            Cli::parse(vec!["restart".into(), "--if-started".into()]).unwrap(),
            CliCommand::Restart(StartConfig {
                if_started: true,
                ..
            })
        ));
        assert_eq!(
            parse_env_file(
                "# threads\nBITTY_STAGE_THREADS=8\nexport BITTY_GPU_LAYERS = \"12\"\n\nbroken\n"
            ),
            vec![
                ("BITTY_STAGE_THREADS".to_string(), "8".to_string()),
                ("BITTY_GPU_LAYERS".to_string(), "12".to_string()),
            ]
        );
        assert!(matches!(
            Cli::parse(vec!["stop".into()]).unwrap(),
            CliCommand::Stop(StopConfig { model: None, .. })
        ));
        assert!(matches!(
            Cli::parse(vec![
                "invite".into(),
                "--name".into(),
                "Home".into(),
                "--replace".into()
            ])
            .unwrap(),
            CliCommand::Invite(InviteConfig {
                name: Some(_),
                replace: true,
                ..
            })
        ));
        assert!(matches!(
            Cli::parse(vec!["share".into(), "Home".into()]).unwrap(),
            CliCommand::Share(InviteConfig { name: Some(_), .. })
        ));
        assert!(matches!(
            Cli::parse(vec![
                "join".into(),
                "iroh://abc?token=secret".into(),
                "--name".into(),
                "home".into(),
                "--model".into(),
                "/m.gguf".into()
            ])
            .unwrap(),
            CliCommand::Join(JoinConfig {
                name: Some(_),
                model: Some(_),
                ..
            })
        ));
        assert!(matches!(
            Cli::parse(vec![
                "connect".into(),
                "iroh://abc?token=secret".into(),
                "--name".into(),
                "home".into()
            ])
            .unwrap(),
            CliCommand::Connect(JoinConfig { name: Some(_), .. })
        ));
        assert!(matches!(
            Cli::parse(vec!["use".into(), "home".into()]).unwrap(),
            CliCommand::Use(UseConfig { target, .. }) if target == "home"
        ));
        assert!(matches!(
            Cli::parse(vec!["clusters".into()]).unwrap(),
            CliCommand::Clusters(DataDirConfig { .. })
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

    #[test]
    fn parses_pull_with_a_layer_range() {
        let command = Cli::parse(vec![
            "pull".into(),
            "qwen3.5-35b-a3b".into(),
            "--layers".into(),
            "10..20".into(),
        ])
        .unwrap();
        match command {
            CliCommand::Pull(config) => assert_eq!(config.layers, Some((10, 20))),
            other => panic!("unexpected command: {other:?}"),
        }
        for bad in ["20..10", "5", "a..b", "3..3"] {
            assert!(parse_layer_range(bad).is_err(), "{bad}");
        }
        // only pull takes a range
        assert!(Cli::parse(vec![
            "show".into(),
            "m".into(),
            "--layers".into(),
            "0..1".into()
        ])
        .is_err());
    }
}
