mod cluster_store;
mod gpu_select;
mod logger;
mod model_store;
mod modelfile;
mod secrets;
mod server;
mod settings;
mod ui;

use crate::ui::{prompt_bool, prompt_line, spinner};
use bitty_bitnet_runtime::BitNetRuntime;
use bitty_coordinator::network::NetworkCoordinator;
use bitty_inference::FakeLayerExecutor;
use bitty_protocol::endpoint::normalize_endpoint;
use bitty_protocol::iroh_transport::{
    self, IrohFrame, BITTY_SCHEDULER_ALPN, BITTY_WORKER_ALPN, DEFAULT_FRAME_LIMIT,
    SCHEDULER_CLUSTER_STATUS, SCHEDULER_GENERATE, SCHEDULER_HEARTBEAT, SCHEDULER_LIST_MODELS,
    SCHEDULER_REGISTER_WORKER, WORKER_APPLY_TOPOLOGY, WORKER_CLEANUP, WORKER_FINAL_LOGITS,
    WORKER_FORWARD_ACTIVATION, WORKER_LOAD_SHARD, WORKER_SAMPLE_TOKEN,
};
use bitty_protocol::pb::coordinator_service_client::CoordinatorServiceClient;
use bitty_protocol::pb::coordinator_service_server::CoordinatorService;
use bitty_protocol::pb::worker_service_server::WorkerService;
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, CleanupRequest, ClusterStatusRequest,
    ClusterStatusResponse, GenerateRequest, GenerateResponse, HeartbeatRequest, HeartbeatResponse,
    ListModelsRequest, ListModelsResponse, LoadShardRequest, RegisterWorkerRequest,
    RegisterWorkerResponse, SampleTokenRequest, TopologyUpdate,
};
use bitty_protocol::security::{
    constant_time_eq, validate_cluster_token, AuthMode, BITTY_TOKEN_HEADER,
};
use bitty_protocol::{HardwareProfile, NodeId, BITTY_PROTOCOL_VERSION};
use bitty_worker::{
    network::{NetworkWorker, RuntimeStats},
    HardwareProfiler,
};
use iroh::{endpoint::presets, Endpoint, EndpointAddr, EndpointId, SecretKey};
use model_store::{copy_model, installed_models, pull_model, registry_models, remove_model, resolve_model};
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
        Ok(CliCommand::Models(config)) => run_models(config).await,
        Ok(CliCommand::Clean(config)) => run_clean(config).await,
        Ok(CliCommand::Reset(config)) => run_reset(config).await,
        Ok(CliCommand::Hardware(config)) => run_hardware(config).await,
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
    Models(ModelsConfig),
    Clean(DataDirConfig),
    Reset(DataDirConfig),
    Hardware(DataDirConfig),
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
    layers: Option<u32>,
    heartbeat_interval_ms: u64,
    iroh: bool,
    data_dir: Option<String>,
    cluster_token: Option<String>,
    visibility: String,
    cluster_name: String,
    cluster_description: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GenerateConfig {
    node: String,
    prompt: String,
    prompt_tokens: Vec<u32>,
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
struct ModelsConfig {
    node: Option<String>,
    data_dir: Option<String>,
    verbose: bool,
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
    gpu: bool,
    gpu_backend: Option<String>,
    force_cpu: bool,
    seed: Option<u64>,
    debug_tokens: bool,
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
struct StopConfig {
    model: Option<String>,
    data_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StartConfig {
    model: Option<String>,
    join: Option<String>,
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
    Benchmark(ClusterConfig),
    Invite(DataDirConfig),
    Models(ClusterConfig),
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
            "models" => parse_models(&mut args).map(CliCommand::Models),
            "clean" => parse_data_dir(&mut args).map(CliCommand::Clean),
            "reset" => parse_data_dir(&mut args).map(CliCommand::Reset),
            "hardware" => parse_data_dir(&mut args).map(CliCommand::Hardware),
            "-h" | "--help" | "help" => Ok(CliCommand::Help),
            "-V" | "--version" | "version" => Ok(CliCommand::Version),
            other => Err(format!("unknown command: {other}")),
        }
    }
}

async fn run_node(mut config: NodeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = bitty_data_dir(config.data_dir.as_deref());
    if config.node_id.is_empty() {
        config.node_id = load_or_generate_node_id(&data_dir);
    }
    logger::log_default(format!(
        "starting node id={} model={} join={}",
        config.node_id,
        config.model,
        config.join.as_deref().unwrap_or("leader")
    ));
    let use_iroh = config.iroh || config.join.as_deref().is_some_and(is_iroh_join_target);
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
    let model_path = std::path::PathBuf::from(&config.model);

    // Load model metadata to derive real layer count from the GGUF
    let model_metadata = match bitty_inference::BitNetBackendProbe::load(&model_path) {
        Ok(probe) => probe.metadata,
        Err(err) => {
            return Err(format!("failed to read model metadata from {}: {err}", model_path.display()).into());
        }
    };
    let layer_metadata = model_metadata.layer_metadata();
    if let Some(cli_layers) = config.layers {
        if cli_layers != model_metadata.layer_count {
            eprintln!(
                "warning: --layers {cli_layers} overrides GGUF layer count {}; using GGUF value",
                model_metadata.layer_count
            );
        }
    }
    eprintln!(
        "info: model has {} layers, hidden_size={}, quant={}, arch={}",
        model_metadata.layer_count,
        model_metadata.hidden_size,
        model_metadata.quantization.as_str(),
        model_metadata.architecture.as_str(),
    );

    // Check whether the model is a BitNet variant (supports GPU sharded execution).
    // Non-BitNet models (llama, gemma, qwen, etc.) require --local for now.
    let is_bitnet_model = matches!(
        model_metadata.architecture,
        bitty_model::BitNetModelFamily::BitNetB158
    );
    if !is_bitnet_model && config.join.is_none() {
        return Err(format!(
            "model {} (arch={}, quant={}) is not a BitNet model.\n\
             Distributed (iroh) mode currently only supports BitNet i2_s or F16 models.\n\
             For llama / gemma / qwen / mistral / phi, run with --local:\n\
             \n  bitty run {} --local \"your prompt\"\n\n\
             or if starting a node:\n\
             \n  bitty run {} --local\n",
            model_metadata.architecture.as_str(),
            model_metadata.architecture.as_str(),
            model_metadata.quantization.as_str(),
            config.model,
            config.model,
        ).into());
    }

    // Load the real BitNet executor so inference produces real tokens and the
    // per-request stateful decoder runs. Falls back to the fake executor if
    // the model file can't be loaded (e.g. missing weights) so the node can
    // still run for cluster topology tests.
    let worker: NetworkWorker<bitty_inference::BitNetLayerExecutor> =
        match bitty_inference::BitNetLayerExecutor::load(&model_path).await {
            Ok(bitnet) => {
                let executor = Arc::new(bitnet);
                NetworkWorker::new(NodeId::new(config.node_id.clone()), executor)
                    .with_auth_mode(AuthMode::PreSharedToken(cluster_token.clone()))
            }
            Err(err) => {
                return Err(format!(
                    "failed to load BitNet runtime from {}: {err}",
                    model_path.display()
                )
                .into());
            }
        };
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
        let mut coordinator = NetworkCoordinator::new(layer_metadata)
            .with_model_path(&config.model)
            .with_auth_mode(AuthMode::PreSharedToken(cluster_token.clone()))
            .with_visibility(&config.visibility)
            .with_cluster_name(&config.cluster_name)
            .with_cluster_description(&config.cluster_description);
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
    let shutdown = Arc::new(tokio::sync::Notify::new());
    register_and_heartbeat(
        &scheduler_target,
        &config,
        iroh_node.as_ref(),
        local_worker_endpoint.as_deref(),
        &cluster_token,
        worker_stats,
        shutdown,
    )
    .await
}

async fn run_model(mut config: RunConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = load_settings(config.data_dir.as_deref());
    logger::log(
        &settings.data_dir,
        format!("running model {}", config.model),
    )?;
    let mut model = resolve_model(&settings, &config.model);
    if model.is_none() && config.auto_pull && settings.auto_pull {
        model = pull_model(&settings, &config.model).ok();
    }
    if model.is_none() && config.prompt.is_none() && !config.model.contains('/') {
        let default = resolve_model(&settings, &settings.default_model);
        if let Some(default_model) = default {
            config.prompt = Some(config.model.clone());
            model = Some(default_model);
        }
    }
    let model = model.ok_or_else(|| format!("model not found: {}", config.model))?;
    // Warn or refuse on experimental/unsupported models
    match model.status.as_str() {
        "unsupported" => {
            return Err(format!(
                "model {} is unsupported: {}. Try a different model with `bitty run MODEL`.",
                model.id(),
                model.display_name
            ).into());
        }
        "experimental" => {
            eprintln!(
                "warning: model {} is experimental and may produce incorrect output.",
                model.id()
            );
        }
        _ => {}
    }
    if !config.local && config.node.is_none() && settings.auto_start_node {
        let model_path = model.model_path(&settings);
        if !model_path.exists() && settings.auto_pull {
            pull_model(&settings, &model.id())?;
        }
        ensure_background_runtime(&settings, Some(model_path.display().to_string()), None).await?;
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
            println!(
                "  {}",
                ui::dim(&format!("cluster: {}", compact_invite(&node)))
            );
        }
        // Load the tokenizer once so we can send properly-tokenized prompts to
        // the cluster instead of raw UTF-8 bytes (the coordinator otherwise
        // interprets each byte as a token ID, which produces garbled output).
        let model_path = model.model_path(&settings);
        let hf_source = if model.source.is_empty() { None } else { Some(model.source.as_str()) };
        let tokenizer = if model_path.exists() {
            match bitty_bitnet_runtime::load_tokenizer(&model_path, hf_source) {
                Ok(tok) => Some(tok),
                Err(err) => {
                    eprintln!(
                        "warning: failed to load tokenizer ({err}); prompts will be sent unencoded"
                    );
                    None
                }
            }
        } else {
            None
        };
        let tokenize = |text: &str| -> Vec<u32> {
            tokenizer
                .as_ref()
                .and_then(|tok| tok.encode(text, true).ok())
                .unwrap_or_default()
        };
        if prompt.is_empty() {
            println!(
                "  {}  {}  {}",
                ui::dim("chat"),
                ui::bold(&model.id()),
                ui::dim("type /exit to quit")
            );
            loop {
                print!("  {}›{} ", ui::C, ui::N);
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
                let tokens = tokenize(line);
                run_generate(GenerateConfig {
                    node: node.clone(),
                    prompt: line.into(),
                    prompt_tokens: tokens,
                    max_tokens: config.max_tokens,
                    temperature: config.temperature.clone(),
                    data_dir: config.data_dir.clone(),
                })
                .await?;
            }
            return Ok(());
        }
        let tokens = tokenize(&prompt);
        return run_generate(GenerateConfig {
            node,
            prompt,
            prompt_tokens: tokens,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            data_dir: config.data_dir,
        })
        .await;
    }
    if prompt.is_empty() {
        println!(
            "  {}  {}  {}",
            ui::dim("chat"),
            ui::bold(&model.id()),
            ui::dim("type /exit to quit")
        );
        let path = model.model_path(&settings);
        let temp = config.temperature.parse().unwrap_or(model.temperature);
        let hf_source = if model.source.is_empty() { None } else { Some(model.source.as_str()) };
        let is_bitnet = model.backend.contains("bitnet") || model.backend.contains("i2s");
        // Default: auto-detect GPU. --cpu forces CPU-only. Explicit backend flags (--cuda etc.) override auto-detect.
        let use_gpu = !config.force_cpu || config.gpu_backend.is_some();

        // Load model once, keep alive across turns
        let mut bitnet_runtime: Option<BitNetRuntime> = None;
        let mut cpu_model: Option<bitty_bitnet_runtime::cpu_backend::CpuModel> = None;
        let mut gpu_backend_kind: Option<gpu_select::GpuBackendKind> = None;
        let mut candle_model: Option<bitty_candle_runtime::CandleModel> = None;
        let mut messages: Vec<bitty_bitnet_runtime::ChatMessage> = Vec::new();
        let tok = bitty_bitnet_runtime::load_tokenizer(&path, hf_source)
            .expect("failed to load tokenizer");

        if use_gpu && !is_bitnet {
            let requested = config.gpu_backend.as_deref()
                .or(if settings.preferred_gpu_backend != "auto" { Some(settings.preferred_gpu_backend.as_str()) } else { None })
                .map(gpu_select::GpuBackendKind::from_cli_flag);
            let kind = gpu_select::select_backend(requested);

            // Try candle GPU backends (CUDA / Metal / ROCm)
            if matches!(kind, gpu_select::GpuBackendKind::Cuda | gpu_select::GpuBackendKind::Metal | gpu_select::GpuBackendKind::Rocm | gpu_select::GpuBackendKind::Auto) {
                let device = bitty_candle_runtime::auto_device();
                if let Ok(m) = bitty_candle_runtime::CandleModel::load(&path.to_string_lossy(), &device) {
                    eprintln!("  {}  {}", ui::dim("running on"), ui::dim(&kind.name()));
                    candle_model = Some(m);
                    gpu_backend_kind = Some(kind);
                }
            }
            // Try wgpu backends (Vulkan / DX12 / wgpu-Metal)
            if candle_model.is_none() && matches!(kind, gpu_select::GpuBackendKind::Vulkan | gpu_select::GpuBackendKind::Dx12 | gpu_select::GpuBackendKind::MetalWgpu | gpu_select::GpuBackendKind::Auto) {
                #[cfg(feature = "gpu-wgpu")]
                if let Ok(dev) = bitty_wgpu_runtime::WgpuDevice::new(bitty_wgpu_runtime::GpuBackend::Auto) {
                    eprintln!("  {}  {}", ui::dim("running on"), ui::dim(&dev.adapter_info));
                    gpu_backend_kind = Some(kind);
                }
            }
            // If nothing worked, fall through to CPU silently
        }

        if gpu_backend_kind.is_none() && !config.force_cpu && use_gpu && !settings.gpu_fallback_to_cpu {
            return Err(format!(
                "GPU not available and gpu_fallback_to_cpu is false. Use --cpu or set gpu_fallback_to_cpu = true in config."
            ).into());
        }

        if gpu_backend_kind.is_none() {
            if !config.force_cpu && !config.local {
                eprintln!("  {}", ui::dim("running on CPU"));
            }
            if is_bitnet {
                match BitNetRuntime::load(&path, hf_source).await {
                    Ok(rt) => bitnet_runtime = Some(rt),
                    Err(e) => {
                        eprintln!("GPU unavailable ({}), using CPU backend...", e);
                        cpu_model = Some(
                            bitty_bitnet_runtime::cpu_backend::CpuModel::load(&path, hf_source)
                                .map_err(|e| format!("CPU load error: {e}"))?
                        );
                    }
                }
            } else {
                cpu_model = Some(
                    bitty_bitnet_runtime::cpu_backend::CpuModel::load(&path, hf_source)
                        .map_err(|e| format!("CPU load error: {e}"))?
                );
            }
        }

        let mut reset_cache = true;
        loop {
            print!("  {}›{} ", ui::C, ui::N);
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

            // Apply chat template
            messages.push(bitty_bitnet_runtime::ChatMessage {
                role: "user".into(),
                content: line.to_string(),
            });
            let prompt_ids = tok.apply_chat_template(&messages)
                .unwrap_or_else(|_| tok.encode(line, true).unwrap_or_default());

            // Build prompt from template (decode template tokens back to text)
            // for the GPU BitNet path, which expects raw text.
            let prompt_text = tok.decode(&prompt_ids).unwrap_or_else(|_| line.to_string());

            if config.debug_tokens {
                eprintln!("debug: prompt token ids ({}) {:?}", prompt_ids.len(), prompt_ids);
            }

            struct Streamer { started: bool }
            impl Streamer {
                fn new() -> Self { Self { started: false } }
                fn emit(&mut self, delta: &str) {
                    if delta.is_empty() { return; }
                    if !self.started { print!("  "); self.started = true; }
                    print!("{delta}");
                    let _ = io::stdout().flush();
                }
            }
            let mut streamer = Streamer::new();
            let max_t = config.max_tokens as usize;

            let response: String = if let Some(ref mut rt) = bitnet_runtime {
                rt.generate_stream_raw(&prompt_text, max_t, temp, reset_cache, |d| streamer.emit(d))
                    .await
                    .map_err(|e| format!("generate error: {e}"))?
            } else if let Some(ref mut candle) = candle_model {
                // candle GPU generation loop with KV cache
                if reset_cache {
                    candle.reset_kv_cache();
                }
                let mut generated = Vec::new();
                let mut emitted = String::new();
                // Process prompt tokens through model (populates KV cache)
                // Only feed prompt_ids on first turn or after cache reset;
                // on subsequent turns the cache already contains history.
                if reset_cache || generated.is_empty() {
                    let _ = candle.forward(&prompt_ids)
                        .map_err(|e| format!("candle forward error: {e}"))?;
                }
                // Auto-regress from KV cache
                let mut current = vec![0u32]; // dummy, will be overwritten
                for _step in 0..max_t {
                    let logits = candle.forward(&current)
                        .map_err(|e| format!("candle forward error: {e}"))?;
                    let next = bitty_candle_runtime::sample_token(
                        &mut logits.clone(), temp, 40, 1.0, 1.1, &generated,
                    );
                    if next == tok.eos_token_id()
                        || tok.eot_token_id() == Some(next)
                        || tok.im_end_token_id() == Some(next)
                    {
                        break;
                    }
                    generated.push(next);
                    if let Ok(full) = tok.decode(&generated) {
                        if full.len() > emitted.len() && full.starts_with(&emitted) {
                            let tail = &full[emitted.len()..];
                            if !tail.ends_with('\u{FFFD}') {
                                streamer.emit(tail);
                            }
                            emitted = full;
                        }
                    }
                    current = vec![next];
                }
                emitted
            } else if let Some(ref cpu) = cpu_model {
                cpu.generate_chat_stream(&messages, reset_cache, max_t, temp, 40, 1.1, config.seed, |d| streamer.emit(d))
                    .map_err(|e| format!("CPU generate error: {e}"))?
            } else {
                break;
            };

            if config.debug_tokens {
                eprintln!("debug: response = {:?}", response);
            }

            println!();
            messages.push(bitty_bitnet_runtime::ChatMessage {
                role: "assistant".into(),
                content: response,
            });
            reset_cache = false;
        }
        return Ok(());
    }
    run_local_model(
        &settings,
        &model,
        &prompt,
        config.max_tokens,
        &config.temperature,
        !config.force_cpu,
        config.gpu_backend.as_deref()
            .or(if settings.preferred_gpu_backend != "auto" { Some(settings.preferred_gpu_backend.as_str()) } else { None }),
    )
    .await
}

async fn run_local_model(
    settings: &BittySettings,
    model: &model_store::ModelSpec,
    prompt: &str,
    max_tokens: u32,
    temperature: &str,
    use_gpu: bool,
    gpu_backend: Option<&str>,
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
    let temp = temperature.parse().unwrap_or(model.temperature);

    // Helper: stream deltas to stdout with flush, prefix the first chunk.
    struct Streamer {
        started: bool,
    }
    impl Streamer {
        fn new() -> Self {
            Self { started: false }
        }
        fn emit(&mut self, delta: &str) {
            if delta.is_empty() {
                return;
            }
            if !self.started {
                print!("  ");
                self.started = true;
            }
            print!("{delta}");
            let _ = io::stdout().flush();
        }
    }

    let is_bitnet = model.backend.contains("bitnet") || model.backend.contains("i2s");
    let hf_source = if model.source.is_empty() { None } else { Some(model.source.as_str()) };
    let mut streamer = Streamer::new();

    // If GPU requested, try GPU backends first
    let gpu_tried = if use_gpu && !is_bitnet {
        let requested = gpu_backend.map(gpu_select::GpuBackendKind::from_cli_flag);
        let kind = gpu_select::select_backend(requested);

        let mut worked = false;
        if matches!(kind, gpu_select::GpuBackendKind::Cuda | gpu_select::GpuBackendKind::Metal | gpu_select::GpuBackendKind::Rocm | gpu_select::GpuBackendKind::Auto) {
            let device = bitty_candle_runtime::auto_device();
            if let Ok(mut candle) = bitty_candle_runtime::CandleModel::load(&path.to_string_lossy(), &device) {
                eprintln!("  {}  {}", ui::dim("running on"), ui::dim(kind.name()));
                let tok = bitty_bitnet_runtime::load_tokenizer(&path, hf_source)
                    .expect("tokenizer");
                let mut generated = Vec::new();
                let mut emitted = String::new();
                let prompt_ids = tok.encode(prompt, true).unwrap_or_default();
                let mut current = prompt_ids;
                let eos = tok.eos_token_id();
                let eot = tok.eot_token_id();
                let im_end = tok.im_end_token_id();
                for _step in 0..max_tokens as usize {
                    let logits = candle.forward(&current)
                        .map_err(|e| format!("candle forward error: {e}"))?;
                    let next = bitty_candle_runtime::sample_token(
                        &mut logits.clone(), temp, 40, 1.0, 1.1, &generated,
                    );
                    if next == eos || Some(next) == eot || Some(next) == im_end {
                        break;
                    }
                    generated.push(next);
                    if let Ok(full) = tok.decode(&generated) {
                        if full.len() > emitted.len() && full.starts_with(&emitted) {
                            let tail = &full[emitted.len()..];
                            if !tail.ends_with('\u{FFFD}') {
                                streamer.emit(tail);
                            }
                            emitted = full;
                        }
                    }
                    current = vec![next];
                }
                worked = true;
            }
        }
        worked
    } else {
        false
    };

    let result: Result<(), String> = if gpu_tried {
        Ok(())
    } else if is_bitnet {
        match spinner("loading model", BitNetRuntime::load(&path, hf_source)).await {
            Ok(mut runtime) => runtime
                .generate_stream(prompt, max_tokens as usize, temp, |delta| {
                    streamer.emit(delta);
                })
                .await
                .map(|_| ())
                .map_err(|e| format!("GPU generate error: {e}")),
            Err(gpu_err) => {
                eprintln!("GPU unavailable ({}), using CPU backend...", gpu_err);
                let cpu_model = spinner("loading model (CPU)", async {
                    bitty_bitnet_runtime::cpu_backend::CpuModel::load(&path, hf_source)
                })
                .await?;
                cpu_model
                    .generate_stream(
                        prompt,
                        max_tokens as usize,
                        temp,
                        40,
                        1.1,
                        |delta| streamer.emit(delta),
                    )
                    .map(|_| ())
            }
        }
    } else {
        // Non-BitNet: go directly to CPU
        let cpu_model = spinner("loading model (CPU)", async {
            bitty_bitnet_runtime::cpu_backend::CpuModel::load(&path, hf_source)
        })
        .await
        .map_err(|e| format!("CPU load error: {e}"))?;
        cpu_model
            .generate_stream(prompt, max_tokens as usize, temp, 40, 1.1, |delta| {
                streamer.emit(delta);
            })
            .map(|_| ())
    };
    result.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    println!();
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
        "  {} {}  {}",
        ui::green("pulled"),
        ui::bold(&model.id()),
        ui::dim(&model.model_path(&settings).display().to_string())
    );
    Ok(())
}

async fn run_list(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let installed = installed_models(&settings);
    let registry = registry_models();

    if installed.is_empty() && registry.is_empty() {
        println!("  {}no models available{}", ui::Y, ui::N);
        return Ok(());
    }

    println!(
        "  {:<24} {:<12} {:<14} {}",
        ui::dim("name"),
        ui::dim("backend"),
        ui::dim("quantization"),
        ui::dim("status")
    );

    let installed_ids: std::collections::HashSet<String> =
        installed.iter().map(|m| m.id()).collect();

    for model in &installed {
        println!(
            "  {:<24} {:<12} {:<14} {}",
            ui::bold(&model.id()),
            ui::dim(&model.backend),
            model.quantization,
            ui::green("installed")
        );
    }

    for model in &registry {
        if !installed_ids.contains(&model.id()) {
            println!(
                "  {:<24} {:<12} {:<14} {}",
                ui::dim(&model.id()),
                ui::dim(&model.backend),
                model.quantization,
                ui::dim("available")
            );
        }
    }

    Ok(())
}

async fn run_show(config: ModelCommand) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let model = resolve_model(&settings, &config.model)
        .ok_or_else(|| format!("model not found: {}", config.model))?;
    println!("  {}  {}", ui::dim("name"), ui::bold(&model.id()));
    println!("  {}  {}", ui::dim("display"), model.display_name);
    println!("  {}  {}", ui::dim("backend"), &model.backend);
    println!("  {}  {}", ui::dim("quantization"), &model.quantization);
    println!("  {}  {}", ui::dim("layers"), model.layers);
    println!(
        "  {}  {}",
        ui::dim("path"),
        ui::dim(&model.model_path(&settings).display().to_string())
    );
    println!("  {}  {}", ui::dim("source"), &model.source);
    println!("  {}  {}", ui::dim("temperature"), model.temperature);
    println!("  {}  {}", ui::dim("num_predict"), model.num_predict);
    println!("  {}  {}", ui::dim("num_ctx"), model.num_ctx);
    Ok(())
}

async fn run_ps(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    print_runtime_summary(&settings);
    let path = running_models_path(&settings);
    let contents = std::fs::read_to_string(path).unwrap_or_default();
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        println!("  {}  {}{}{}", ui::bold(line), ui::G, "loaded", ui::N);
    }
    if contents.trim().is_empty() {
        println!("  {}no loaded models{}", ui::Y, ui::N);
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
    println!("  {}  {}", ui::green("stopped"), ui::bold(&model));
    Ok(())
}

async fn run_start(config: StartConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    ensure_background_runtime(&settings, config.model, config.join).await?;
    print_runtime_summary(&settings);
    Ok(())
}

async fn run_restart(config: StartConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let _ = stop_background_runtime(&settings);
    ensure_background_runtime(&settings, config.model, config.join).await?;
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

const C: &str = "\x1b[36m";
const G: &str = "\x1b[32m";
const Y: &str = "\x1b[33m";
const D: &str = "\x1b[2m";
const B: &str = "\x1b[1m";
const R: &str = "\x1b[0m";

fn setup_header(settings: &BittySettings) {
    println!();
    println!(
        "  {}bitty setup{}  {}v{}  ·  {}{}",
        B,
        R,
        D,
        bitty_version(),
        settings.data_dir.display(),
        R
    );
    println!("  {}", "─".repeat(50));
    println!();
}
fn setup_status(settings: &BittySettings) {
    let model = resolve_model(settings, &settings.default_model)
        .unwrap_or_else(|| model_store::find_registry_model(&settings.default_model).unwrap());
    let exists = model.model_path(settings).exists();
    print!("  {}model{}  {}{}{}", D, R, B, model.id(), R);
    if exists {
        println!("  {}ready{}", G, R);
    } else {
        println!("  {}not downloaded{}", Y, R);
    }

    let cluster = active_cluster_target(settings);
    print!("  {}cluster{}  ", D, R);
    if cluster.is_some() {
        println!("{}configured{}", G, R);
    } else {
        println!("{}not configured{}", Y, R);
    }
    println!();
}

async fn setup_model(settings: &BittySettings) -> Result<(), Box<dyn std::error::Error>> {
    let registry = model_store::registry_models();
    let installed = model_store::installed_models(settings);

    println!("  {}model selection{}", B, R);
    println!();

    for m in &registry {
        let is_installed = installed.iter().any(|i| i.name == m.name);
        let status = if is_installed {
            format!("{}downloaded{}", G, R)
        } else {
            format!("{}not downloaded{}", Y, R)
        };
        println!("  {} {:<16} {}  {}", D, m.id(), m.display_name, status);
    }

    println!();
    let name = prompt_line("model name or URL (Enter for default)");
    let model_name = if name.is_empty() {
        settings.default_model.clone()
    } else {
        name
    };

    let spec = model_store::find_registry_model(&model_name).or_else(|| {
        let path = std::path::Path::new(&model_name);
        if path.exists() {
            let layers = model_store::peek_gguf_layers(path);
            Some(model_store::ModelSpec {
                name: model_name.clone(),
                tag: "local".into(),
                display_name: "Local GGUF model".into(),
                backend: "bitnet-i2s".into(),
                quantization: "i2_s".into(),
                filename: path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model.gguf")
                    .into(),
                layers,
                url: String::new(),
                source: String::new(),
                temperature: settings.default_temperature,
                num_predict: settings.default_num_predict,
                num_ctx: settings.default_num_ctx,
                path: Some(path.to_path_buf()),
                status: "stable".into(),
            })
        } else {
            None
        }
    });

    let Some(spec) = spec else {
        println!("  {}unknown model: {}{}", Y, model_name, R);
        return Ok(());
    };

    let model_path = spec.model_path(settings);
    if model_path.exists() {
        println!("  {}model ready: {}{}", G, model_path.display(), R);
        return Ok(());
    }

    if spec.url.is_empty() && spec.path.is_none() {
        println!("  {}no download source for {}{}", Y, spec.id(), R);
        return Ok(());
    }

    if let Some(ref custom_path) = spec.path {
        println!("  {}using: {}{}", G, custom_path.display(), R);
        return Ok(());
    }

    if prompt_bool(&format!("download {}?", spec.display_name), true) {
        println!();
        let model = pull_model(settings, &model_name)?;
        println!(
            "  {} {} {}downloaded to {}{}",
            G,
            B,
            D,
            model.model_path(settings).display(),
            R
        );
    }
    Ok(())
}

async fn setup_cluster(settings: &BittySettings) -> Result<(), Box<dyn std::error::Error>> {
    let existing = active_cluster_target(settings);
    if existing.is_some() {
        println!("  {}cluster already configured{}", G, R);
        if !prompt_bool("switch to a different cluster?", false) {
            return Ok(());
        }
    }

    println!();
    println!("  {}join  ─ paste an invite from another machine{}", D, R);
    println!("  {}share ─ start your own cluster{}", D, R);
    println!();
    let choice = prompt_line("join or share? (j/s, Enter to skip)");
    match choice.to_lowercase().as_str() {
        "j" | "join" => {
            let invite = prompt_line("paste the invite");
            if !invite.is_empty() {
                let _ = run_connect(JoinConfig {
                    target: invite,
                    name: None,
                    replace: true,
                    model: None,
                    node_id: None,
                    data_dir: Some(settings.data_dir.display().to_string()),
                })
                .await;
            }
        }
        "s" | "share" => {
            println!();
            println!("  {}private ─ invite-only, requires cluster token{}", D, R);
            println!(
                "  {}public ─ anyone can browse models and cluster info{}",
                D, R
            );
            println!();
            let visibility = if prompt_bool("make cluster public?", false) {
                "public"
            } else {
                "private"
            };
            let mut settings = settings.clone();
            settings.cluster_mode = visibility.to_string();
            if visibility == "public" {
                let name = prompt_line("cluster name (leave empty to skip)");
                if !name.is_empty() {
                    settings.cluster_name = name;
                }
                let desc = prompt_line("description (leave empty to skip)");
                if !desc.is_empty() {
                    settings.cluster_description = desc;
                }
            }
            settings.save()?;
            let _ = run_share(InviteConfig {
                name: Some("home".into()),
                replace: true,
                data_dir: Some(settings.data_dir.display().to_string()),
            })
            .await;
        }
        _ => {}
    }
    Ok(())
}

async fn run_setup(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    setup_header(&settings);
    spinner("checking", async {
        tokio::time::sleep(Duration::from_millis(400)).await
    })
    .await;
    setup_status(&settings);

    let model_ready = resolve_model(&settings, &settings.default_model)
        .map(|m| m.model_path(&settings).exists())
        .unwrap_or(false);
    let cluster_ready = active_cluster_target(&settings).is_some();

    if model_ready && cluster_ready {
        println!("  {}all set!{}", G, R);
        println!("  run {}bitty run bitnet-b1.58 \"hello\"{}", B, R);
        println!();
        return Ok(());
    }

    if !model_ready {
        setup_model(&settings).await?;
        println!();
    }

    if !cluster_ready {
        setup_cluster(&settings).await?;
        println!();
    }

    setup_status(&settings);
    println!("  {}done{}", G, R);
    if !cluster_ready {
        println!("  run {}bitty share home{} to start your cluster", B, R);
        println!(
            "  or connect with: {}bitty connect INVITE --name home{}",
            B, R
        );
    }
    println!();
    Ok(())
}

async fn run_clean(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    assert_destructive_data_dir(&settings.data_dir)?;
    if !ui::confirm_destructive(
        "This removes local models, saved cluster aliases, cluster token, Iroh identity key, logs, and runtime state.\n\
         Your config.toml is kept (API host and defaults), but the active cluster is cleared.",
    ) {
        println!("  {} {}", ui::yellow("aborted"), ui::dim("no changes"));
        return Ok(());
    }
    let mut settings = settings;
    stop_background_runtime(&settings)?;
    clean_local_state(&settings)?;
    settings.active_cluster.clear();
    settings.save()?;
    println!(
        "  {}  cleaned {}",
        ui::green("done"),
        ui::dim(&settings.data_dir.display().to_string())
    );
    println!(
        "  {}  run {}bitty setup{} or {}bitty pull{} to restore models",
        ui::dim("next"),
        ui::B,
        ui::N,
        ui::B,
        ui::N
    );
    Ok(())
}

async fn run_reset(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    assert_destructive_data_dir(&settings.data_dir)?;
    if !ui::confirm_destructive(
        "This permanently deletes the entire Bitty data directory (models, config, clusters, tokens, keys, logs).\n\
         Afterward you get a fresh default config, as after a first install.",
    ) {
        println!("  {} {}", ui::yellow("aborted"), ui::dim("no changes"));
        return Ok(());
    }
    let data_dir = settings.data_dir.clone();
    stop_background_runtime(&settings)?;
    std::fs::remove_dir_all(&data_dir)?;
    BittySettings::defaults(data_dir.clone()).save()?;
    println!(
        "  {}  reset {}",
        ui::green("done"),
        ui::dim(&data_dir.display().to_string())
    );
    println!(
        "  {}  run {}bitty setup{} to get started",
        ui::dim("next"),
        ui::B,
        ui::N
    );
    Ok(())
}

async fn run_hardware(config: DataDirConfig) -> Result<(), Box<dyn std::error::Error>> {
    let _settings = load_settings(config.data_dir.as_deref());
    println!("  bitty hardware");
    println!();
    println!("  CPU:");
    let cpu_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let os = std::env::consts::OS;
    println!("    platform:     {os}");
    println!("    cpu threads:  {cpu_count}");

    // wgpu probe
    print!("    wgpu:         ");
    #[cfg(feature = "gpu-wgpu")]
    {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            println!("no adapters found");
        } else {
            println!("{} adapter(s)", adapters.len());
            for a in &adapters {
                let info = a.get_info();
                println!("      - {} ({:?}, {:?})", info.name, info.backend, info.device_type);
            }
        }
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        println!("not compiled (enable --features gpu-wgpu)");
    }

    // candle backends
    println!();
    println!("  Available backends:");
    println!("    cpu:           always available");
    #[cfg(feature = "gpu-cuda")] { println!("    cuda:          compiled"); }
    #[cfg(not(feature = "gpu-cuda"))] { println!("    cuda:          not compiled"); }
    #[cfg(feature = "gpu-rocm")] { println!("    rocm:          compiled (experimental)"); }
    #[cfg(not(feature = "gpu-rocm"))] { println!("    rocm:          not compiled"); }
    #[cfg(feature = "gpu-metal")] { println!("    metal:         compiled"); }
    #[cfg(not(feature = "gpu-metal"))] { println!("    metal:         not compiled (enable --features gpu-metal)"); }

    // CUDA probe
    let has_cuda = std::process::Command::new("nvidia-smi")
        .arg("-L").output().map(|o| o.status.success()).unwrap_or(false);
    if has_cuda { println!("    cuda runtime:  detected (nvidia-smi)"); }

    println!();
    println!("  Auto-selection order (default):");
    if cfg!(feature = "gpu-cuda") && has_cuda { println!("    1. candle-cuda"); }
    if cfg!(feature = "gpu-metal") && os == "macos" { println!("    2. candle-metal"); }
    if cfg!(feature = "gpu-rocm") && os == "linux" { println!("    3. candle-rocm"); }
    if cfg!(feature = "gpu-wgpu") { println!("    4. wgpu (Vulkan/Metal/DX12)"); }
    println!("    fallback: CPU");

    Ok(())
}

async fn run_create(config: CreateConfig) -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_settings(config.data_dir.as_deref());
    let model = modelfile::create_profile(&settings, &config.name, &PathBuf::from(config.file))?;
    println!("  {}  {}", ui::green("created"), ui::bold(&model.id()));
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
        ClusterCommand::Models(config) => {
            let models =
                fetch_list_models(config.node.as_deref(), config.data_dir.as_deref()).await?;
            print_list_models(&models, false);
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
    println!("  {}  {}", ui::dim("invite name"), ui::bold(&name));
    println!("  {}", invite);
    println!();
    println!(
        "  {}  bitty connect '{}' --name {}{}",
        ui::dim("on another machine"),
        invite,
        name,
        ui::N
    );
    println!(
        "  {}bitty stop{} to stop the background runtime",
        ui::dim(""),
        ui::N
    );
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
    let settings = BittySettings::load(data_dir.clone());
    let _ = stop_background_runtime(&settings);
    run_node(NodeConfig {
        model: config
            .model
            .unwrap_or_else(|| default_model_path(&data_dir)),
        node_id: config.node_id.unwrap_or_default(),
        listen: "0.0.0.0:50051".into(),
        worker_listen: None,
        public_endpoint: None,
        join: Some(invite),
        layers: None,
        heartbeat_interval_ms: 1000,
        iroh: true,
        data_dir: config.data_dir,
        cluster_token: None,
        visibility: "private".into(),
        cluster_name: String::new(),
        cluster_description: String::new(),
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
    println!("  {} cluster `{}`{}", ui::green("connected"), name, ui::N);
    println!(
        "  run {}bitty run bitnet-b1.58 \"hello\"{}",
        ui::bold(""),
        ui::N
    );
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
    let has_any = store.aliases().next().is_some();
    if !has_any {
        println!("  {}no saved clusters{}", ui::Y, ui::N);
        return Ok(());
    }
    for (name, invite) in store.aliases() {
        let is_active = Some(invite) == active.as_deref();
        let marker = if is_active {
            ui::green("active")
        } else {
            ui::dim("")
        };
        println!(
            "  {} {}  {}",
            if is_active {
                ui::bold(name)
            } else {
                ui::dim(name)
            },
            marker,
            ui::dim(&compact_invite(invite))
        );
    }
    Ok(())
}

async fn run_models(config: ModelsConfig) -> Result<(), Box<dyn std::error::Error>> {
    let response = fetch_list_models(config.node.as_deref(), config.data_dir.as_deref()).await?;
    print_list_models(&response, config.verbose);
    if config.verbose {
        let status =
            fetch_cluster_status(config.node.as_deref(), config.data_dir.as_deref()).await?;
        ui::rule();
        print_cluster_status(status, true);
    }
    Ok(())
}

async fn fetch_list_models(
    node: Option<&str>,
    data_dir: Option<&str>,
) -> Result<ListModelsResponse, Box<dyn std::error::Error>> {
    let node = resolve_cluster_node(node.unwrap_or(""), data_dir)?;
    let response = if let Some(target) = iroh_transport::parse_iroh_target(&node) {
        let endpoint = start_iroh_client().await?;
        let client = IrohSchedulerClient {
            endpoint,
            remote: target.endpoint_addr,
            token: target.token.unwrap_or_default(),
        };
        client
            .request::<_, ListModelsResponse>(
                SCHEDULER_LIST_MODELS,
                &ListModelsRequest {
                    cluster_name: String::new(),
                },
            )
            .await?
    } else {
        let mut client = CoordinatorServiceClient::connect(normalize_endpoint(&node)).await?;
        client
            .list_models(ListModelsRequest {
                cluster_name: String::new(),
            })
            .await?
            .into_inner()
    };
    Ok(response)
}

fn print_list_models(response: &ListModelsResponse, verbose: bool) {
    if verbose {
        ui::header("bitty models", &PathBuf::from(""), &bitty_version());
        let visibility = if response.visibility == "public" {
            ui::green(&response.visibility)
        } else {
            ui::dim(&response.visibility)
        };
        println!(
            "  {}  {}",
            ui::dim("cluster"),
            ui::bold(&response.cluster_name)
        );
        if !response.cluster_description.is_empty() {
            println!(
                "  {}  {}",
                ui::dim("description"),
                response.cluster_description
            );
        }
        println!("  {}  {}", ui::dim("visibility"), visibility);
        println!("  {}  {}", ui::dim("workers"), response.active_workers);
    }
    let model_status = if response.model_ready {
        ui::ready()
    } else {
        ui::not_ready()
    };
    println!("  {}  {}", ui::dim("model"), model_status);
    if !response.model_path.is_empty() {
        println!("  {}  {}", ui::dim("path"), ui::dim(&response.model_path));
    }
    println!("  {}  {}", ui::dim("layers"), response.layer_count);
    ui::rule();
    if response.model_ready && verbose {
        println!();
        println!(
            "  run {}bitty run \"hello\"{} to generate text",
            ui::bold(""),
            ui::N
        );
    }
    println!();
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
    shutdown: Arc<tokio::sync::Notify>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut profile = HardwareProfiler::new(config.node_id.clone()).profile();
    profile.network_rtt_ms = estimate_scheduler_rtt_ms(target).await;
    profile.model_path = config.model.clone();
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
                        inference_backend_id: "bitnet".into(),
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
                tokio::select! {
                    _ = shutdown.notified() => break,
                    _ = sleep(Duration::from_millis(config.heartbeat_interval_ms)) => {}
                }
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
            Ok(())
        }
        SchedulerTarget::Tcp(leader) => {
            let endpoint = normalize_endpoint(leader);
            let mut client = CoordinatorServiceClient::connect(endpoint.clone()).await?;
            let registration = client
                .register_worker(request_with_token(
                    RegisterWorkerRequest {
                        profile: Some((&profile).into()),
                        protocol_version: BITTY_PROTOCOL_VERSION,
                        inference_backend_id: "bitnet".into(),
                    },
                    cluster_token,
                ))
                .await?
                .into_inner();
            println!(
                "bitty node joined scheduler at {endpoint}; topology_epoch={} assignments={}",
                registration.topology_epoch,
                registration.assignments.len()
            );
            loop {
                tokio::select! {
                    _ = shutdown.notified() => break,
                    _ = sleep(Duration::from_millis(config.heartbeat_interval_ms)) => {}
                }
                let snapshot = worker_stats.snapshot().await;
                let response = client
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
            Ok(())
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
                        protocol_version: BITTY_PROTOCOL_VERSION,
                        inference_backend_id: "bitnet".into(),
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
                tokio::select! {
                    _ = shutdown.notified() => break,
                    _ = sleep(Duration::from_millis(config.heartbeat_interval_ms)) => {}
                }
                let snapshot = worker_stats.snapshot().await;
                let response = client
                    .request::<_, HeartbeatResponse>(
                        SCHEDULER_HEARTBEAT,
                        &HeartbeatRequest {
                            node_id: profile.node_id.0.clone(),
                            observed_tokens_per_second: observed_tokens_per_second(
                                &profile,
                                snapshot.observed_tokens_per_second,
                            ),
                            avg_forward_latency_ms: snapshot.avg_forward_latency_ms,
                            activation_bytes_per_second: snapshot.activation_bytes_per_second,
                            backend_type: profile.backend_type.clone(),
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
            Ok(())
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
    let prompt_tokens = config.prompt_tokens.clone();
    if let Some(target) = iroh_transport::parse_iroh_target(&node) {
        let endpoint = spinner("connecting", async { start_iroh_client().await }).await?;
        let client = IrohSchedulerClient {
            endpoint,
            remote: target.endpoint_addr,
            token: target.token.unwrap_or_default(),
        };
        let response: GenerateResponse = spinner("generating", async {
            client
                .request(
                    SCHEDULER_GENERATE,
                    &GenerateRequest {
                        request_id: request_id(),
                        prompt_tokens,
                        prompt: config.prompt,
                        max_new_tokens: config.max_tokens,
                        temperature: config.temperature.parse().unwrap_or(0.0),
                    },
                )
                .await
        })
        .await?;
        print!("  ");
        for token in response.tokens {
            print!("{}", token.text);
            let _ = io::stdout().flush();
            if token.finished {
                println!();
            }
        }
        return Ok(());
    }

    let mut client = spinner("connecting", async {
        CoordinatorServiceClient::connect(normalize_endpoint(&node)).await
    })
    .await?;
    let mut stream = spinner("generating", async {
        client
            .generate(GenerateRequest {
                request_id: request_id(),
                prompt_tokens,
                prompt: config.prompt,
                max_new_tokens: config.max_tokens,
                temperature: config.temperature.parse().unwrap_or(0.0),
            })
            .await
            .map(|r| r.into_inner())
    })
    .await?;

    print!("  ");
    let _ = io::stdout().flush();
    while let Some(token) = stream.message().await? {
        print!("{}", token.text);
        let _ = io::stdout().flush();
        if token.finished {
            println!();
        }
    }
    Ok(())
}

async fn run_chat(config: ChatConfig) -> Result<(), Box<dyn std::error::Error>> {
    let model = config
        .model
        .clone()
        .unwrap_or_else(|| load_settings(config.data_dir.as_deref()).default_model);
    run_model(RunConfig {
        model,
        prompt: config.prompt,
        node: config.node,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        data_dir: config.data_dir,
        auto_pull: true,
        local: false,
        gpu: false,
        gpu_backend: None,
        force_cpu: false,
        seed: None,
        debug_tokens: false,
    })
    .await
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
    let ready = if status.model_ready {
        ui::ready()
    } else {
        ui::not_ready()
    };
    ui::rule();
    println!("  {}  {}", ui::dim("leader"), status.leader_id);
    println!("  {}  {}", ui::dim("epoch"), status.topology_epoch);
    println!("  {}  {}", ui::dim("workers"), status.active_workers);
    println!("  {}  {}", ui::dim("model"), ready);
    if !status.model_path.is_empty() {
        println!("  {}  {}", ui::dim("path"), ui::dim(&status.model_path));
    }
    println!("  {}  {}", ui::dim("assignments"), status.assignments.len());
    if !include_assignments {
        return;
    }
    if !status.assignments.is_empty() {
        println!();
    }
    for assignment in status.assignments {
        if let Some(range) = assignment.range {
            let node = &assignment.node_id;
            let next = &assignment.next_node_id;
            println!(
                "  {}  layers {}..{}  {}  next:{}",
                ui::dim(node),
                range.start_layer,
                range.end_layer_exclusive,
                ui::dim(&assignment.model_stage),
                ui::dim(next)
            );
        }
    }
}

fn print_cluster_nodes(status: ClusterStatusResponse) {
    for assignment in status.assignments {
        if let Some(range) = assignment.range {
            let node = &assignment.node_id;
            let next = &assignment.next_node_id;
            println!(
                "  {}  layers {}..{}  {}  next:{}",
                ui::bold(node),
                range.start_layer,
                range.end_layer_exclusive,
                ui::dim(&assignment.model_stage),
                ui::dim(next)
            );
        }
    }
}

fn print_cluster_check(status: &ClusterStatusResponse) {
    let green = status.active_workers > 0 && status.model_ready;
    let status_text = if green {
        ui::green("ready")
    } else {
        ui::yellow("not ready")
    };
    println!("  {} {}", ui::dim("status"), status_text);
    println!("  {}  {}", ui::dim("leader"), status.leader_id);
    println!("  {}  {}", ui::dim("workers"), status.active_workers);
    println!(
        "  {}  {}",
        ui::dim("model"),
        if status.model_ready {
            ui::ready()
        } else {
            ui::not_ready()
        }
    );
    println!("  {}  {}", ui::dim("assignments"), status.assignments.len());
    if !status.profiles.is_empty() {
        println!();
        print_cluster_benchmark(status);
    }
}

fn print_cluster_benchmark(status: &ClusterStatusResponse) {
    println!("  {} {}", ui::dim("hardware"), ui::dim("profiles"));
    for profile in &status.profiles {
        let backend = if profile.backend_type.is_empty() {
            if profile.gpu_tflops > 0.0 {
                "gpu"
            } else {
                "cpu"
            }
        } else {
            profile.backend_type.as_str()
        };
        let eligible = if profile.layer_eligible {
            ui::green("yes")
        } else {
            ui::yellow("no")
        };
        println!(
            "  {}  {}  {}  {} MB  {} MB  {} TFLOPS  {} TFLOPS  {}ms  {} Mbps",
            ui::bold(&profile.node_id),
            ui::dim(backend),
            eligible,
            profile.ram_mb,
            profile.vram_mb,
            profile.cpu_tflops,
            profile.gpu_tflops,
            profile.network_rtt_ms,
            profile.uplink_mbps
        );
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
        layers: None,
        heartbeat_interval_ms: 1000,
        iroh: true,
        data_dir: None,
        cluster_token: None,
        visibility: "private".into(),
        cluster_name: String::new(),
        cluster_description: String::new(),
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
            "--layers" => {
                let raw: String = required_next(args, "--layers")?;
                config.layers = Some(raw.parse::<u32>().map_err(|_| format!("invalid --layers value: {raw}"))?);
            }
            "--heartbeat-interval-ms" => {
                config.heartbeat_interval_ms = parse_next(args, "--heartbeat-interval-ms")?
            }
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            "--cluster-token" => {
                config.cluster_token = Some(required_next(args, "--cluster-token")?)
            }
            "--visibility" => config.visibility = required_next(args, "--visibility")?,
            "--cluster-name" => config.cluster_name = required_next(args, "--cluster-name")?,
            "--description" => config.cluster_description = required_next(args, "--description")?,
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
        prompt_tokens: Vec::new(),
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

fn parse_models(args: &mut impl Iterator<Item = String>) -> Result<ModelsConfig, String> {
    let mut config = ModelsConfig {
        node: None,
        data_dir: None,
        verbose: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node" => config.node = Some(required_next(args, "--node")?),
            "--data-dir" => config.data_dir = Some(required_next(args, "--data-dir")?),
            "--verbose" | "-v" | "--detail" => config.verbose = true,
            other => return Err(format!("unknown models argument: {other}")),
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
        gpu: false,
        gpu_backend: None,
        force_cpu: false,
        seed: None,
        debug_tokens: false,
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
            "--cpu" => config.force_cpu = true,
            "--gpu" => config.gpu = true,
            "--vulkan" => {
                config.gpu = true;
                config.gpu_backend = Some("vulkan".into());
            }
            "--metal-gpu" => {
                config.gpu = true;
                config.gpu_backend = Some("metal".into());
            }
            "--dx12" => {
                config.gpu = true;
                config.gpu_backend = Some("dx12".into());
            }
            "--cuda" => {
                config.gpu = true;
                config.gpu_backend = Some("cuda".into());
            }
            "--rocm" => {
                config.gpu = true;
                config.gpu_backend = Some("rocm".into());
            }
            "--backend" => {
                config.gpu = true;
                config.gpu_backend = Some(required_next(args, "--backend")?);
            }
            "--debug-tokens" => config.debug_tokens = true,
            "--no-daemon" => {}
            "--join" => config.node = Some(required_next(args, "--join")?),
            "--seed" => {
                config.seed = Some(parse_next(args, "--seed")?);
            }
            "--num-ctx" | "--top-k" | "--top-p" | "--system" | "--template" => {
                let _ = required_next(args, arg.as_str()).ok();
            }
            value if config.model.is_empty() => config.model = value.into(),
            value => config.prompt = Some(value.into()),
        }
    }
    if config.model.is_empty() {
        config.model = load_settings(config.data_dir.as_deref()).default_model;
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
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
        "models" => parse_cluster_config(args).map(ClusterCommand::Models),
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
    fn serve_protocols<E>(
        &self,
        coordinator: Option<NetworkCoordinator>,
        worker: NetworkWorker<E>,
        cluster_token: String,
    ) where
        E: bitty_inference::LayerExecutor + Clone + 'static,
    {
        let endpoint = self.endpoint.clone();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(256));
        tokio::spawn(async move {
            while let Some(incoming) = endpoint.accept().await {
                let coordinator = coordinator.clone();
                let worker = worker.clone();
                let cluster_token = cluster_token.clone();
                let permit = match semaphore.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => {
                        eprintln!("iroh: max concurrent connections reached, dropping incoming");
                        continue;
                    }
                };
                tokio::spawn(async move {
                    let _permit = permit;
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

async fn handle_iroh_request<E>(
    incoming: iroh::endpoint::Incoming,
    coordinator: Option<NetworkCoordinator>,
    worker: NetworkWorker<E>,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>>
where
    E: bitty_inference::LayerExecutor + Clone + 'static,
{
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
    let is_query_op = matches!(frame.op, SCHEDULER_CLUSTER_STATUS | SCHEDULER_LIST_MODELS);
    let is_public = coordinator.visibility() == "public";
    let skip_token = is_query_op && is_public;

    if !skip_token && !constant_time_eq(frame.token.as_bytes(), cluster_token.as_bytes()) {
        return Err("invalid cluster token".into());
    }

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
            let response = if skip_token {
                coordinator
                    .cluster_status(tonic::Request::new(request))
                    .await?
            } else {
                coordinator
                    .cluster_status(request_with_token(request, cluster_token))
                    .await?
            };
            IrohFrame::message(SCHEDULER_CLUSTER_STATUS, "", &response.into_inner())
        }
        SCHEDULER_LIST_MODELS => {
            let request: ListModelsRequest = frame.decode_message(SCHEDULER_LIST_MODELS)?;
            let response = if skip_token {
                coordinator
                    .list_models(tonic::Request::new(request))
                    .await?
            } else {
                coordinator
                    .list_models(request_with_token(request, cluster_token))
                    .await?
            };
            IrohFrame::message(SCHEDULER_LIST_MODELS, "", &response.into_inner())
        }
        _ => return Err(format!("unknown scheduler op {}", frame.op).into()),
    };
    Ok(response)
}

async fn handle_worker_connection<E>(
    connection: iroh::endpoint::Connection,
    worker: NetworkWorker<E>,
    cluster_token: String,
) -> Result<(), Box<dyn std::error::Error>>
where
    E: bitty_inference::LayerExecutor + Clone + 'static,
{
    let (mut send, mut recv) = connection.accept_bi().await?;
    let frame = iroh_transport::read_frame(&mut recv, DEFAULT_FRAME_LIMIT).await?;
    if !constant_time_eq(frame.token.as_bytes(), cluster_token.as_bytes()) {
        return Err("invalid cluster token".into());
    }
    let response = match handle_worker_frame(frame, worker, &cluster_token).await {
        Ok(response) => response,
        Err(err) => iroh_transport::error_frame(err),
    };
    iroh_transport::write_frame(&mut send, &response).await?;
    send.finish()?;
    send.stopped().await?;
    Ok(())
}

async fn handle_worker_frame<E>(
    frame: IrohFrame,
    worker: NetworkWorker<E>,
    cluster_token: &str,
) -> Result<IrohFrame, Box<dyn std::error::Error>>
where
    E: bitty_inference::LayerExecutor + Clone + 'static,
{
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
            if let Some(ref new_join) = join_override {
                if state.target != *new_join {
                    stop_background_runtime(settings)?;
                } else {
                    return Ok(());
                }
            } else {
                return Ok(());
            }
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
    let model_path = model_path_for_runtime(settings, &model)?;
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
    command.arg("--visibility").arg(&settings.cluster_mode);
    if !settings.cluster_name.is_empty() {
        command.arg("--cluster-name").arg(&settings.cluster_name);
    }
    if !settings.cluster_description.is_empty() {
        command
            .arg("--description")
            .arg(&settings.cluster_description);
    }
    let mode = if join.is_some() { "worker" } else { "leader" };
    let mut child = command.spawn()?;
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
        if let Err(err) = wait_for_active_cluster(&settings.data_dir, Duration::from_secs(5)).await
        {
            let _ = child.kill();
            return Err(format!("cluster failed to become active within 5s: {err}").into());
        }
    }
    println!("Bitty is running in the background (pid {}).", state.pid);
    Ok(())
}

fn stop_background_runtime(settings: &BittySettings) -> Result<(), Box<dyn std::error::Error>> {
    let Some(state) = read_runtime_state(&settings.data_dir) else {
        println!("  {} {}", ui::dim("runtime"), ui::yellow("not running"));
        return Ok(());
    };
    if is_pid_running(state.pid) {
        let status = Command::new("kill").arg(state.pid.to_string()).status()?;
        if !status.success() {
            return Err(format!("failed to stop Bitty runtime pid {}", state.pid).into());
        }
        println!(
            "  {} {} pid {}",
            ui::green("stopped"),
            ui::dim("runtime"),
            state.pid
        );
    } else {
        println!(
            "  {} {} pid {} was {}",
            ui::yellow("stopped"),
            ui::dim("runtime"),
            state.pid,
            ui::dim("not running")
        );
    }
    let _ = std::fs::remove_file(runtime_state_path(&settings.data_dir));
    Ok(())
}

fn print_runtime_summary(settings: &BittySettings) {
    if let Some(state) = read_runtime_state(&settings.data_dir) {
        let alive = is_pid_running(state.pid);
        let status = if alive {
            ui::green("running")
        } else {
            ui::yellow("stopped")
        };
        println!(
            "  {} {}  (pid {}, mode {})",
            ui::dim("runtime"),
            status,
            state.pid,
            state.mode
        );
        if !state.target.is_empty() && alive {
            println!(
                "  {} {}",
                ui::dim("cluster"),
                ui::dim(&compact_invite(&state.target))
            );
        }
    } else {
        println!("  {} {}", ui::dim("runtime"), ui::yellow("stopped"));
    }
    if let Some(active) = active_cluster_target(settings) {
        println!(
            "  {} {}",
            ui::dim("active cluster"),
            ui::dim(&compact_invite(&active))
        );
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
    let contents = std::fs::read_to_string(runtime_state_path(data_dir)).ok()?;
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

fn is_pid_running(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
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
                .join("bitnet-b1.58/latest/ggml-model-i2_s.gguf")
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

fn node_id_path(data_dir: &Path) -> PathBuf {
    data_dir.join("node-id")
}

fn load_or_generate_node_id(data_dir: &Path) -> String {
    let path = node_id_path(data_dir);
    if let Ok(id) = std::fs::read_to_string(&path) {
        let id = id.trim().to_string();
        if !id.is_empty() {
            return id;
        }
    }
    let id = default_node_id();
    let _ = std::fs::write(&path, &id);
    id
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

fn bitty_version() -> String {
    option_env!("BITTY_GIT_SHA")
        .map(|sha| format!("{} ({sha})", env!("CARGO_PKG_VERSION")))
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

fn print_version() {
    println!("  {}  {}", ui::bold("bitty"), ui::dim(&bitty_version()));
    println!(
        "  {}  {}",
        ui::dim("repository"),
        env!("CARGO_PKG_REPOSITORY")
    );
}

fn print_help() {
    println!("  {}bitty{} — distributed inference runtime", ui::B, ui::N);
    println!(
        "  {}",
        ui::dim(&format!(
            "v{}  {}",
            bitty_version(),
            env!("CARGO_PKG_REPOSITORY")
        ))
    );
    println!();
    println!("  {}", ui::dim("getting started"));
    println!(
        "    {}     {}",
        ui::bold("bitty setup"),
        ui::dim("one-time setup wizard")
    );
    println!(
        "    {}     {}",
        ui::bold("bitty share NAME"),
        ui::dim("share this machine as a cluster leader")
    );
    println!(
        "    {}  {}",
        ui::bold("bitty connect INVITE"),
        ui::dim("join an existing cluster")
    );
    println!(
        "    {}          {}",
        ui::bold("bitty run MODEL"),
        ui::dim("run inference")
    );
    println!();
    println!("  {}", ui::dim("commands"));
    println!(
        "    {}          {}",
        ui::bold("bitty pull MODEL"),
        ui::dim("download a model")
    );
    println!(
        "    {}                     {}",
        ui::bold("bitty ls"),
        ui::dim("list installed models")
    );
    println!(
        "    {}          {}",
        ui::bold("bitty show MODEL"),
        ui::dim("show model details")
    );
    println!(
        "    {}                       {}",
        ui::bold("bitty ps"),
        ui::dim("show running models")
    );
    println!(
        "    {}               {}",
        ui::bold("bitty stop [MODEL]"),
        ui::dim("stop runtime or model")
    );
    println!(
        "    {}                    {}",
        ui::bold("bitty start"),
        ui::dim("start background runtime")
    );
    println!(
        "    {}                 {}",
        ui::bold("bitty restart"),
        ui::dim("restart background runtime")
    );
    println!(
        "    {}                    {}",
        ui::bold("bitty serve"),
        ui::dim("start HTTP API server")
    );
    println!(
        "    {}  {}",
        ui::bold("bitty create NAME -f FILE"),
        ui::dim("create model from Modelfile")
    );
    println!(
        "    {}               {}",
        ui::bold("bitty rm MODEL"),
        ui::dim("remove a model")
    );
    println!(
        "    {}            {}",
        ui::bold("bitty cp SRC DEST"),
        ui::dim("copy a model profile")
    );
    println!(
        "    {}                {}",
        ui::bold("bitty settings"),
        ui::dim("view or change settings")
    );
    println!(
        "    {}                    {}",
        ui::bold("bitty logs"),
        ui::dim("view runtime logs")
    );
    println!(
        "    {}                 {}",
        ui::bold("bitty version"),
        ui::dim("show version")
    );
    println!();
    println!("  {}", ui::dim("cluster"));
    println!(
        "    {}                   {}",
        ui::bold("bitty invite"),
        ui::dim("print cluster invite")
    );
    println!(
        "    {}          {}",
        ui::bold("bitty join INVITE"),
        ui::dim("join as a foreground worker")
    );
    println!(
        "    {}             {}",
        ui::bold("bitty use NAME"),
        ui::dim("switch active cluster")
    );
    println!(
        "    {}                {}",
        ui::bold("bitty clusters"),
        ui::dim("list saved clusters")
    );
    println!(
        "    {}  {}",
        ui::bold("bitty cluster status"),
        ui::dim("cluster status & assignments")
    );
    println!(
        "    {}   {}",
        ui::bold("bitty cluster nodes"),
        ui::dim("cluster topology")
    );
    println!(
        "    {}   {}",
        ui::bold("bitty cluster check"),
        ui::dim("cluster readiness check")
    );
    println!(
        "    {}  {}",
        ui::bold("bitty cluster benchmark"),
        ui::dim("hardware profiles")
    );
    println!(
        "    {}         {}",
        ui::bold("bitty node --model PATH"),
        ui::dim("advanced: run a node")
    );
    println!(
        "    {}               {}",
        ui::bold("bitty generate"),
        ui::dim("send a generate request")
    );
    println!(
        "    {}                     {}",
        ui::bold("bitty chat"),
        ui::dim("interactive chat")
    );
    println!(
        "    {}                  {}",
        ui::bold("bitty status"),
        ui::dim("full cluster + runtime status")
    );
    println!();
    println!("  {}", ui::dim("maintenance"));
    println!(
        "    {}                    {}",
        ui::bold("bitty clean"),
        ui::dim("remove models, config, cluster state")
    );
    println!(
        "    {}                    {}",
        ui::bold("bitty reset"),
        ui::dim("delete everything, fresh start")
    );
    println!();
    println!(
        "  {}  background runtime auto-starts for simple commands, {} to stop",
        ui::dim("tip:"),
        ui::dim("bitty stop")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
                "bitnet-b1.58".into()
            ])
            .unwrap(),
            CliCommand::Start(StartConfig { model: Some(_), .. })
        ));
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
}
