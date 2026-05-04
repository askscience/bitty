use bitty_inference::{BitNetLayerExecutor, FakeLayerExecutor};
use bitty_protocol::cli::{parse_next_or_exit, required_next_or_exit};
use bitty_protocol::endpoint::normalize_endpoint;
use bitty_protocol::BITTY_PROTOCOL_VERSION;
use bitty_protocol::pb::coordinator_service_client::CoordinatorServiceClient;
use bitty_protocol::pb::{HeartbeatRequest, RegisterWorkerRequest};
use bitty_protocol::security::{AuthMode, BITTY_TOKEN_HEADER};
use bitty_protocol::{AssignedLayerRange, LayerAssignment, ModelStage, NodeId, Quantization};
use bitty_worker::{keepalive, network::NetworkWorker, HardwareProfiler};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = WorkerConfig::from_env();
    let mut profile = HardwareProfiler::new(config.node_id.clone()).profile();
    if let Some(endpoint) = config
        .public_endpoint
        .clone()
        .or_else(|| config.listen.clone())
    {
        profile.worker_endpoint = endpoint;
    }

    println!("bitty-worker: node={}", profile.node_id);
    println!(
        "profile tier={:?} cpu_tflops={:.2} memory_gb={:.1} os={}",
        profile.tier, profile.cpu_tflops, profile.memory_gb, profile.os
    );

    if let Some(listen_addr) = &config.listen {
        if let Some(model_path) = &config.model {
            let executor = Arc::new(BitNetLayerExecutor::load(model_path).await?);
            let worker = NetworkWorker::new(NodeId::new(config.node_id.clone()), executor)
                .with_auth_mode(config.auth_mode());
            if let Some(coordinator) = &config.coordinator {
                let server = worker.clone();
                let listen_addr = listen_addr.clone();
                tokio::spawn(async move {
                    if let Err(err) = server.serve(&listen_addr).await {
                        eprintln!("worker server stopped: {err}");
                    }
                });
                run_network_worker(coordinator, &config, profile).await?;
            } else {
                worker.serve(listen_addr).await?;
            }
        } else {
            let worker = NetworkWorker::with_fake_executor(config.node_id.clone())
                .with_auth_mode(config.auth_mode());
            if let Some(coordinator) = &config.coordinator {
                let server = worker.clone();
                let listen_addr = listen_addr.clone();
                tokio::spawn(async move {
                    if let Err(err) = server.serve(&listen_addr).await {
                        eprintln!("worker server stopped: {err}");
                    }
                });
                run_network_worker(coordinator, &config, profile).await?;
            } else {
                worker.serve(listen_addr).await?;
            }
        }
        return Ok(());
    }

    if let Some(coordinator) = &config.coordinator {
        run_network_worker(coordinator, &config, profile).await?;
        return Ok(());
    }

    if config.keepalive {
        let assignment = LayerAssignment {
            node_id: NodeId::new(config.node_id),
            range: AssignedLayerRange {
                start_layer: 0,
                end_layer_exclusive: 1,
                quantization: Quantization::Bit1,
            },
            assigned_weight_bytes: 128,
            expected_latency_ms: 1.0,
            next_node_id: None,
            disk_offload_fraction: 0.0,
            model_stage: ModelStage::LayerRange,
        };
        keepalive::touch_weights(&FakeLayerExecutor, &assignment).await?;
        println!(
            "keepalive touched dummy assignment; interval_secs={}",
            keepalive::default_keepalive_interval().as_secs()
        );
    }

    Ok(())
}

async fn run_network_worker(
    coordinator: &str,
    config: &WorkerConfig,
    profile: bitty_protocol::HardwareProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = normalize_endpoint(coordinator);
    let mut client = CoordinatorServiceClient::connect(endpoint.clone()).await?;

    println!("connecting to coordinator at {endpoint}");
    let registration = client
        .register_worker(request_with_token(
            RegisterWorkerRequest {
                profile: Some((&profile).into()),
                protocol_version: BITTY_PROTOCOL_VERSION,
                inference_backend_id: if config.model.is_some() {
                    "bitnet".into()
                } else {
                    "stub".into()
                },
            },
            config.token.as_deref(),
        )?)
        .await?
        .into_inner();

    println!(
        "registered; topology_epoch={} assignments={}",
        registration.topology_epoch,
        registration.assignments.len()
    );
    for assignment in &registration.assignments {
        println!(
            "assignment node={} layers={}..{} quant={} next={}",
            assignment.node_id,
            assignment
                .range
                .as_ref()
                .map(|range| range.start_layer)
                .unwrap_or_default(),
            assignment
                .range
                .as_ref()
                .map(|range| range.end_layer_exclusive)
                .unwrap_or_default(),
            assignment
                .range
                .as_ref()
                .map(|range| range.quantization.as_str())
                .unwrap_or("<none>"),
            assignment.next_node_id
        );
    }

    for heartbeat_index in 0..config.heartbeat_count {
        sleep(Duration::from_millis(config.heartbeat_interval_ms)).await;
        let response = client
            .heartbeat(request_with_token(
                HeartbeatRequest {
                    node_id: profile.node_id.0.clone(),
                    observed_tokens_per_second: profile.effective_compute_score(),
                    avg_forward_latency_ms: 0.0,
                    activation_bytes_per_second: 0,
                    backend_type: profile.backend_type.clone(),
                },
                config.token.as_deref(),
            )?)
            .await?
            .into_inner();
        println!(
            "heartbeat={} accepted={} topology_epoch={}",
            heartbeat_index + 1,
            response.accepted,
            response.topology_epoch
        );
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct WorkerConfig {
    node_id: String,
    keepalive: bool,
    coordinator: Option<String>,
    heartbeat_count: u32,
    heartbeat_interval_ms: u64,
    listen: Option<String>,
    public_endpoint: Option<String>,
    model: Option<String>,
    token: Option<String>,
}

impl WorkerConfig {
    fn from_env() -> Self {
        let mut config = Self {
            node_id: "local-worker-0".into(),
            keepalive: false,
            coordinator: None,
            heartbeat_count: u32::MAX,
            heartbeat_interval_ms: 1000,
            listen: None,
            public_endpoint: None,
            model: None,
            token: None,
        };
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--node-id" => config.node_id = required_next_or_exit(&mut args, "--node-id"),
                "--keepalive" => config.keepalive = true,
                "--coordinator" => {
                    config.coordinator = Some(required_next_or_exit(&mut args, "--coordinator"))
                }
                "--heartbeat-count" => {
                    config.heartbeat_count = parse_next_or_exit(&mut args, "--heartbeat-count")
                }
                "--heartbeat-interval-ms" => {
                    config.heartbeat_interval_ms =
                        parse_next_or_exit(&mut args, "--heartbeat-interval-ms")
                }
                "--listen" => config.listen = Some(required_next_or_exit(&mut args, "--listen")),
                "--public-endpoint" => {
                    config.public_endpoint =
                        Some(required_next_or_exit(&mut args, "--public-endpoint"))
                }
                "--model" => config.model = Some(required_next_or_exit(&mut args, "--model")),
                "--token" | "--cluster-token" => {
                    config.token = Some(required_next_or_exit(&mut args, "--token"))
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => {
                    eprintln!("unknown argument: {unknown}");
                    print_help();
                    std::process::exit(2);
                }
            }
        }

        config
    }
}

impl WorkerConfig {
    fn auth_mode(&self) -> AuthMode {
        self.token
            .clone()
            .map(AuthMode::PreSharedToken)
            .unwrap_or(AuthMode::InsecureLocal)
    }
}

fn request_with_token<T>(
    message: T,
    token: Option<&str>,
) -> Result<tonic::Request<T>, Box<dyn std::error::Error>> {
    let mut request = tonic::Request::new(message);
    if let Some(token) = token.filter(|token| !token.is_empty()) {
        request
            .metadata_mut()
            .insert(BITTY_TOKEN_HEADER, token.parse()?);
    }
    Ok(request)
}

fn print_help() {
    println!(
        "Usage: cargo run -p bitty-worker -- [--node-id ID] [--keepalive] [--coordinator HOST:PORT] [--listen ADDR] [--public-endpoint HOST:PORT] [--model PATH] [--token TOKEN]"
    );
    println!(
        "Example: cargo run -p bitty-worker -- --node-id pc2 --coordinator 192.168.1.10:50051"
    );
}
