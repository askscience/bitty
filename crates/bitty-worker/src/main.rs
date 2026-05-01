use bitty_inference::FakeLayerExecutor;
use bitty_protocol::pb::coordinator_service_client::CoordinatorServiceClient;
use bitty_protocol::pb::{HeartbeatRequest, RegisterWorkerRequest};
use bitty_protocol::{AssignedLayerRange, LayerAssignment, NodeId, Quantization};
use bitty_worker::{keepalive, HardwareProfiler};
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = WorkerConfig::from_env();
    let profile = HardwareProfiler::new(config.node_id.clone()).profile();

    println!("bitty-worker: node={}", profile.node_id);
    println!(
        "profile tier={:?} cpu_tflops={:.2} memory_gb={:.1} os={}",
        profile.tier, profile.cpu_tflops, profile.memory_gb, profile.os
    );

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
        .register_worker(RegisterWorkerRequest {
            profile: Some((&profile).into()),
        })
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
            .heartbeat(HeartbeatRequest {
                node_id: profile.node_id.0.clone(),
                observed_tokens_per_second: profile.effective_compute_score(),
            })
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
}

impl WorkerConfig {
    fn from_env() -> Self {
        let mut config = Self {
            node_id: "local-worker-0".into(),
            keepalive: false,
            coordinator: None,
            heartbeat_count: 3,
            heartbeat_interval_ms: 1000,
        };
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--node-id" => {
                    config.node_id = args.next().unwrap_or_else(|| {
                        eprintln!("missing value for --node-id");
                        std::process::exit(2);
                    })
                }
                "--keepalive" => config.keepalive = true,
                "--coordinator" => {
                    config.coordinator = Some(args.next().unwrap_or_else(|| {
                        eprintln!("missing value for --coordinator");
                        std::process::exit(2);
                    }))
                }
                "--heartbeat-count" => {
                    config.heartbeat_count = parse_next(&mut args, "--heartbeat-count")
                }
                "--heartbeat-interval-ms" => {
                    config.heartbeat_interval_ms = parse_next(&mut args, "--heartbeat-interval-ms")
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

fn normalize_endpoint(endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.into()
    } else {
        format!("http://{endpoint}")
    }
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = args.next().unwrap_or_else(|| {
        eprintln!("missing value for {name}");
        std::process::exit(2);
    });
    value.parse().unwrap_or_else(|err| {
        eprintln!("invalid value for {name}: {err}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!(
        "Usage: cargo run -p bitty-worker -- [--node-id ID] [--keepalive] [--coordinator HOST:PORT]"
    );
    println!(
        "Example: cargo run -p bitty-worker -- --node-id pc2 --coordinator 192.168.1.10:50051"
    );
}
