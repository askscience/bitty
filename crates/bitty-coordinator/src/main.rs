use bitty_coordinator::{network::NetworkCoordinator, Halda, RingTopology, SchedulerConfig};
use bitty_protocol::{HardwareProfile, LayerMetadata, NodeId, NodeTier};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CoordinatorConfig::from_env();
    if let Some(listen_addr) = &config.listen {
        let coordinator = NetworkCoordinator::new(demo_layers(config.layers));
        let coordinator = if let Some(model_path) = &config.model {
            coordinator.with_model_path(model_path)
        } else {
            coordinator
        };
        coordinator.serve(listen_addr).await?;
        return Ok(());
    }

    let profiles = demo_profiles(config.nodes);
    let layers = demo_layers(config.layers);
    let assignments = Halda::new(SchedulerConfig::default()).assign(&profiles, &layers)?;
    let topology = RingTopology::new("local-coordinator-epoch-0", assignments);

    println!(
        "bitty-coordinator: scheduled {} layers across {} assignments",
        config.layers,
        topology.assignments.len()
    );
    println!("topology epoch={}", topology.epoch);

    for assignment in &topology.assignments {
        println!(
            "node={} layers={}..{} quant={:?} weight_bytes={} next={}",
            assignment.node_id,
            assignment.range.start_layer,
            assignment.range.end_layer_exclusive,
            assignment.range.quantization,
            assignment.assigned_weight_bytes,
            assignment
                .next_node_id
                .as_ref()
                .map(|node_id| node_id.0.as_str())
                .unwrap_or("<none>")
        );
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct CoordinatorConfig {
    nodes: usize,
    layers: u32,
    listen: Option<String>,
    model: Option<String>,
}

impl CoordinatorConfig {
    fn from_env() -> Self {
        let mut config = Self {
            nodes: 8,
            layers: 16,
            listen: None,
            model: None,
        };
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--nodes" => config.nodes = parse_next(&mut args, "--nodes"),
                "--layers" => config.layers = parse_next(&mut args, "--layers"),
                "--listen" => config.listen = Some(required_next(&mut args, "--listen")),
                "--model" => config.model = Some(required_next(&mut args, "--model")),
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

fn demo_profiles(count: usize) -> Vec<HardwareProfile> {
    (0..count)
        .map(|index| {
            let tier = match index {
                0 => NodeTier::S,
                1 | 2 => NodeTier::A,
                3 | 4 => NodeTier::B,
                _ => NodeTier::D,
            };
            HardwareProfile {
                node_id: NodeId::new(format!("local-{index}")),
                cpu_tflops: 0.5 + index as f64,
                gpu_tflops: match tier {
                    NodeTier::S => 30.0,
                    NodeTier::A => 12.0,
                    NodeTier::B => 3.0,
                    NodeTier::C | NodeTier::D => 0.0,
                },
                memory_gb: 4.0,
                memory_bandwidth_gbps: 20.0 + index as f64,
                disk_bandwidth_mbps: 400.0,
                network_rtt_ms: 10.0 + index as f64,
                uplink_mbps: 100.0,
                os: "local".into(),
                tier,
                ram_mb: 4096,
                vram_mb: 0,
                architecture: std::env::consts::ARCH.into(),
                gpus: Vec::new(),
                os_reclaim_score: 0.0,
                worker_endpoint: String::new(),
                model_path: String::new(),
                backend_type: if tier == NodeTier::D {
                    "cpu".into()
                } else {
                    "gpu".into()
                },
                layer_eligible: true,
                max_layers: u32::MAX,
            }
        })
        .collect()
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

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = required_next(args, name);
    value.parse().unwrap_or_else(|err| {
        eprintln!("invalid value for {name}: {err}");
        std::process::exit(2);
    })
}

fn required_next(args: &mut impl Iterator<Item = String>, name: &str) -> String {
    args.next().unwrap_or_else(|| {
        eprintln!("missing value for {name}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!("Usage: cargo run -p bitty-coordinator -- [--nodes N] [--layers N] [--listen ADDR] [--model PATH]");
    println!("Example: cargo run -p bitty-coordinator -- --listen 0.0.0.0:50051 --model /models/ggml-model-i2_s.gguf --layers 30");
}
