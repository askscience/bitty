use bitty_inference::FakeLayerExecutor;
use bitty_protocol::{AssignedLayerRange, LayerAssignment, NodeId, Quantization};
use bitty_worker::{keepalive, HardwareProfiler};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = WorkerConfig::from_env();
    let profile = HardwareProfiler::new(config.node_id.clone()).profile();

    println!("bitty-worker: node={}", profile.node_id);
    println!(
        "profile tier={:?} cpu_tflops={:.2} memory_gb={:.1} os={}",
        profile.tier, profile.cpu_tflops, profile.memory_gb, profile.os
    );

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

#[derive(Clone, Debug)]
struct WorkerConfig {
    node_id: String,
    keepalive: bool,
}

impl WorkerConfig {
    fn from_env() -> Self {
        let mut config = Self {
            node_id: "local-worker-0".into(),
            keepalive: false,
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

fn print_help() {
    println!("Usage: cargo run -p bitty-worker -- [--node-id ID] [--keepalive]");
}
