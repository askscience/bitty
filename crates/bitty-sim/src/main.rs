use bitty_protocol::NodeId;
use bitty_sim::{demo_layers, demo_profiles, ChaosProfile, SimulatedCluster};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SimConfig::from_env();
    let mut cluster =
        SimulatedCluster::build(demo_profiles(config.nodes), demo_layers(config.layers))?;

    if config.drop_node.is_some() || config.corrupt_node.is_some() {
        cluster = cluster.with_chaos(ChaosProfile {
            drop_node: config.drop_node.map(NodeId::new),
            corrupt_node: config.corrupt_node.map(NodeId::new),
        });
    }

    println!(
        "bitty-sim: nodes={} layers={} tokens={}",
        config.nodes, config.layers, config.tokens
    );
    println!(
        "topology epoch={} assignments={}",
        cluster.topology().epoch,
        cluster.topology().assignments.len()
    );

    let stream = cluster.stream_tokens("local", config.tokens).await?;
    for (token, report) in stream.tokens.iter().zip(stream.reports.iter()) {
        let total_latency_us = report
            .hops
            .iter()
            .map(|hop| hop.simulated_micros)
            .sum::<u64>();
        println!(
            "token={} text={} hops={} simulated_latency_us={} checksum_ok={}",
            token.token_position,
            token.text,
            report.hops.len(),
            total_latency_us,
            report.final_activation.verify_checksum()
        );
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct SimConfig {
    nodes: usize,
    layers: u32,
    tokens: u32,
    drop_node: Option<String>,
    corrupt_node: Option<String>,
}

impl SimConfig {
    fn from_env() -> Self {
        let mut config = Self {
            nodes: 8,
            layers: 16,
            tokens: 4,
            drop_node: None,
            corrupt_node: None,
        };

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--nodes" => config.nodes = parse_next(&mut args, "--nodes"),
                "--layers" => config.layers = parse_next(&mut args, "--layers"),
                "--tokens" => config.tokens = parse_next(&mut args, "--tokens"),
                "--drop-node" => config.drop_node = args.next(),
                "--corrupt-node" => config.corrupt_node = args.next(),
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
        "Usage: cargo run -p bitty-sim -- [--nodes N] [--layers N] [--tokens N] [--drop-node ID] [--corrupt-node ID]"
    );
}
