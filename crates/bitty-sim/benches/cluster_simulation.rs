use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_protocol::{
    HardwareProfile, LayerMetadata, NodeId, NodeTier,
};
use bitty_sim::SimulatedCluster;
use tokio::runtime::Runtime;

fn make_profiles(count: usize) -> Vec<HardwareProfile> {
    (0..count)
        .map(|index| {
            let tier = match index {
                0 => NodeTier::S,
                1 | 2 => NodeTier::A,
                3..=5 => NodeTier::B,
                _ => NodeTier::C,
            };
            let gpu_tflops = match tier {
                NodeTier::S => 30.0,
                NodeTier::A => 12.0,
                NodeTier::B => 3.0,
                _ => 0.0,
            };
            HardwareProfile {
                node_id: NodeId::new(format!("sim-{index}")),
                cpu_tflops: 0.5 + index as f64 * 0.1,
                gpu_tflops,
                memory_gb: 4.0 + index as f64 * 0.5,
                memory_bandwidth_gbps: 20.0 + index as f64 * 2.0,
                disk_bandwidth_mbps: 400.0,
                network_rtt_ms: 5.0 + index as f64 * 0.5,
                uplink_mbps: 100.0,
                os: "linux".into(),
                tier,
                ram_mb: 4096 + index as u64 * 1024,
                vram_mb: if gpu_tflops > 0.0 { 8192 } else { 0 },
                architecture: "x86_64".into(),
                gpus: Vec::new(),
                os_reclaim_score: 0.0,
                worker_endpoint: format!("sim-{index}:50051"),
                model_path: String::new(),
                backend_type: if gpu_tflops > 0.0 { "gpu" } else { "cpu" }.into(),
                layer_eligible: true,
                max_layers: u32::MAX,
            }
        })
        .collect()
}

fn make_layers(count: u32) -> Vec<LayerMetadata> {
    (0..count)
        .map(|layer_id| LayerMetadata {
            layer_id,
            weight_bytes: 512_000 + layer_id as u64 * 128_000,
            activation_bytes: 4096,
            estimated_flops: 1e9,
            precision_critical: layer_id == 0 || layer_id + 1 == count,
        })
        .collect()
}

fn bench_cluster_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("cluster_build");
    group.sample_size(30);

    let profiles_4 = make_profiles(4);
    let layers_30 = make_layers(30);
    group.bench_function("4nodes_30layers", |b| {
        b.iter(|| {
            let _ = black_box(
                SimulatedCluster::build(
                    black_box(profiles_4.clone()),
                    black_box(layers_30.clone()),
                ),
            );
        });
    });

    let profiles_8 = make_profiles(8);
    let layers_30b = make_layers(30);
    group.bench_function("8nodes_30layers", |b| {
        b.iter(|| {
            let _ = black_box(
                SimulatedCluster::build(
                    black_box(profiles_8.clone()),
                    black_box(layers_30b.clone()),
                ),
            );
        });
    });

    let profiles_16 = make_profiles(16);
    let layers_80 = make_layers(80);
    group.bench_function("16nodes_80layers", |b| {
        b.iter(|| {
            let _ = black_box(
                SimulatedCluster::build(
                    black_box(profiles_16.clone()),
                    black_box(layers_80.clone()),
                ),
            );
        });
    });
    group.finish();
}

fn bench_token_streaming(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");

    let mut group = c.benchmark_group("token_streaming");
    group.sample_size(30);

    let profiles_4 = make_profiles(4);
    let layers_30 = make_layers(30);

    group.bench_function("4nodes_30layers_100tokens", |b| {
        let cluster =
            SimulatedCluster::build(profiles_4.clone(), layers_30.clone()).expect("build cluster");
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(
                    cluster.stream_tokens("bench", 100),
                );
            });
        });
    });

    group.bench_function("4nodes_30layers_20tokens", |b| {
        let cluster =
            SimulatedCluster::build(profiles_4.clone(), layers_30.clone()).expect("build cluster");
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(
                    cluster.stream_tokens("bench", 20),
                );
            });
        });
    });
    group.finish();
}

criterion_group!(benches, bench_cluster_build, bench_token_streaming);
criterion_main!(benches);
