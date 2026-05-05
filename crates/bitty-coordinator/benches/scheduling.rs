use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_coordinator::{Halda, SchedulerConfig};
use bitty_protocol::{HardwareProfile, LayerMetadata, NodeId, NodeTier};

fn make_profiles(count: usize) -> Vec<HardwareProfile> {
    (0..count)
        .map(|index| {
            let tier = match index {
                0 => NodeTier::S,
                1 | 2 => NodeTier::A,
                3..=5 => NodeTier::B,
                6..=8 => NodeTier::C,
                _ => NodeTier::D,
            };
            let gpu_tflops = match tier {
                NodeTier::S => 30.0,
                NodeTier::A => 12.0,
                NodeTier::B => 3.0,
                _ => 0.0,
            };
            HardwareProfile {
                node_id: NodeId::new(format!("node-{index}")),
                cpu_tflops: 0.5 + index as f64 * 0.1,
                gpu_tflops,
                memory_gb: 4.0 + index as f64 * 0.5,
                memory_bandwidth_gbps: 20.0 + index as f64 * 2.0,
                disk_bandwidth_mbps: 400.0,
                network_rtt_ms: 5.0 + index as f64 * 0.5,
                uplink_mbps: 100.0 - index as f64,
                os: "linux".into(),
                tier,
                ram_mb: 4096 + index as u64 * 1024,
                vram_mb: if gpu_tflops > 0.0 { 8192 } else { 0 },
                architecture: "x86_64".into(),
                gpus: Vec::new(),
                os_reclaim_score: 0.0,
                worker_endpoint: format!("node-{index}:50051"),
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
            estimated_flops: 1e9 + layer_id as f64 * 1e7,
            precision_critical: layer_id == 0 || layer_id + 1 == count,
        })
        .collect()
}

fn bench_halda_assign(c: &mut Criterion) {
    let halda = Halda::new(SchedulerConfig::default());

    c.benchmark_group("halda_assign")
        .sample_size(50)
        .bench_function("4nodes_30layers", |b| {
            let profiles = make_profiles(4);
            let layers = make_layers(30);
            b.iter(|| {
                let _ = black_box(halda.assign(
                    black_box(&profiles),
                    black_box(&layers),
                ));
            });
        })
        .bench_function("16nodes_30layers", |b| {
            let profiles = make_profiles(16);
            let layers = make_layers(30);
            b.iter(|| {
                let _ = black_box(halda.assign(
                    black_box(&profiles),
                    black_box(&layers),
                ));
            });
        })
        .bench_function("64nodes_80layers", |b| {
            let profiles = make_profiles(64);
            let layers = make_layers(80);
            b.iter(|| {
                let _ = black_box(halda.assign(
                    black_box(&profiles),
                    black_box(&layers),
                ));
            });
        })
        .bench_function("256nodes_120layers", |b| {
            let profiles = make_profiles(256);
            let layers = make_layers(120);
            b.iter(|| {
                let _ = black_box(halda.assign(
                    black_box(&profiles),
                    black_box(&layers),
                ));
            });
        });
}

fn bench_effective_compute_score(c: &mut Criterion) {
    c.benchmark_group("effective_compute_score")
        .sample_size(200)
        .bench_function("single_node", |b| {
            let profile = &make_profiles(1)[0];
            b.iter(|| {
                let _ = black_box(profile.effective_compute_score());
            });
        })
        .bench_function("bulk_100_nodes", |b| {
            let profiles = make_profiles(100);
            b.iter(|| {
                for profile in black_box(&profiles) {
                    let _ = black_box(profile.effective_compute_score());
                }
            });
        })
        .bench_function("bulk_1000_nodes", |b| {
            let profiles = make_profiles(1000);
            b.iter(|| {
                for profile in black_box(&profiles) {
                    let _ = black_box(profile.effective_compute_score());
                }
            });
        });
}

criterion_group!(benches, bench_halda_assign, bench_effective_compute_score);
criterion_main!(benches);
