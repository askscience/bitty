use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_model::ModelMetadata;
use bitty_protocol::{AssignedLayerRange, LayerAssignment, ModelStage, NodeId, Quantization};

mod common;

fn make_assignment(node_id: &str, start: u32, end: u32) -> LayerAssignment {
    LayerAssignment {
        node_id: NodeId::new(node_id),
        range: AssignedLayerRange {
            start_layer: start,
            end_layer_exclusive: end,
            quantization: Quantization::Q2,
        },
        assigned_weight_bytes: 0,
        expected_latency_ms: 0.0,
        next_node_id: None,
        disk_offload_fraction: 0.0,
        model_stage: ModelStage::LayerRange,
    }
}

fn bench_shard_plan(c: &mut Criterion) {
    let mut group = c.benchmark_group("shard_plan");
    group.sample_size(100);

    for &num_layers in &[4u32, 30, 80] {
        let gguf = common::make_small_gguf("llama", num_layers, 4096);
        let metadata =
            ModelMetadata::from_gguf(gguf, None).expect("extract metadata");

        for &num_nodes in &[1u32, 4, 8] {
            let chunk = num_layers / num_nodes;
            let assignments: Vec<LayerAssignment> = (0..num_nodes)
                .map(|i| {
                    let start = i * chunk;
                    let end = if i + 1 == num_nodes { num_layers } else { (i + 1) * chunk };
                    make_assignment(&format!("node-{i}"), start, end)
                })
                .collect();

            let name = format!("{num_layers}layers_{num_nodes}nodes");
            group.bench_function(name.as_str(), |b| {
                b.iter(|| {
                    for assignment in black_box(&assignments) {
                        let _ = black_box(metadata.shard_plan(assignment));
                    }
                });
            });
        }
    }
    group.finish();
}

fn bench_layer_metadata_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_metadata_computation");
    group.sample_size(100);

    for &num_layers in &[10u32, 30, 60, 120] {
        let gguf = common::make_small_gguf("llama", num_layers, 4096);
        let metadata = ModelMetadata::from_gguf(gguf, None).expect("extract metadata");

        let name = format!("{num_layers}_layers");
        group.bench_function(name.as_str(), |b| {
            b.iter(|| {
                let _ = black_box(metadata.layer_metadata());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_shard_plan, bench_layer_metadata_scaling);
criterion_main!(benches);
