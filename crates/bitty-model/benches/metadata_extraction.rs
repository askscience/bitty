use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_model::{ModelMetadata, classify_architecture};
use bitty_model::gguf::GgufMetadataValue;

mod common;

fn bench_classify_architecture(c: &mut Criterion) {
    let mut group = c.benchmark_group("classify_architecture");
    group.sample_size(200);

    group.bench_function("bitnet", |b| {
        b.iter(|| {
            let _ = black_box(classify_architecture(black_box("bitnet-25")));
        });
    });
    group.bench_function("llama", |b| {
        b.iter(|| {
            let _ = black_box(classify_architecture(black_box("llama")));
        });
    });
    group.bench_function("unknown", |b| {
        b.iter(|| {
            let _ = black_box(classify_architecture(black_box("custom-transformer-v2")));
        });
    });
    group.bench_function("all_14_architectures", |b| {
        b.iter(|| {
            for arch in common::ARCHITECTURES {
                let _ = black_box(classify_architecture(black_box(arch)));
            }
        });
    });
    group.finish();
}

fn bench_from_gguf(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_from_gguf");
    group.sample_size(50);

    let gguf_2 = common::make_small_gguf("llama", 2, 4096);
    group.bench_function("small_2layers", |b| {
        b.iter(|| {
            let _ = black_box(ModelMetadata::from_gguf(
                black_box(gguf_2.clone()),
                black_box(None),
            ));
        });
    });

    let gguf_30 = common::make_small_gguf("llama", 30, 4096);
    group.bench_function("medium_30layers", |b| {
        b.iter(|| {
            let _ = black_box(ModelMetadata::from_gguf(
                black_box(gguf_30.clone()),
                black_box(None),
            ));
        });
    });

    let gguf_80 = common::make_small_gguf("llama", 80, 4096);
    group.bench_function("large_80layers", |b| {
        b.iter(|| {
            let _ = black_box(ModelMetadata::from_gguf(
                black_box(gguf_80.clone()),
                black_box(None),
            ));
        });
    });
    group.finish();
}

fn bench_layer_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_metadata_scaling");
    group.sample_size(100);

    let gguf_30 = common::make_small_gguf("llama", 30, 4096);
    let meta_30 = ModelMetadata::from_gguf(gguf_30, None).expect("extract metadata");
    group.bench_function("30_layers", |b| {
        b.iter(|| {
            let _ = black_box(meta_30.layer_metadata());
        });
    });

    let gguf_80 = common::make_small_gguf("llama", 80, 4096);
    let meta_80 = ModelMetadata::from_gguf(gguf_80, None).expect("extract metadata");
    group.bench_function("80_layers", |b| {
        b.iter(|| {
            let _ = black_box(meta_80.layer_metadata());
        });
    });
    group.finish();
}

fn bench_architecture_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_architecture_variants");
    group.sample_size(50);

    let arch_configs: &[(&str, &[(&str, &str)])] = &[
        ("llama", &[("llama.embedding_length", "2048")]),
        ("mistral", &[("mistral.dim", "2048")]),
        ("phi-3", &[("phi.hidden_size", "2048")]),
        ("qwen2", &[("qwen2.embedding_length", "2048")]),
        ("gemma", &[("gemma.embedding_length", "2048")]),
        ("gemma2", &[("gemma.embedding_length", "2048")]),
        ("falcon", &[("falcon.hidden_size", "2048")]),
        ("stablelm-3b", &[("stablelm.embedding_length", "2048")]),
        ("deepseek", &[("deepseek.embedding_length", "2048")]),
        ("mamba", &[("mamba.embedding_length", "2048")]),
        ("bitnet-25", &[("bitnet.embedding_length", "2048")]),
        ("onebit-7b", &[("bitnet.embedding_length", "2048")]),
        ("custom-transformer", &[("llama.embedding_length", "2048")]),
        ("unknown-arch", &[("llama.embedding_length", "2048")]),
    ];

    for &(arch, keys) in arch_configs {
        group.bench_function(arch, |b| {
            let mut gguf = common::make_small_gguf(arch, 4, 2048);
            gguf.metadata.clear();
            gguf.metadata.insert(
                "general.architecture".into(),
                GgufMetadataValue::String(arch.to_string()),
            );
            for &(key, val) in keys {
                gguf.metadata.insert(
                    key.to_string(),
                    GgufMetadataValue::U64(val.parse().unwrap()),
                );
            }
            b.iter(|| {
                let _ = black_box(ModelMetadata::from_gguf(
                    black_box(gguf.clone()),
                    black_box(None),
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_classify_architecture,
    bench_from_gguf,
    bench_layer_metadata,
    bench_architecture_variants,
);
criterion_main!(benches);
