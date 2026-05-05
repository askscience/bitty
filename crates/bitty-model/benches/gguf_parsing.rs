use criterion::{black_box, criterion_group, criterion_main, Criterion};

mod common;

fn bench_parse_gguf_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("gguf_parsing");
    group.sample_size(50);

    let gguf_2 = common::make_small_gguf("llama", 2, 4096);
    let bytes_2 = common::serialize_gguf(&gguf_2);
    group.bench_function("parse_small_2layers", |b| {
        b.iter(|| {
            let _ = black_box(bitty_model::gguf::parse_gguf_bytes(black_box(
                &bytes_2,
            )));
        });
    });

    let gguf_30 = common::make_small_gguf("llama", 30, 4096);
    let bytes_30 = common::serialize_gguf(&gguf_30);
    group.bench_function("parse_medium_30layers", |b| {
        b.iter(|| {
            let _ = black_box(bitty_model::gguf::parse_gguf_bytes(black_box(
                &bytes_30,
            )));
        });
    });

    let gguf_80 = common::make_small_gguf("llama", 80, 4096);
    let bytes_80 = common::serialize_gguf(&gguf_80);
    group.bench_function("parse_large_80layers", |b| {
        b.iter(|| {
            let _ = black_box(bitty_model::gguf::parse_gguf_bytes(black_box(
                &bytes_80,
            )));
        });
    });
    group.finish();
}

fn bench_parse_many_tensors(c: &mut Criterion) {
    let mut group = c.benchmark_group("gguf_many_tensors");
    group.sample_size(30);

    let gguf_100 = common::make_many_tensors(10, 10);
    let bytes_100 = common::serialize_gguf(&gguf_100);
    group.bench_function("100_tensors", |b| {
        b.iter(|| {
            let _ = black_box(bitty_model::gguf::parse_gguf_bytes(black_box(
                &bytes_100,
            )));
        });
    });

    let gguf_1k = common::make_many_tensors(100, 10);
    let bytes_1k = common::serialize_gguf(&gguf_1k);
    group.bench_function("1000_tensors", |b| {
        b.iter(|| {
            let _ = black_box(bitty_model::gguf::parse_gguf_bytes(black_box(
                &bytes_1k,
            )));
        });
    });
    group.finish();
}

fn bench_parse_many_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("gguf_many_metadata");
    group.sample_size(50);

    let mut gguf_20 = common::make_small_gguf("llama", 4, 4096);
    for i in 0..20 {
        gguf_20.metadata.insert(format!("extra.key.{i}"), bitty_model::gguf::GgufMetadataValue::U64(i));
    }
    let bytes_20 = common::serialize_gguf(&gguf_20);
    group.bench_function("20_metadata_keys", |b| {
        b.iter(|| {
            let _ = black_box(bitty_model::gguf::parse_gguf_bytes(black_box(
                &bytes_20,
            )));
        });
    });

    let mut gguf_100 = common::make_small_gguf("llama", 4, 4096);
    for i in 0..100 {
        gguf_100.metadata.insert(format!("extra.key.{i}"), bitty_model::gguf::GgufMetadataValue::U64(i));
    }
    let bytes_100 = common::serialize_gguf(&gguf_100);
    group.bench_function("100_metadata_keys", |b| {
        b.iter(|| {
            let _ = black_box(bitty_model::gguf::parse_gguf_bytes(black_box(
                &bytes_100,
            )));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_parse_gguf_bytes,
    bench_parse_many_tensors,
    bench_parse_many_metadata,
);
criterion_main!(benches);
