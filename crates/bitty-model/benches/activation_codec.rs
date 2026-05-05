use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_model::activation_codec::{ActivationCodec, CodecKind};
use bitty_protocol::{ActivationDType, ActivationTensor};

fn make_activation(payload_len: usize) -> ActivationTensor {
    let payload: Vec<u8> = (0..payload_len)
        .map(|i| (i as u8).wrapping_mul(13).wrapping_add(1))
        .collect();
    ActivationTensor::new(
        "bench-req",
        0,
        0,
        1,
        vec![payload_len as u32 / 2, 1],
        ActivationDType::Fp16,
        payload,
    )
}

fn bench_fp8_codec(c: &mut Criterion) {
    let codec = ActivationCodec::new(CodecKind::Fp8Linear);
    let mut group = c.benchmark_group("fp8_codec");
    group.sample_size(100);

    for &size_kb in &[4usize, 64, 1024] {
        let activation = make_activation(size_kb * 1024);
        let encoded = codec.encode(&activation).expect("fp8 encode");

        group.bench_function(format!("encode_{size_kb}KB").as_str(), |b| {
            b.iter(|| {
                let _ = black_box(codec.encode(black_box(&activation)));
            });
        });
        group.bench_function(format!("decode_{size_kb}KB").as_str(), |b| {
            b.iter(|| {
                let _ = black_box(codec.decode(black_box(&encoded)));
            });
        });
    }
    group.finish();
}

fn bench_sparse_topk_codec(c: &mut Criterion) {
    let codec = ActivationCodec::new(CodecKind::SparseTopK);
    let mut group = c.benchmark_group("sparse_topk_codec");
    group.sample_size(100);

    for &size_kb in &[4usize, 64, 256] {
        let activation = make_activation(size_kb * 1024);
        let encoded = codec.encode(&activation).expect("topk encode");

        group.bench_function(format!("encode_{size_kb}KB").as_str(), |b| {
            b.iter(|| {
                let _ = black_box(codec.encode(black_box(&activation)));
            });
        });
        group.bench_function(format!("decode_{size_kb}KB").as_str(), |b| {
            b.iter(|| {
                let _ = black_box(codec.decode(black_box(&encoded)));
            });
        });
    }
    group.finish();
}

fn bench_raw_codec(c: &mut Criterion) {
    let codec = ActivationCodec::new(CodecKind::Raw);
    let mut group = c.benchmark_group("raw_codec");
    group.sample_size(100);

    let activation_64k = make_activation(64 * 1024);
    group.bench_function("clone_64KB", |b| {
        b.iter(|| {
            let _ = black_box(codec.encode(black_box(&activation_64k)));
        });
    });

    let activation_1m = make_activation(1024 * 1024);
    group.bench_function("clone_1MB", |b| {
        b.iter(|| {
            let _ = black_box(codec.encode(black_box(&activation_1m)));
        });
    });
    group.finish();
}

fn bench_codec_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_roundtrip");
    group.sample_size(100);

    let codec_fp8 = ActivationCodec::new(CodecKind::Fp8Linear);
    let activation = make_activation(64 * 1024);
    group.bench_function("fp8_64KB_roundtrip", |b| {
        b.iter(|| {
            let encoded = codec_fp8.encode(black_box(&activation)).expect("encode");
            let _ = black_box(codec_fp8.decode(&encoded).expect("decode"));
        });
    });

    let codec_sparse = ActivationCodec::new(CodecKind::SparseTopK);
    group.bench_function("sparse_topk_64KB_roundtrip", |b| {
        b.iter(|| {
            let encoded = codec_sparse.encode(black_box(&activation)).expect("encode");
            let _ = black_box(codec_sparse.decode(&encoded).expect("decode"));
        });
    });

    let codec_raw = ActivationCodec::new(CodecKind::Raw);
    group.bench_function("raw_64KB_roundtrip", |b| {
        b.iter(|| {
            let encoded = codec_raw.encode(black_box(&activation)).expect("encode");
            let _ = black_box(codec_raw.decode(&encoded).expect("decode"));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_fp8_codec, bench_sparse_topk_codec, bench_raw_codec, bench_codec_roundtrip);
criterion_main!(benches);
