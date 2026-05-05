use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_protocol::{
    ActivationDType, ActivationTensor, logits_f32_le_bytes, logits_from_f32_le_bytes,
};
use prost::Message;

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

fn bench_checksum(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_checksum");
    group.sample_size(100);

    let payload_4k = vec![7u8; 4096];
    group.bench_function("new_4KB_payload", |b| {
        b.iter(|| {
            let _ = black_box(ActivationTensor::new(
                "req",
                0,
                0,
                1,
                vec![2048, 1],
                ActivationDType::Fp16,
                black_box(payload_4k.clone()),
            ));
        });
    });

    let payload_64k = vec![7u8; 64 * 1024];
    group.bench_function("new_64KB_payload", |b| {
        b.iter(|| {
            let _ = black_box(ActivationTensor::new(
                "req",
                0,
                0,
                1,
                vec![32768, 1],
                ActivationDType::Fp16,
                black_box(payload_64k.clone()),
            ));
        });
    });

    let payload_1m = vec![7u8; 1024 * 1024];
    group.bench_function("new_1MB_payload", |b| {
        b.iter(|| {
            let _ = black_box(ActivationTensor::new(
                "req",
                0,
                0,
                1,
                vec![524288, 1],
                ActivationDType::Fp16,
                black_box(payload_1m.clone()),
            ));
        });
    });

    let activation_64k = make_activation(64 * 1024);
    group.bench_function("verify_checksum_64KB", |b| {
        b.iter(|| {
            let _ = black_box(activation_64k.verify_checksum());
        });
    });

    let activation_1m = make_activation(1024 * 1024);
    group.bench_function("verify_checksum_1MB", |b| {
        b.iter(|| {
            let _ = black_box(activation_1m.verify_checksum());
        });
    });
    group.finish();
}

fn bench_logits_codec(c: &mut Criterion) {
    let mut group = c.benchmark_group("logits_codec");
    group.sample_size(100);

    for &size in &[8192usize, 32000, 128000] {
        let logits: Vec<f32> = (0..size)
            .map(|i| f32::NEG_INFINITY + (i as f32 * 0.001))
            .collect();
        let encoded = logits_f32_le_bytes(&logits);

        group.bench_function(format!("encode_{size}_floats").as_str(), |b| {
            b.iter(|| {
                let _ = black_box(logits_f32_le_bytes(black_box(&logits)));
            });
        });
        group.bench_function(format!("decode_{size}_floats").as_str(), |b| {
            b.iter(|| {
                let _ = black_box(logits_from_f32_le_bytes(black_box(&encoded)));
            });
        });
    }
    group.finish();
}

fn bench_activation_wire_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_wire");
    group.sample_size(100);

    for &size_kb in &[4usize, 64, 1024] {
        let activation = make_activation(size_kb * 1024);

        let p: bitty_protocol::pb::ActivationTensor = (&activation).into();
        let buf = prost::Message::encode_to_vec(&p);

        group.bench_function(format!("to_proto_{size_kb}KB").as_str(), |b| {
            b.iter(|| {
                let _: bitty_protocol::pb::ActivationTensor =
                    black_box(&activation ).into();
            });
        });

        let p2: bitty_protocol::pb::ActivationTensor = (&activation).into();
        group.bench_function(format!("encode_proto_{size_kb}KB").as_str(), |b| {
            b.iter(|| {
                let _ = black_box(prost::Message::encode_to_vec(black_box(&p2)));
            });
        });

        group.bench_function(format!("decode_proto_{size_kb}KB").as_str(), |b| {
            b.iter(|| {
                let _ = bitty_protocol::pb::ActivationTensor::decode(
                    black_box(buf.as_slice()),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_checksum, bench_logits_codec, bench_activation_wire_roundtrip);
criterion_main!(benches);
