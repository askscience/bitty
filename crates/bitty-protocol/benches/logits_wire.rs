//! Compare protobuf encode size and throughput for legacy `repeated float logits` vs `logits_f32_le`.
use bitty_protocol::logits_codec::logits_f32_le_bytes;
use bitty_protocol::pb::BitNetLogits;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prost::Message;

fn sample_logits(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32) * 1e-6 - 0.5).collect()
}

fn encode_repeated(logits: &[f32]) -> Vec<u8> {
    let msg = BitNetLogits {
        request_id: "bench".into(),
        token_position: 0,
        logits: logits.to_vec(),
        crc32: 0,
        logits_f32_le: Vec::new(),
    };
    msg.encode_to_vec()
}

fn encode_f32_le(logits: &[f32]) -> Vec<u8> {
    let msg = BitNetLogits {
        request_id: "bench".into(),
        token_position: 0,
        logits: Vec::new(),
        crc32: 0,
        logits_f32_le: logits_f32_le_bytes(logits),
    };
    msg.encode_to_vec()
}

fn bench_logits_encode(c: &mut Criterion) {
    let n = 8_192;
    let logits = sample_logits(n);
    let repeated_len = encode_repeated(&logits).len();
    let packed_len = encode_f32_le(&logits).len();
    println!("BitNetLogits encode size n={n}: repeated={repeated_len} bytes, f32_le={packed_len} bytes");

    c.bench_function("bitnet_logits_encode_repeated_f32", |b| {
        b.iter(|| black_box(encode_repeated(&logits)))
    });
    c.bench_function("bitnet_logits_encode_logits_f32_le", |b| {
        b.iter(|| black_box(encode_f32_le(&logits)))
    });
}

criterion_group!(benches, bench_logits_encode);
criterion_main!(benches);
