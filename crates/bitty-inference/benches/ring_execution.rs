use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_inference::{FakeLayerExecutor, LayerExecutor};
use bitty_protocol::{ActivationDType, ActivationTensor, AssignedLayerRange, Quantization};
use tokio::runtime::Runtime;

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

fn bench_fake_executor_forward(c: &mut Criterion) {
    let executor = FakeLayerExecutor;
    let rt = Runtime::new().expect("tokio runtime");

    let mut group = c.benchmark_group("fake_executor_forward");
    group.sample_size(100);

    for &size_kb in &[4usize, 64, 256] {
        let activation = make_activation(size_kb * 1024);

        for &num_layers in &[1u32, 4, 30] {
            let range = AssignedLayerRange {
                start_layer: 0,
                end_layer_exclusive: num_layers,
                quantization: Quantization::Bit1,
            };

            group.bench_function(
                format!("{num_layers}layers_{size_kb}KB").as_str(),
                |b| {
                    b.iter(|| {
                        rt.block_on(async {
                            let _ = black_box(
                                executor
                                    .execute_range(
                                        black_box(&range),
                                        black_box(activation.clone()),
                                    )
                                    .await,
                            );
                        });
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_fake_executor_logits(c: &mut Criterion) {
    let executor = FakeLayerExecutor;
    let rt = Runtime::new().expect("tokio runtime");

    let mut group = c.benchmark_group("fake_executor_logits");
    group.sample_size(100);

    let activation_4k = make_activation(4 * 1024);
    group.bench_function("4KB_activation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(
                    executor
                        .final_logits(black_box(activation_4k.clone()))
                        .await,
                );
            });
        });
    });

    let activation_64k = make_activation(64 * 1024);
    group.bench_function("64KB_activation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(
                    executor
                        .final_logits(black_box(activation_64k.clone()))
                        .await,
                );
            });
        });
    });
    group.finish();
}

fn bench_layer_executor_dispatch(c: &mut Criterion) {
    let executor = FakeLayerExecutor;
    let rt = Runtime::new().expect("tokio runtime");

    let mut group = c.benchmark_group("executor_dispatch");
    group.sample_size(200);

    let activation = make_activation(4 * 1024);
    let range = AssignedLayerRange {
        start_layer: 0,
        end_layer_exclusive: 1,
        quantization: Quantization::Bit1,
    };
    group.bench_function("single_layer_4KB", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(
                    executor
                        .execute_range(black_box(&range), black_box(activation.clone()))
                        .await,
                );
            });
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_fake_executor_forward,
    bench_fake_executor_logits,
    bench_layer_executor_dispatch,
);
criterion_main!(benches);
