use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitty_model::tensor::{LowBitTensor, TensorShape};
use bitty_protocol::Quantization;

fn bench_packed_len_for(c: &mut Criterion) {
    let sizes: &[usize] = &[1024, 65536, 1048576];

    let mut group = c.benchmark_group("packed_len_for");
    group.sample_size(200);

    group.bench_function("fp16_1M", |b| {
        b.iter(|| {
            let _ = black_box(LowBitTensor::packed_len_for(
                black_box(1048576),
                black_box(Quantization::Fp16),
            ));
        });
    });

    group.bench_function("all_quantizations_64K", |b| {
        let all: &[Quantization] = &[
            Quantization::F32,
            Quantization::Fp16,
            Quantization::Q8,
            Quantization::Q6,
            Quantization::Q5,
            Quantization::Q4,
            Quantization::Q3,
            Quantization::Q2,
            Quantization::Bit1,
        ];
        b.iter(|| {
            for &q in all {
                let _ = black_box(LowBitTensor::packed_len_for(
                    black_box(65536),
                    black_box(q),
                ));
            }
        });
    });

    for &size in sizes {
        for quant in &[Quantization::F32, Quantization::Fp16, Quantization::Q4, Quantization::Bit1] {
            let name = format!("{}_{}", quant.as_str(), size);
            group.bench_function(name.as_str(), |b| {
                b.iter(|| {
                    let _ = black_box(LowBitTensor::packed_len_for(
                        black_box(size),
                        black_box(*quant),
                    ));
                });
            });
        }
    }
    group.finish();
}

fn bench_validate_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_len");
    group.sample_size(200);

    let small = LowBitTensor {
        shape: TensorShape(vec![1024]),
        quantization: Quantization::Fp16,
        packed_weights: vec![0u8; 2048],
    };
    group.bench_function("small_1024_elements_fp16", |b| {
        b.iter(|| {
            let _ = black_box(small.validate_len());
        });
    });

    let medium = LowBitTensor {
        shape: TensorShape(vec![65536]),
        quantization: Quantization::Q4,
        packed_weights: vec![0u8; 32768],
    };
    group.bench_function("medium_64K_elements_q4", |b| {
        b.iter(|| {
            let _ = black_box(medium.validate_len());
        });
    });

    let large = LowBitTensor {
        shape: TensorShape(vec![1048576]),
        quantization: Quantization::Bit1,
        packed_weights: vec![0u8; 131072],
    };
    group.bench_function("large_1M_elements_bit1", |b| {
        b.iter(|| {
            let _ = black_box(large.validate_len());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_packed_len_for, bench_validate_len);
criterion_main!(benches);
