use bitty_worker::HardwareProfiler;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_hardware_profiler(c: &mut Criterion) {
    let profiler = HardwareProfiler::new("bench-node");

    c.benchmark_group("hardware_profiling")
        .sample_size(10)
        .bench_function("full_profile", |b| {
            b.iter(|| {
                let _ = black_box(profiler.profile());
            });
        });

    let profile = profiler.profile();
    c.bench_function("effective_compute_score_on_profile", |b| {
        b.iter(|| {
            let _ = black_box(profile.effective_compute_score());
        });
    });
    c.bench_function("memory_bytes_on_profile", |b| {
        b.iter(|| {
            let _ = black_box(profile.memory_bytes());
        });
    });
    c.bench_function("backend_type_on_profile", |b| {
        b.iter(|| {
            let _ = black_box(profile.backend_type());
        });
    });
}

criterion_group!(benches, bench_hardware_profiler);
criterion_main!(benches);
