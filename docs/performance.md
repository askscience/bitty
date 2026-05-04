# Performance notes (evidence-driven)

## SIMD / CPU intrinsics

Bitty’s own Rust sources do not use `std::arch` SIMD. Real BitNet inference in this repo is delegated to **`oxbitnet` + `wgpu`** (`crates/bitty-bitnet-runtime`). Before adding AVX/NEON here, profile the path you care about:

1. GPU vs CPU-bound: if the GPU queue is saturated, intrinsics in Bitty will not help.
2. If a flamegraph points inside **`oxbitnet`**, the fix belongs in that dependency or its configuration—not random intrinsics in Bitty.

## Wire format

`BitNetLogits` now prefers **`logits_f32_le`** (raw little-endian `f32` bytes) over `repeated float logits` for large vectors. Compare encode throughput and payload size with:

```bash
cargo bench -p bitty-protocol --bench logits_wire
```

## Regression detection

Runtime metrics already exist (Halda duration, worker counters, executor histograms). Micro-benchmarks (e.g. the one above) complement metrics; they do not replace them. Add more `criterion` benches only for code you own and hot paths you measure.
