# Crate Reference

Bitty is organized as a Rust workspace with 11 crates.

## `bitty-protocol`
Shared types, protobuf definitions, serialization, endpoint resolution, Iroh transport layer. Defines `ActivationTensor`, `BitNetLogits`, `AssignedLayerRange`, codec traits, and security primitives (constant-time token comparison).

## `bitty-model`
GGUF file parsing (`gguf.rs`), model metadata extraction, i2s block decoding, layer ID classification, quantization type derivation, and `ModelSpec` registry loading.

## `bitty-inference`
Layer execution logic: `LayerExecutor` trait and `FakeLayerExecutor` for testing. Handles the compile-time detection of which backend (candle / bitnet / cpu) to use.

## `bitty-bitnet-runtime`
BitNet inference engine with GPU path (`BitNetRuntime` using candle-core) and CPU backend (`cpu_backend/`) with quantized matmul kernels (Q4_K, Q6_K, Q8_0, F32, F16), RoPE, GQA attention, and Mamba SSM support.

## `bitty-candle-runtime`
Candle-core GPU inference: GGUF loading, `CandleModel` with `TransformerBlock`, `Attention` (RoPE, KV cache, GQA), `FFN` (SiLU gated), quantized dequantization. Models can run on CUDA, Metal, or ROCm.

## `bitty-wgpu-runtime`
Cross-GPU wgpu backend (Vulkan, Metal, DX12, WebGPU). `WgpuDevice` for adapter enumeration, `WgpuModel` for GGUF loading and inference using WGSL compute shaders (rmsnorm, embedding, Q4_K/Q8_0/F32 matmul, swiglu).

## `bitty-coordinator`
Cluster scheduler (`Halda`): tier-aware topology computation, layer-to-worker assignment, request routing, batching. Includes gRPC server and Iroh P2P endpoint for cluster management.

## `bitty-worker`
Worker node: hardware profiling (CPU/GPU detection via sysinfo + nvml + wgpu), model shard loading, activation forwarding in ring topology, gRPC + Iroh endpoints.

## `bitty-sim`
Simulator for distributed execution: spawns a virtual cluster in a single process, simulates ring passing with chaos testing (node drops), measures throughput and latency. Used for Halda coverage CI.

## `bitty-cli`
CLI binary (`bitty`): model management, inference dispatch (auto-detect CPU/GPU/Cluster), interactive chat, chat template handling (Jinja via minijinja), settings management, Modelfile support, server mode.

## `bitty-observability`
Prometheus metrics, tracing (OpenTelemetry), logging. Provides `install_prometheus_recorder()`, `record_halda_run()`, `init_tracing()`.
