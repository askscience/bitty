# Stage engine (patched llama.cpp)

Bitty splits Qwen3.5-35B-A3B into contiguous layer ranges ("stages") that run on
different peers. The compute engine is llama.cpp with a small patch that lets one
process load and run only a layer range.

- Upstream: https://github.com/ggml-org/llama.cpp
- Pinned commit: `45c8ea837489be1304500dd1d298f2a671c985d4`
- Patch: `llama-cpp-stage.patch`

## What the patch does

- `llama_model_params.stage_layer_start` / `stage_layer_end` select the range
  `[start, end)`. `end == 0` keeps the normal whole-model behaviour.
- Tensors of blocks outside the range are skipped at load time. The token
  embedding is loaded only by the first stage (and by the last one when the
  output head is tied to it); the final norm and the LM head only by the last.
- The first stage takes tokens. Later stages take hidden states through
  `llama_batch.embd`, with 4 positions per token laid out section-major
  (`pos[s * n_tokens + i]`, text tokens are `[p, p, p, 0]`).
- Non-final stages return the raw residual stream through
  `llama_get_embeddings` (context created with `embeddings = true` and
  `LLAMA_POOLING_TYPE_NONE`). The final stage returns logits as usual.
- KV cache and recurrent (Gated DeltaNet) state are allocated only for the
  layers in the range.
- Only the `qwen35` and `qwen35moe` architectures are supported; other
  architectures refuse a stage range.

## Build and check

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 45c8ea837489be1304500dd1d298f2a671c985d4
git apply ../llama-cpp-stage.patch
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON \
      -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-stage-check
build/bin/llama-stage-check -m model.gguf -c 8,16 -n 16
```

`llama-stage-check` runs the whole model as a reference, then the same prompt and
greedy generation through a chain of stages cut at the given layers, and compares
the logits at every position.

## Verified so far

Qwen3.5-0.8B Q4_K_M (dense `qwen35`, 24 layers), CPU: logits are bit-identical
to the whole model (max difference 0) for cuts `8,16`, `12`, `3,4,13,23`,
`1,2,5,12,20` and for one layer per stage (24 stages). This covers stages that
hold only recurrent layers and stages that hold only an attention layer.

Weight buffers on that model: whole model 497 MiB; stage `[0,8)` 298 MiB,
`[8,16)` 99 MiB, `[16,24)` 299 MiB. The first and last stages carry the 199 MiB
vocabulary matrix, which dominates such a small model.

## Not done yet

- `qwen35moe` (the real 35B-A3B target) carries the same patch but has not been
  run: no small MoE model of this family exists.
- GPU offload: `n_gpu_layers` is not rebased to the range.
- A stage still needs the whole GGUF file on disk; per-stage files are future work.
- State save/load does not record the range.
- `llama_model_size` (and the "MiB of weights" line of the check tool) still
  reports the whole file.
