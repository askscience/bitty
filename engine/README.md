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
- Tensors outside the range are also not required to exist, so a stage can load a
  file that holds only its own tensors (see "Stage files" below).
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
cmake --build build --target llama-stage-check bitty-stage
build/bin/llama-stage-check -m model.gguf -c 8,16 -n 16
```

The patch adds two programs:

- `bitty-stage` is the stage server. A bitty worker starts it as a child process
  for its layer range and talks to it over stdin/stdout (frame format at the top
  of `examples/stage-server/stage-server.cpp`). It keeps the KV cache and the
  recurrent state of each request, evicting the least recently used one when it
  runs out of slots. Bitty finds it through `BITTY_STAGE_BIN`, next to its own
  binary, or on `PATH`.
- `llama-stage-check` is the correctness check described below.

The Rust side has an end-to-end test that chains three stage processes and
compares them with the whole model:

```sh
BITTY_STAGE_BIN=$PWD/build/bin/bitty-stage BITTY_TEST_MODEL=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
    cargo test -p bitty-inference --test stage_engine -- --nocapture
```

`llama-stage-check` runs the whole model as a reference, then the same prompt and
greedy generation through a chain of stages cut at the given layers, and compares
the logits at every position.

## Stage files

A peer does not need the whole model on disk. A stage file is a normal GGUF with
the full metadata (hyperparameters, tokenizer) and only the tensors of one layer
range, plus the keys `bitty.stage.layer_start` / `bitty.stage.layer_end`:

```sh
bitty pull qwen3.5-35b-a3b --layers 10..20      # END is not included
bitty pull /path/to/model.gguf --layers 0..10 --output stage-0-10.gguf
bitty pull https://host/model.gguf --layers 30..40
```

When the whole model is already on disk it is used as the source. Otherwise bitty
reads the header of the published file and then only the byte ranges of the
stage's tensors, with HTTP range requests through `curl`; neighbouring tensors
are fetched as one request. The code is in `crates/bitty-model/src/stage_file.rs`.

The stage process is started on the stage file exactly as on the whole model,
with the same `--start` / `--end`. `llama-stage-check -s a.gguf,b.gguf,...` runs
the chain with one file per stage and compares it with the whole model given by
`-m`.

Limits: the downloaded ranges are protected only by TLS, there is no hash check
of a stage file against the published model yet, and an interrupted pull starts
again from the beginning.

## Verified so far

Qwen3.5-0.8B Q4_K_M (dense `qwen35`, 24 layers), CPU: logits are bit-identical
to the whole model (max difference 0) for cuts `8,16`, `12`, `3,4,13,23`,
`1,2,5,12,20` and for one layer per stage (24 stages). This covers stages that
hold only recurrent layers and stages that hold only an attention layer.

Weight buffers on that model: whole model 497 MiB; stage `[0,8)` 298 MiB,
`[8,16)` 99 MiB, `[16,24)` 299 MiB. The first and last stages carry the 199 MiB
vocabulary matrix, which dominates such a small model.

Qwen3.5-35B-A3B Q4_K_M (`qwen35moe`, 40 layers, the real target), CPU: cuts
`10,20,30` give logits bit-identical to the whole model. Weight buffers: whole
model 20986 MiB; the four stages 5537, 5015, 5021 and 5413 MiB.

Stage files: on the 0.8B model the three files for cuts `8,16` are 309, 109 and
310 MiB (whole file 508 MiB) and give bit-identical logits; the middle one pulled
from HuggingFace with range requests is byte-identical to the one built from the
local copy. On the 35B model the four files for cuts `10,20,30` are 5547, 5025,
5032 and 5423 MiB (whole file 20996 MiB): the chain running on
those four files gives logits bit-identical to the whole model.

## Not done yet

- The coordinator still turns a text prompt into bytes instead of tokens; the
  stage server can tokenize (`StageExecutor::tokenize`) but the cluster path does
  not call it yet. Chat templates and sampling other than greedy are missing too.

- GPU offload: `n_gpu_layers` is not rebased to the range.
- Workers do not fetch their stage file on their own when they get a layer
  assignment; `bitty pull --layers` is a manual step for now.
- State save/load does not record the range.
- `llama_model_size` still reports the whole file for a stage loaded from it.
