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

A node does this by itself. Give `bitty node --model` a registry name or a URL
of a model that is not on disk: the node fetches the vocabulary file (the range
`0..0`: all the metadata and no weights, about 10 MB) to start, and the stage
file of its layers when the coordinator hands them over, which happens on the
first request that uses the node. That request waits for the download. A node
that holds the whole model keeps opening it directly. When the cluster changes
and a node gets other layers it fetches the new stage file; the old one stays on
disk.

Limits: the downloaded ranges are protected only by TLS, there is no hash check
of a stage file against the published model yet, and an interrupted pull starts
again from the beginning.

## GPU and threads

Build llama.cpp with `-DGGML_VULKAN=ON` (or another GPU backend) and set on the
node:

- `BITTY_GPU_LAYERS=N`: keep N layers of the node's range on the GPU, counted
  from the top of the range. More than the range holds means all of it.
- `BITTY_STAGE_THREADS=N`: CPU threads of the engine (llama.cpp picks 4 when
  unset).
- `BITTY_GPU_OUTPUT=1`: when the node holds the final stage and no block was
  asked on the GPU, put the output layer alone there. It is the most expensive
  single step of the model (47 ms against 64 for the last five-block stage of
  the 35B model) and takes little video memory.
- `GGML_VK_VISIBLE_DEVICES=0`: llama.cpp's own variable, to pick one card when
  the machine has several.

Only the tensors of the range that go to the GPU take video memory, so a card
too small for the model can still hold one stage of it.

## Loading ahead of the first request

The coordinator asks the workers to load their layers as soon as the cluster
stops changing: when no node has joined or left for 10 seconds, every worker
gets its range, fetches the stage file if it has none and starts its engine.
The first request then finds everything in memory. The wait is there because
every node that joins moves the ranges of the others, and a range that is about
to change is not worth a download of several gigabytes.
`BITTY_PREPARE_SETTLE_SECS` changes the wait; 0 turns this off, and the layers
are loaded by the first request as before.

The final stage sends the coordinator only the best 1024 logits, as pairs of
token id and logit: a few KB per token where the whole vocabulary is about
1 MB. The engine picks them itself (`bitty-stage --top-logits N`), so an engine
built before this option has to be rebuilt. `BITTY_TOP_LOGITS` changes the
number; 0 sends everything. Sampling gives the same text as long as the number
is not below `--top-k`.

Engines are started with `GOMP_SPINCOUNT=1000` unless that variable or
`OMP_WAIT_POLICY` is already set: idle threads spin briefly and then sleep.
The libgomp default spins long enough to take the cores from the other stages
of the same machine, and a fully passive wait costs 7 to 8 ms per call.

`BITTY_TIMING=1` on the coordinator prints, after each request, the time per
token spent in every worker.

See `docs/split-on-one-machine.md` for the measurements behind all this.
`engine/stage_ab.py` compares engine settings on one stage.

## Text and sampling

Workers only see token ids and hidden states. The coordinator, which is the
machine that takes the request, turns the prompt into tokens and the tokens back
into text with `bitty-stage --vocab-only`: a process that reads the vocabulary
and loads no weights (any stage file of the model is enough for it). It formats
the prompt with the model's chat template, picks each token from the logits of
the final stage (temperature, top-k, top-p, seed) and stops at the end of the
turn.

```sh
bitty generate --prompt "What is the capital of France?" --max-tokens 64
bitty generate --prompt "..." --temperature 0.7 --top-k 20 --top-p 0.95 --seed 3
bitty generate --prompt "..." --think        # the model reasons first
bitty generate --prompt "The capital of Italy is" --raw   # no chat template
```

Defaults: top-k 20, top-p 0.95, a new seed per request, thinking off.
`--temperature 0` always takes the most likely token. The chat prompt is built
by llama.cpp's built-in ChatML formatter, not by the Jinja template of the
model, with the opening of the reasoning block added as the template does.

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

Cluster path: two `bitty node` processes on one machine, each serving half of
the 0.8B model, answer `bitty generate` requests with the chat template, stop at
the end of the turn, repeat the same text for the same seed and deliver
multi-byte characters intact. This has not been run between different machines yet.

35B model, four `bitty node` processes on one machine (8 cores, 32 GB, CPU
only), ten layers each: the answers are correct in English and Italian and the
same seed gives the same text as a single node holding all forty layers.

| Setup | 120 tokens | Speed |
| --- | --- | --- |
| llama.cpp alone, whole model | 45 s | 2.6 tokens/s |
| one node, all 40 layers | 49 s | 2.7 tokens/s |
| four nodes, 10 layers each | 48 s | 2.7 tokens/s |
| one node, engine with 8 threads | 40 s | 3.1 tokens/s |

Splitting the model costs nothing measurable when the hops are local, so the
speed is the speed of the processors. Real links between machines add their
latency four times per token and are not measured yet. The engine runs with
llama.cpp's default of 4 threads; 8 helped a little and 16 (all hyperthreads)
was four times slower, and nodes have no setting for it yet. The first request
after start waits for the stages to load (about 100 s here). The nodes in this
run opened the whole model file with their layer range, not the stage files.

Node without the model: a second node started with the URL of the 0.8B model
and an empty data directory fetched 10 MiB of vocabulary at start and the 358
MiB stage file of layers 0..12 on the first request (43 s in all), then answered
with the node that holds the whole model. This is also the first run where a
node loads a stage file. Not tried on the 35B model.

GPU (Radeon RX 5500M through Vulkan, about 3.9 GB of its 8 GB free): the 0.8B
model split in three stages on the GPU produces the same tokens as the CPU
reference (`STAGE_CHECK_GPU_LAYERS=99 llama-stage-check -c 8,16`; logits differ
by up to 1.2 from rounding). On the 35B model, three layers of a middle stage on
the GPU took 1.3 GB of video memory, so tensors outside the range are not
loaded there. One node with all 40 layers, 6 of them on the GPU (3.4 GB) and 8
threads: 120 tokens in 33 s, against 40 s on the CPU alone. A whole 10-layer
stage (about 5 GB) did not fit in the free memory of this card, so a fully
offloaded stage is not measured.

Per-stage timing (`engine/stage_bench.py`): the 35B model in eight stages of
five layers, each run alone and fed the hidden states of the one before, 8
threads, one token at a time after a 16-token prompt.

| Stage | CPU, ms per token | GPU, ms per token |
| --- | --- | --- |
| each of the first seven | 26 to 29 | 11 to 12 |
| 35..40 (with the output layer) | 59 | 17 |
| sum of the eight | 251 (4.0 tokens/s) | 101 (9.9 tokens/s) |

A fully offloaded 5-layer stage takes 2.9 GB of video memory. The sums leave out
the network and the coordinator, so they are the ceiling for a chain of eight
such peers, not a measured cluster speed. The same script with all 40 layers in
one CPU process needs 278 ms per token in a clean run, a little more than the
eight separate stages together. Whole-model runs on this machine vary a lot
(up to 420 ms), most likely from memory pressure with 21 GB of weights on a
32 GB machine.

```sh
BITTY_STAGE_BIN=build/bin/bitty-stage python3 engine/stage_bench.py cpu model.gguf 5
BITTY_STAGE_BIN=build-vk/bin/bitty-stage python3 engine/stage_bench.py gpu model.gguf 5
```

## Not done yet

- Requests carry one user message: there is no system prompt, conversation
  history or tool calling in the cluster path yet.
- `bitty run --local` is not wired to the engine yet. `bitty chat` sends each
  line as a separate request with the default sampling and no memory of the
  earlier lines.
- Stage files are not checked against a hash and old ones are not cleaned up
  when the range of a node changes.
- GPU settings come from environment variables only, and nothing checks the
  free video memory before loading.
- State save/load does not record the range.
- `llama_model_size` still reports the whole file for a stage loaded from it.
