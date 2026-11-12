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

A release carries the engine built with `-DGGML_VULKAN=ON` as
`bitty-stage-vulkan` (and `bitty-stage-vulkan-avx`, `bitty-stage-vulkan-sse`
for processors without AVX2), next to the builds for the processor. It is a file of its own because
it needs the Vulkan library of the system to start.

`bitty-stage --list-devices` loads nothing and writes one line for every card
the build can compute on: `device`, `gpu` or `igpu`, the free and the whole
memory in bytes, the name of the backend device and the name of the card,
separated by tabs. A build without a GPU backend writes nothing. This is how
bitty knows that the engine works on a machine, and not from a list of
hardware: a card without a working driver is not there.

When a node gets its layers it asks the engine for cards (`BITTY_STAGE_BIN`
when set, else `bitty-stage-vulkan`) and reads the sizes of the tensors of its
range from the model file. As many layers as fit in the free video memory, less
a reserve of 512 MB, are counted for the card, from the top of the range down;
on the final stage the output layer goes first. If the engine fails to load
with them, half of that memory is tried once.

Then the card has to prove itself. The node starts the engine for the
processor, decodes five tokens of made-up input and keeps the best time of the
last four; it does the same with the layers on the card (the first token is
left out because the card builds its programs on it). The card stays when its
time is at most 0.9 of the processor's, otherwise the engine for the processor
is started again. The verdict for a range is kept while the node runs. This
costs two or three loads the first time. A card built into the processor is
used only when asked for, and software drivers such as llvmpipe never. From
reading the free memory to the end of the load a node holds a lock file in the
temporary directory, so two nodes of one machine do not count the same memory.
The code is in `crates/bitty-inference/src/gpu.rs`.

The node tells the leader about the card only when it may use it. Its weight in
the plan is the one of its processor times `1 / ((1 - s) + s / 2.5)`, where `s`
is the video memory for weights over the memory the node can fill, and 2.5 the
gain of a layer on a card measured below. `BITTY_GPU_TFLOPS` still sets the
number by hand. The settings:

- `BITTY_GPU=off`: never use a card. `BITTY_GPU=igpu`: also use a built-in one
  when there is no other.
- `BITTY_GPU_LAYERS=N`: keep N layers of the node's range on the GPU, counted
  from the top of the range, in place of bitty's own count. More than the
  range holds means all of it, 0 none. A load that fails is then an error.
- `BITTY_GPU_RESERVE_MB=N`: the video memory bitty's own count leaves free.
- `BITTY_GPU_CHECK=0`: no timing, a card that loads is used.
- `BITTY_STAGE_THREADS=N`: CPU threads of the engine. Unset, bitty gives it
  one thread per physical core (llama.cpp alone would pick 4).
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

The final stage sends the coordinator only the best 64 logits of each token, as
pairs of token id and logit: 512 bytes where the whole vocabulary is about 1 MB.
The engine picks them itself (`bitty-stage --top-logits N`). `BITTY_TOP_LOGITS`
on the node that holds the final stage changes the number; 0 sends everything.
Sampling gives the same text as long as the number is not below `--top-k` (20
by default); a request with a larger `--top-k` samples among the 64. It was
1024 until a pass came to carry up to 9 tokens: 72 KB on the way back through
the relays for a pass of 8 guesses, 4.5 KB now.

### f16 after every layer

The hidden states go from stage to stage, 8 KB per token as f32 on the 35B
model. Sent as f16 they are half of that, but rounding them only where the
model is cut makes the text depend on where the cuts are: on the 0.8B model cut
in three that moved the best ten logits by up to 0.45, against 0 with f32. The
hidden states of that model are all below 4, so this is the precision of f16
itself, and keeping a few large values aside would not help.

So the engine rounds the residual stream to f16 after every layer, in every
stage alike (`bitty-stage --round-f16`, `stage_round_f16` in
`llama_model_params`, for `qwen35` and `qwen35moe`). The result is then the
same wherever the model is cut, and the stages send f16 without losing
anything: on the 0.8B model the rounded whole model and a rounded chain of three
stages sending f16 give the same logits to the last bit. The rounding itself
moves the logits by up to 0.54 against the model without it (the same tokens
there). On the 35B model it costs about 5% of the compute (290.8 ms against
276.0 for a token through four stages of ten layers on the CPU) and saves 4 KB a
token on every hop; on the ring of four machines with relays that is about 70 ms
a token. The text differs a little from the one without the rounding.

It is on by default. `BITTY_ROUND_F16=0` on every node turns it off, and the
hidden states then go as f32 unless `BITTY_WIRE_F16=1`; every stage reads both.
All the nodes of a ring should have the same setting.

### Lower quantizations for slower nodes

A node can hold its layers in a lower quantization than the model file. The
hidden states have the same form whatever the quantization of the weights, so
stages of different quantizations work together. For the 35B model the registry
names `Q3_K_M` (74% of the size) and `UD-Q2_K_XL` (55%) beside the `Q4_K_M` file.

The leader chooses per node by speed: the first node, which holds the
embeddings, and every node at least half as fast as the fastest keep the model
file; a node between a quarter and a half takes Q3, a slower one Q2. In the plan
a node on a lower quantization counts as faster in proportion to its smaller
weights (a token reads less memory) and its layers as smaller, so it takes more
of them. Its log shows the quantization beside the range (`bitty-1234 34..40 in
q2`); the plan file keeps it, so a node keeps its files while the plan holds.
The node fetches the stage file of that quantization from its published file,
with range requests, also when it holds the whole model in another one.

A chain of layers 0..38 from the Q4 file and 38..40 from the Q2 file (825 MiB
for the two layers and the output head, against 1408 in Q4) answers "The capital
of France is Paris. It is the country's most populous city and serves as its
center for finance, commerce", where the Q4 file alone writes "It is also the
country's largest city and serves as its major center for finance". The text
depends on which node holds which layers, so it changes when the ring does.

`BITTY_LOW_QUANT=off` on the leader keeps every node on the model file.

## A draft model on the leader

A small model with the same vocabulary can guess the next tokens, and the ring
checks several of them in one pass. The coordinator sends the last certain token
followed by the guesses; the final stage answers with the logits of every one of
them (`KIND_TOP_ROWS`), and the coordinator samples row by row. A guess is kept
while it is the token the model itself chooses. At the first other one the rest
is dropped: the next pass carries `rewind_to`, and every stage takes the wrong
tokens out of its memory before it decodes. One pass gives between 1 and N + 1
tokens.

The stages can only take back tokens of their last decode call, and not its
first one. `bitty-stage --rollback N` keeps a copy of the recurrent state for
each of the last N tokens (8 by default, `BITTY_STAGE_ROLLBACK`), so a pass
gives up to 9 tokens. A request for more guesses than the workers can take back
is cut down to that number.

The copies are not small. The state of one recurrent layer of the 35B model is
2.09 MiB for one request (llama.cpp reports it: `inner_size * state_size` f32
plus the window of the convolution), the engine keeps it for each of its 4
requests, and once more for every token it can take back. Measured on layers
0..4 (3 recurrent): 25 MiB with none, 126 with 4, 226 with 8; on 0..12 (9
recurrent): 75 and 678 MiB. A stage of 28 layers (21 recurrent) keeps about
1.6 GB with 8, a stage of 2 layers about 75 MB. The leader counts this for
every recurrent layer when it plans (`planning with ... MB of state` in its
log), with the `BITTY_STAGE_ROLLBACK` and sessions of its own settings, and a
node counts it again before it loads. Every node should have the same
`BITTY_STAGE_ROLLBACK` as the leader.

The draft model runs whole on the leader, as one more `bitty-stage` process.
For a model of the registry the file named by `draft_url` is downloaded next
to the model the first time the leader starts; `BITTY_DRAFT_MODEL` names
another file, or `off`. `BITTY_DRAFT_THREADS` sets its threads. When it fits
in the free video memory of the leader it is timed on the card and on the
processor, and stays where it is faster; `BITTY_DRAFT_GPU_LAYERS` sets the
layers on the card by hand (0 for none).

It is off unless asked for: `bitty generate --draft 4` or `bitty run MODEL
--draft 4` for one request, `BITTY_DRAFT_TOKENS=4` on the leader for all.
`--no-draft` turns it off for one request. Every worker must run a build
that knows it, otherwise the request fails with a message that says so.

Whether it is faster depends on the ring. A pass with five tokens costs each
stage more than a pass with one, and the draft model costs its own time, so on
one machine it is slower. It pays when most of the time of a pass is spent on
the network. `BITTY_TIMING=1` on the leader prints passes, guesses kept and
milliseconds per token for each request, with and without it.

The text is not always the one of a request without a draft. Decoding several
tokens in one call gives logits that differ by up to 0.3 from decoding them one
by one, and where two tokens are that close the choice can fall on the other.
`engine/spec_check.py` checks that every token chosen this way is within 0.5 of
the best one when the model reads the same text token by token.

### Guesses from text

The leader also guesses with no model at all, from text it has already seen, and
it does so by default. Before every pass that carries one certain token it looks
for the last three tokens of the text (then the last two) earlier in the same
request, the prompt and what was written so far, and sends what followed them
there as guesses. When neither is found but the model was copying a moment ago,
it looks for the last token alone a few places around where the copy had
reached, and goes on from there: a model that corrects a text changes a word and
copies on right after it. If the request has nothing, it looks in the answers the ring
gave before: the last three tokens are looked up in an index of them, and the
continuation most of them share is sent. The answers are kept as token ids in
`guesses/` in the data directory of the leader, about 4 MB for a million tokens
(`BITTY_GUESS_HISTORY_TOKENS` sets the limit, the oldest half goes first), one
file per model file and vocabulary size.

The guesses go through the same check as the ones of the draft model, which is
asked only when the text has none. A pass with no match is a plain pass.

How many guesses go in a pass is chosen by speed. With `k` guesses that are each
right with chance `p` a pass gives `(1 - p^(k+1)) / (1 - p)` tokens on average,
and it costs what a pass with `k` guesses costs on this ring. The leader keeps
the average time of the passes by their number of guesses, forgotten when the
plan changes. A pass without guesses keeps its own average; for the passes with
guesses it fits a straight line, a fixed cost plus a cost for each guess, through
the averages of the sizes tried, each weighed by the square of the passes behind
it, so that one pass that met a slow relay does not decide. It estimates `p`
from the request, starting from the earlier ones, and sends the number, between
0 and the 8 the engines can take back, that gives the most tokens a second. On the four-machine ring of the measurements a pass of 8
guesses cost about twice one of 4, so a text that was guessed nearly whole went
faster with 4. On one machine, where no trip is saved, it seldom sends any;
then every 16th pass still carries two, to notice a text that starts to repeat.
`BITTY_TIMING=1` prints the times it holds and the chance; `BITTY_GUESS_COUNT=N`
on the leader fixes the number for every pass with a match.

It pays on answers that copy the prompt (quotes, summaries, corrections, code)
and on answers that come back (fixed phrases, formats, the same question asked
again); on new text it finds little and costs nothing. The text can differ in
the same rare way as with the draft model.

`BITTY_GUESS_CACHE=request` on the leader keeps nothing between requests,
`BITTY_GUESS_CACHE=off` turns this off; `--no-draft` turns off every guess for
one request. Kept answers can be proposed to another person who asks something
close: a guess is only sent on when it is the token the model chooses anyway,
but it does reach the workers. `BITTY_TIMING=1` reports how many guessed tokens
were kept and how many of them came from text; `BITTY_TIMING=2` names the source
on every pass.

### The pipeline

`--pipeline` (or `BITTY_PIPELINE=1` on the leader) uses the draft model the
other way. In a ring only one machine works at a time: the others wait for the
token to come round. With the pipeline the leader sends the token and, right
behind it, the draft's guess for the next one as a pass of its own. While the
last machine works on the token, the first already works on the guess.

- The guess was right: its pass is already on its way, the next token costs
  only what is left of it.
- The guess was wrong: the leader takes it back. A machine that has not
  started it drops it, the one that is working on it stops within a few
  milliseconds (op `ABORT` of the engine) and puts its state back, one that
  finished it goes back with the next token.

Each worker does the steps of a request in the order they came, so a guess
never overtakes its token. One token is decoded at a time, so the text is the
one of a request without a draft: the smoke test compares the two, with guesses
that are right and with guesses that are wrong.

What it can give: with two machines the time of the shorter stage, on the
tokens the draft gets right. A ring cut in equal parts, or with long trips
between the machines, gains more. It needs two workers or more and a route
from worker to worker; `--no-draft` turns it off. It is not used together with
`--draft N`.

Engines are started with `GOMP_SPINCOUNT=1000` unless that variable or
`OMP_WAIT_POLICY` is already set: idle threads spin briefly and then sleep.
The libgomp default spins long enough to take the cores from the other stages
of the same machine, and a fully passive wait costs 7 to 8 ms per call.

A token goes from worker to worker: the coordinator gives it to the first,
each worker runs its layers and hands the result to the next one itself, and
the last one keeps the logits for the coordinator, which has been waiting on
it since the token left. The data crosses the network once per worker plus
once, where it used to go back to the coordinator after every worker (twice
per worker). The workers reach each other at the addresses the coordinator
knows them by, so they have to be reachable from one another, not only from
the coordinator.

If a worker cannot hand over the first token of a request (no route to the
next one, or a node with an older version), the coordinator says so in its
log, starts the request again with every worker answering it directly, and
keeps doing so until a node joins or leaves. `BITTY_CHAIN=0` on the
coordinator asks for that from the start. A worker that dies in the middle of
a token is noticed only by a timeout, 10 minutes unless
`BITTY_CHAIN_TIMEOUT_SECS` says otherwise.

`BITTY_TIMING=1` on the coordinator prints, after each request, the time per
token, the part of it spent in every worker, and which of the two ways was
used. From worker to worker the workers count their own time; through the
coordinator it is the coordinator's wait, so it includes the network.

`BITTY_TIMING=2` adds one line for every pass: whether it rode a guess of the
pipeline or was sent, its whole time, the time of the draft model, the wait
for the answer, the time of each worker, what became of the guess sent behind
it and how long taking a wrong one back took.

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
- The layers for the card are counted from the weights alone: the context and
  the buffers of the computation are covered by a fixed reserve, not measured.
- The verdict of the timing is not kept across restarts of a node.
- The release builds have no OpenMP, so `GOMP_SPINCOUNT` does nothing in them
  and each call starts its own threads. Which of the two is faster in a ring
  is not measured.
- The prebuilt engines for cards are checked in the release only on a runner
  without a card.
- State save/load does not record the range.
- `llama_model_size` still reports the whole file for a stage loaded from it.
