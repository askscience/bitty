# Splitting the model on one machine: what was measured

Notes from 2026-11-10, branch `qwen-llama-cpp`. They record one finding that
looks worth working on and, just as carefully, what has not been shown yet.

## The finding in one paragraph

Qwen3.5-35B-A3B (Q4_K_M, 21 GB, 40 blocks) run as eight stages of five blocks,
each stage **measured alone**, costs about the same per token as the 40 blocks
in one process. On CPU the eight stages add up to about 255 ms per token (3.9
tokens/s); the whole model in one engine needs about 278 ms in a clean run. With
each stage fully on the GPU the eight add up to about 100 ms per token (10
tokens/s).

An earlier version of these notes said one process needs 357 ms and is clearly
slower than its parts. That was noise: see "Optimisation round" below.

That last figure is the sum of stages measured one after the other. It is what
eight peers with an 8 GB GPU each would compute, before network time. This
machine cannot do it by itself: only one stage fits in the free video memory at
a time. Running all eight stages together here gave 2.5 tokens/s at best.

## Machine

- Intel i9-9980HK, 8 cores, 16 threads, laptop cooling
- 32 GB RAM, no swap
- AMD Radeon RX 5500M, 8 GB, through Vulkan (RADV); the desktop already uses
  about 2.5 GB, so about 3.9 GB are free
- llama.cpp at `45c8ea8` with `engine/llama-cpp-stage.patch`, CPU build and
  Vulkan build
- `GGML_VK_VISIBLE_DEVICES=0` selects the Radeon (the other device is the
  Intel UHD)

## Numbers

Prompt and seed are the same everywhere ("Spiega in italiano, in modo semplice,
come funziona un motore a vapore.", seed 7, 120 new tokens) unless noted.

### Whole request, 120 tokens

| Setup | Seconds | Tokens/s |
| --- | --- | --- |
| llama.cpp alone (`llama-simple`), 4 threads | 45 | 2.65 |
| one bitty node, 40 blocks, 4 threads | 49 | 2.45 |
| four bitty nodes, 10 blocks each, 4 threads each | 48 | 2.5 |
| one bitty node, 8 threads | 40 | 3.0 |
| one bitty node, 8 threads, 6 blocks on GPU (3.4 GB VRAM) | 33 | 3.6 |
| eight bitty nodes, 5 blocks each, 8 threads each | 74.3 | 1.6 |
| the same, `OMP_WAIT_POLICY=passive` | 47.1 | 2.5 |
| eight nodes, last stage on GPU, spinning threads | 61.0 | 2.0 |

16 threads are about four times slower than 4: hyperthreads hurt here.

### Stage by stage (`engine/stage_bench.py`)

Eight stages of five blocks. Each stage is started alone, fed the real hidden
states produced by the stage before it, timed, and stopped before the next one
starts. 8 threads. Median milliseconds per token.

| Stage | CPU | GPU (all 5 blocks) |
| --- | --- | --- |
| 0..5 to 30..35, each | 26 to 29 | 11 to 12 |
| 35..40 (also the output layer) | 59 | 17 |
| Sum, 60 tokens per stage | 251 (4.0 tokens/s) | 101 (9.9 tokens/s) |
| Sum, 400 tokens per stage | 258 (3.88 tokens/s) | 96.9 (10.32 tokens/s) |

The run with 400 tokens per stage is there to rule out a short burst before the
CPU heats up: the figures hold.

A stage fully on the GPU takes about 2.9 GB of video memory. All 40 blocks in
one CPU engine on the same bench: 357 ms per token in the first run, 278 ms in
a later clean one (whole-model runs on this machine swing between 280 and 420
ms, see below).

## What this does and does not show

Shown:

- The split costs nothing in quality: four nodes and one node give the same
  text with the same seed.
- The overhead of bitty itself (gRPC between local nodes, hidden states of
  2048 floats per token) is small: four nodes take the same time as one.
- A five-block stage of this model fits in 3 GB of video memory, and on a
  mid-range laptop GPU it runs about 2.5 times faster than on eight CPU cores.
- Idle engines must sleep. With the default OpenMP wait policy the threads of
  the seven waiting stages keep spinning and take the cores from the working
  one: 74 s against 47 s with `OMP_WAIT_POLICY=passive`. Fully passive threads
  turned out to cost 7 to 8 ms per stage call, so `bitty` now starts every
  engine with a short spin (`GOMP_SPINCOUNT=1000`): 39 s.

Not shown:

- Ten tokens/s on this machine. It is a projection for eight GPUs.
- That separate stages really beat one process on CPU when they run as a
  cluster. Alone they sum to 255 ms per token, but the real eight-node run
  needed about 390 ms per token (47.1 s for 120 tokens, with about 5 s for the
  prompt). After the optimisation round below it needs 305 ms; the rest of the
  gap is about 5 ms per hop.
- Anything about a real network. All nodes were on `127.0.0.1`.

## Open questions, with the experiment for each

Questions 1, 2, 3 and 6 were worked on after this list was written: the answers
so far are in "Optimisation round" and "First run over the Internet" below. The
list is kept as it was.

1. **Why is one 40-block process slower than the sum of its parts (357 vs 255
   ms)?** Guesses: memory. 21 GB of weights on a 32 GB machine with a desktop
   running leaves little page cache, and a MoE model touches different experts
   at every token; a five-block stage keeps its 2.6 GB hot. Check with
   `perf stat` (cache and TLB misses), with `--mlock`, with huge pages, and by
   timing blocks inside the single process (per-block timers in the patch).
   Also check heat: log the CPU frequency during both runs.
2. **Why does the real eight-node run lose what the stage bench promises (390
   vs 255 ms)?** Guesses: thread wake-up after passive wait at every hop (eight
   wake-ups per token), the eight processes fighting for the same page cache,
   context switches. Try: pin each stage to its cores, 4 threads per stage,
   `GOMP_SPINCOUNT` with a short spin in place of fully passive, one engine
   process holding several ranges, timing each hop in the coordinator (the
   `gen_latency_us` field is already there, per-worker time is not).
3. **Does a middle ground help on one machine?** Two or three stages on CPU
   plus one on GPU, picked by the bench: the GPU stage should be the last one
   (59 ms on CPU, 17 on GPU, saving 42 ms per token). The eight-node run with
   the last stage on GPU was measured only with spinning threads (61 s); it
   has to be repeated with passive wait, where about 42 s is expected.
4. **How much of the prompt time can be removed?** The prompt is evaluated one
   stage after the other with the whole batch; that is already parallel inside
   a stage. Speculative pipelining between stages only helps prompts, not
   generation.
5. **Batching several requests.** The stage engine already has sessions. With
   eight GPU peers a token costs 11 ms per stage, so a peer is idle seven
   eighths of the time: eight requests could run through the chain together at
   close to the same speed each. This is the real advantage of the split and
   has not been measured.
6. **Network cost per hop.** 2048 floats are 8 KB per token per hop, and the
   last stage sends about 1 MB of logits. On a home link the logits dominate:
   send only the top values (the sampler uses top-k 20 by default).

## Optimisation round (2026-11-10, later the same day)

Different techniques tried one at a time. Unless noted: CPU build, 8 threads,
stage 10..15 of the 35B model, median ms per token over 60 tokens, measured
with `engine/stage_ab.py`, which runs the variants in turn and repeats within
0.3 ms.

### Threads of one stage

| Variant | ms per token |
|---|---|
| default (libgomp spins for a while after each parallel region) | 27.4 |
| `OMP_WAIT_POLICY=active` | 27.4 |
| `OMP_PLACES=cores OMP_PROC_BIND=close` | 27.6 |
| `OMP_WAIT_POLICY=passive` | 35.0 |
| cores bound and passive | 32.7 |
| 4 threads | 34 |
| 6 threads | 29.2 |
| 12 threads | 26.1 |
| 16 threads | 47 |
| `GOMP_SPINCOUNT=100` | 32.7 |
| `GOMP_SPINCOUNT=1000` | 28.6 to 29.2 |
| `GOMP_SPINCOUNT=10000` | 28.0 to 28.7 |
| `GOMP_SPINCOUNT=50000` | 27.6 |

Binding threads to cores does nothing. Passive waiting is expensive because the
threads go to sleep at every barrier inside one token, not only between
requests. Twelve threads on eight cores give 5%; sixteen are a disaster.

### The eight-node cluster, 120 tokens

| Variant | Seconds |
|---|---|
| default spinning (earlier run) | 74.3 |
| `OMP_WAIT_POLICY=passive` | 46.1 |
| `GOMP_SPINCOUNT=50000` | 44.9 |
| `GOMP_SPINCOUNT=10000` | 43.3 |
| `GOMP_SPINCOUNT=1000` (now the default) | 39.3 |
| the same, final stage picks the best logits itself | 39.0 |

One run each. With many stages on one machine a long spin steals the cores of
the stage that works, a sleep costs wake-ups inside every token; a spin of
about a thousand iterations is the best of the values tried. On peers with one
stage each the choice matters less, and `GOMP_SPINCOUNT` or `OMP_WAIT_POLICY`
set by hand are left alone.

`BITTY_TIMING=1` on the coordinator prints the time of each hop. With the
defaults: 305 ms per token in the workers, 32 to 37 ms for each of the first
seven hops (the engine alone needs 27.5) and 66 for the last (engine alone 59
to 64).

### Where the time of a hop goes outside the engine

Measured with temporary timers on one node with the 0.8B model: the engine call
took 60 ms and the worker 67 to 70. The 7 to 8 ms between them were spent on
the megabyte of logits: two conversions, a checksum and the choice of the best
1024. This is much slower than the same code should be, and the reason is
probably the CPU frequency: the machine uses the `powersave` governor, the
thread that receives the reply has slept during the whole decode, and it wakes
on a core at 800 MHz. The idle-gap test supports the engine side of this (a 250
ms pause between calls costs the engine only 1 ms, its threads spin up
quickly), the frequency of the receiving thread was not measured directly.

The fix does not depend on the cause: the engine now picks the best logits
itself (`--top-logits`), in the thread that has just done the decode, and
sends 8 KB through the pipe. One node, 0.8B: 69 to 63 ms per token, the engine
alone 60. Eight nodes, 35B: the last hop went from 71 to 66 ms.

The frequency guess was then tested and is wrong. With the `performance`
governor on all cores (and the settings put back afterwards) nothing moved:

| | `powersave` | `performance` |
|---|---|---|
| one node, 0.8B, ms per token | 63 | 62 |
| 0.8B engine alone | 58.5 | 58.8 |
| stage 10..15 of 35B alone, default / spin 1000 / passive | 27.4 / 28.9 / 35.7 | 27.3 / 28.7 / 35.1 |
| eight nodes, 35B, 120 tokens | 39.0 s | 39.9 s |
| the first seven hops, ms each | 32 to 37 | 32 to 36 |

So the governor is not worth changing, and the 5 ms that every hop adds to its
engine (32 to 36 ms against 27.5) are still unexplained. Next thing to do is
to time the inside of a middle hop the way the last one was timed: gRPC in,
pipe, engine, gRPC out.

### Output layer on the GPU

The last stage does the output projection over 248320 tokens, which is why it
costs twice the others. Last stage 35..40 alone, Vulkan build:

| Layers on the GPU | ms per token |
|---|---|
| none | 64.5 |
| output layer only (`--gpu-layers 1`) | 46.8 |
| output layer and one block | 39.9 |
| output layer and two blocks | 35.3 |

`BITTY_GPU_OUTPUT=1` on a node asks for the first row: the output layer on the
GPU when the node holds the final stage, with no block there. In the
eight-node cluster the last hop went from 66 to 49 ms. The whole run was not
faster (41.5 s against 39.0) because all eight nodes used the Vulkan build and
its CPU-only stages were slower in that run (37 to 41 ms a hop against 32 to
37); the run with the Vulkan build on the last node only has not been done.

### One process against its parts, again

Clean run of `stage_bench.py`, sum of the stages in ms per token: stages of 5
blocks 248 to 258, of 10 blocks 248, of 20 blocks 263, all 40 in one process
278. A/B runs of the whole model swing from 310 to 420 ms with prompt times
from 4.6 to 17 s, on a machine where 21 GB of weights leave little page cache
next to a desktop. So one process is at most a little slower than its parts,
and the 357 ms of the first notes was a bad run.

### Starting a client

`bitty status` and `bitty generate` through an invite waited about 3.5 s for
their own relay connection before dialling. An invite already carries the
relay and the addresses of the node, so the wait is now skipped: status takes
0.05 s on the same machine and 1.1 s with a relay-only invite. Not yet
repeated between two machines.

### The small model on one node

Qwen3.5-0.8B on one node needed about 83 ms per token with the default 4
threads and 69 with 8; the engine alone 72 and 58. With the logits fix: 63 ms
with 8 threads.

## Would a small draft model help? (2026-11-10)

Speculative decoding: a small model proposes a few tokens, the chain checks
them in one pass and keeps the leading ones that match, plus its own token at
the first miss. Two things decide whether it pays, both measured with
`engine/spec_probe.py`. Nothing of this is implemented.

### How often the draft is right

Target Qwen3.5-35B-A3B, draft Qwen3.5-0.8B (same vocabulary, same prompt
tokens), greedy on both, six prompts, 659 tokens.

| Prompt | Same token |
|---|---|
| steam engine, Italian | 58% |
| fall of Rome | 68% |
| Python function | 91% |
| short story, Italian | 48% |
| list of tips | 65% |
| TCP and UDP | 64% |
| all | 64% |

| Tokens the draft proposes | Tokens per pass of the chain |
|---|---|
| 2 | 1.63 |
| 3 | 2.00 |
| 4 | 2.26 |
| 6 | 2.47 |
| 8 | 2.63 |

Greedy matching only; with sampling the acceptance rule is different and was
not measured.

### What a pass with several tokens costs

One decode call, median ms, random inputs after 32 tokens of context:

| Tokens | CPU 10..15 | CPU 35..40 | GPU 10..15 | 0.8B whole, CPU | 0.8B whole, GPU |
|---|---|---|---|---|---|
| 1 | 28.9 | 57.9 | 16.3 | 62.4 | 38.8 |
| 2 | 36.5 | 64.3 | 19.5 | 77.7 | 40.2 |
| 4 | 57.0 | 81.2 | 21.8 | 103.0 | 44.4 |
| 8 | 94.9 | 114.0 | 32.4 | 163.9 | 50.4 |

The final stage here computed the logits of the last token only; checking a
draft needs them for every token, so its real cost is higher (about 30 ms per
extra token on CPU if the output layer does not batch well; not measured).

### What follows

Estimates, not measurements. With a draft of 4 a pass of the eight CPU stages
costs about 590 ms (seven times 57, the last about 150, hop overhead), and the
draft another 250 ms on CPU: about 840 ms for 2.26 tokens, 370 ms per token.
Today the same cluster needs 305. On one machine with CPU stages it is a loss.

It pays when the time of a pass is mostly not compute. If the network adds N
ms per pass: now 305 + N per token, with the draft (840 + N) / 2.26. The two
are equal at N of about 120 ms, and at N = 500 it is 805 against 590 ms per
token. With GPU stages (a pass of 4 tokens costs about 1.3 times a pass of
one) the gain is larger. The weak point is the draft itself: the 0.8B model
needs 62 ms per token on CPU and 39 on this GPU, a large share of it in the
output layer over 248320 tokens.

## How to repeat

```sh
export BITTY_STAGE_BIN=~/bitty-engine/llama.cpp/build/bin/bitty-stage
python3 engine/stage_bench.py cpu model.gguf 5

export BITTY_STAGE_BIN=~/bitty-engine/llama.cpp/build-vk/bin/bitty-stage
GGML_VK_VISIBLE_DEVICES=0 python3 engine/stage_bench.py gpu model.gguf 5

BENCH_TOKENS=400 python3 engine/stage_bench.py cpu model.gguf 5   # sustained
```

The eight-node run: one `bitty node --no-iroh --model ... --layers 40` as
leader and seven with `--join`, all with `BITTY_STAGE_THREADS=8`, then
`bitty generate --seed 7 --max-tokens 120`.

## First run over the Internet (2026-11-10)

Two machines, Qwen3.5-0.8B: the laptop above as leader with layers 0..15, and a
small VPS (1 core, 1.8 GB RAM, Debian 13) with layers 15..24. The VPS had no
model: it joined through an iroh invite, fetched the vocabulary (10 MiB) and
its stage (319 MiB) from HuggingFace and loaded it before the first request.
Its engine was built without `-march=native` and without OpenMP so that it runs
on any x86_64 with AVX2.

- The text is the same as on one local node with the same seed.
- 60 tokens took 103 s, about 1.7 s per token, against 9.3 s on one local
  node.
- The VPS engine used about 1% of its core: compute is not the cost.
- The laptop received 22 MB for 20 tokens, at 4.9 Mbit/s. That is the logits
  of the last stage, about 1 MB per token (248320 floats).
- The plain link between the two machines carries 43 Mbit/s (measured with
  ssh) with a 33 ms round trip, so the iroh path gave a ninth of the link.
  Whether the connection was direct or went through the relay was not checked.

So over a real network the full logits are the bottleneck, by a wide margin.
Sending only the top values (a few hundred pairs of id and logit, a few KB)
should bring a token close to the round trip time plus compute. After that the
next thing to look at is the iroh path (direct or relay, and its throughput).

### After the two fixes (same day)

| Version | 60 tokens | Received by the leader |
| --- | --- | --- |
| full logits, new connection per request | 103 s | 22 MB per 20 tokens |
| best 1024 logits | 31 to 35 s | 1.4 MB per 60 tokens |
| best 1024 logits, one kept connection per worker | 16 to 22 s | |
| one local node, for reference | 5.3 to 8.5 s | |

1. The worker now sends only the best 1024 logits as pairs of id and logit
   (`BITTY_TOP_LOGITS`, 0 for everything). The text is unchanged.
2. The coordinator opened a new iroh connection for every request, so every
   token paid a handshake of several round trips. It now keeps one connection
   per worker and opens a stream for each request.

About 3.5 s of every `bitty generate` through an iroh invite is the start of
the client's own iroh endpoint (a plain `bitty status` takes that long). Without
it a token costs about 0.2 s over this link, against roughly 0.1 s locally.

Still open here: whether the path is direct or through the relay, the start-up
time of the client, and why one local node needs about 90 ms per token for a
model this small.
