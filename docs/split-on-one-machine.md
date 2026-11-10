# Splitting the model on one machine: what was measured

Notes from 2026-11-10, branch `qwen-llama-cpp`. They record one finding that
looks worth working on and, just as carefully, what has not been shown yet.

## The finding in one paragraph

Qwen3.5-35B-A3B (Q4_K_M, 21 GB, 40 blocks) run as eight stages of five blocks,
each stage **measured alone**, costs less per token than the same 40 blocks in
one process. On CPU the eight stages add up to about 255 ms per token (3.9
tokens/s) against 357 ms (2.8 tokens/s) for the whole model in one engine. With
each stage fully on the GPU the eight add up to about 100 ms per token (10
tokens/s).

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
one CPU engine on the same bench: 357 ms per token.

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
  one: 74 s against 47 s. `bitty` now starts every engine with
  `OMP_WAIT_POLICY=passive` unless the variable is already set.

Not shown:

- Ten tokens/s on this machine. It is a projection for eight GPUs.
- That separate stages really beat one process on CPU when they run as a
  cluster. Alone they sum to 255 ms per token, but the real eight-node run
  needs about 390 ms per token (47.1 s for 120 tokens, with about 5 s for the
  prompt). The gap between 255 and 390 is not explained yet.
- Anything about a real network. All nodes were on `127.0.0.1`.

## Open questions, with the experiment for each

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
