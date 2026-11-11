# Tests on the first ring

Three machines: the leader (Linux, 32 GB), an Ubuntu office PC (8 GB, 4 cores)
and a Windows PC. All runs are started on the leader with `BITTY_TIMING=1`, so
its log has one line per request with the passes, the draft tokens kept and the
milliseconds per token, split between workers and draft.

Every run uses the same prompt, `--seed 7` and `--max-tokens 80`, three times,
and the median counts. A number that was not measured is not written down.

## 1. The 35B ring as planned

| Run | Command | What it answers |
| --- | --- | --- |
| plain | `bitty run qwen3.5-35b-a3b "..." --no-draft` | tokens per second of the ring, and where the time goes |
| draft of 2, 4, 8 | the same with `--draft N` | whether guesses pay on a real network; one machine said 2.26 tokens per pass with 4 |
| temperature 0.7 | `--draft 4 --temperature 0.7` | how much a greedy draft loses when the answer is sampled |
| pipeline | the same with `--pipeline` | whether a guess that travels behind its token pays: the time of the shorter stage on right guesses, the cost of the draft and of the way back on wrong ones |

The draft pays when a pass over the network costs more than the draft's own
tokens on the leader. On one machine it did not (153 against 104 ms per token),
because a local pass is cheap. The ring is the first place where it can.

## 2. The 0.8B model on the same ring

The model is small enough for any cut, so the cost of the network alone shows:
the time per token minus the time the same layers take on one machine.

| Cut | Why |
| --- | --- |
| all layers on the leader | the base line |
| leader and one other machine | one hop |
| three machines | two hops, the ring as it is |

`bitty restart --model qwen3.5-0.8b` on the leader moves the whole network: each
node is told the model when it registers and comes back with it within about a
minute. `bitty restart --model qwen3.5-35b-a3b` goes back, and the layers of the
35B are still on the disks.

## 3. What a machine adds

For each machine of the ring: the tokens per second with it and without it.
A machine that makes the ring slower and is not needed for the memory should
stay out. This is the number the planner will use when "join only if faster"
is built, so it is measured before that is written.

## 4. Coming back

| Case | Expected |
| --- | --- |
| a worker restarts | it registers again, the plan is kept, no layer is fetched again |
| the leader restarts | the plan of the file `plan` is taken up, requests wait for the workers |
| a worker stays away past `BITTY_PLAN_WAIT_SECS` | a new plan without it, or "no plan" when the memory is not enough |
| a download is cut | it goes on from the `.partial` file |

## 5. A fourth, weak node

A server with 1 vCPU and 2 GB cannot hold a layer of the 35B model. It joins
only in the 0.8B runs, to see what a slow far machine costs in the ring. It
joins after the three machines have their numbers, because a new node changes
the plan.

## Measured on 2026-11-11: two machines

Leader 0..36, office PC 36..40, joined over iroh. Prompt "Explain in a few
sentences how a steam engine works.", `--seed 7 --max-tokens 80`, 68 tokens
come out. Median of three runs, milliseconds per token after the first pass.

| Run | Passes | Draft tokens kept | Total | Leader 0..36 | Office PC 36..40 | Draft and the rest |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 68 | none | 368 | 265 | 97 | 6 |
| draft of 2 | 31 | 37 of 60 | 401 | 253 | 60 | 88 |
| draft of 4 | 25 | 43 of 93 | 503 | 298 | 75 | 130 |
| draft of 8 | 25 | 43 of 93 | 490 | 298 | 63 | 129 |

What it says:

- The ring of two is as fast as the leader alone (about 380 ms per token with
  the whole model). A layer on the office PC costs about 25 ms with the
  network, against 7 ms on the leader: that machine is there for the memory.
- The draft does not pay here. It halves the passes, but a pass that carries
  three tokens costs the leader nearly three times one token, so only the
  network share shrinks (97 to 60), and the draft model costs more than that.
- A draft of 8 gives exactly what a draft of 4 gives: the engines keep copies
  of their state for 4 tokens (`BITTY_STAGE_ROLLBACK`), so a longer draft is
  cut to 4. The leader now says so in its log. A real draft of 8 needs that
  setting on every node, and each token more costs a copy of the state.
- The leader started with the path of the model file did not find its draft
  model; these runs had `BITTY_DRAFT_MODEL` set by hand. Corrected since.
- The office PC fetched 36..40 from the model URL although it held 25..40.
  Corrected since: a stage is cut out of a wider one that is on the disk.

Not measured yet: the Windows PC in the ring, a cut that gives the office PC
more layers, the 0.8B runs.
