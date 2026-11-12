# Install and automatic updates

Bitty is two programs: `bitty` (network and orchestration, Rust) and
`bitty-stage` (the patched llama.cpp that computes a range of layers). The
installers put both on a machine and keep them current.

## Linux

```bash
curl -fsSL https://raw.githubusercontent.com/askscience/bitty/qwen-llama-cpp/scripts/install.sh | bash
```

- Binaries: `~/.local/share/bitty/bin` (`bitty`, `bitty-stage`, `bitty-update`).
- Command: `~/.local/bin/bitty` is a link to the first one.
- Updater: a systemd user timer, `bitty-update.timer`. Where there is no systemd
  user session it is a cron job.
- Needs `curl`, `tar` and `sha256sum`. No root.

## Windows

In PowerShell:

```powershell
irm https://raw.githubusercontent.com/askscience/bitty/qwen-llama-cpp/scripts/install.ps1 | iex
```

- Binaries: `%LOCALAPPDATA%\bitty\bin` (`bitty.exe`, `bitty-stage.exe`,
  `bitty-update.ps1`). The folder is added to the user `PATH`: open a new
  terminal after installing.
- Updater: a scheduled task named `bitty-update`.
- Windows 10 or later, 64-bit Intel or AMD. No administrator rights.

The first time a node starts, Windows Firewall may ask whether `bitty.exe` may
accept connections. Nodes that reach each other through iroh work without it,
because every connection is opened from the inside.

Builds up to `747bd74` doubled the backslashes of the paths in `config.toml` and
`runtime\runtime.toml` at every save. After enough restarts the node no longer
started, with `Os { code: 206 ... }` (file name too long). Later builds read the
doubled paths back as they should be, so the updater alone repairs such a
machine.

## What the updater does

Every minute, and when the machine starts (Linux) or the user logs in
(Windows):

1. It reads `VERSION` from the `nightly` release. The same as the installed
   one: nothing else is downloaded.
2. A new one: it downloads the archive, checks it against `SHA256SUMS`, checks
   that the new `bitty` runs, and swaps the files. A failed download or a wrong
   checksum changes nothing.
3. If a node was running it is restarted with `bitty restart --if-started`. It
   comes back in the same role: a worker joins the same cluster again, a leader
   keeps the same invite.
4. `bitty start --if-started` brings the node back if it is not running, which
   is what happens after a reboot or a crash.

Steps 3 and 4 do nothing on a machine where no node was started, or where it
was stopped with `bitty stop`: stopping by hand is remembered. A plain `bitty start` after
that goes back to what the node was doing: a node that had joined a cluster
joins it again.

A new build is published by the `Release` workflow on every push to the
`qwen-llama-cpp` branch, after it has built both programs for Linux and
Windows and run two nodes end to end on each. So a push reaches every machine
within about a minute of the workflow ending, with no command to give.

What happened is in `update.log`, next to the `bin` folder.

The machines of a cluster do not update at the same moment. That is fine:
workers register again by themselves when the leader comes back, and a worker
that is restarting is seen as gone and then as new. A request that is being
answered while a node restarts fails and has to be asked again.

## Settings of one machine

`bitty` reads `NAME=value` lines from the file `env` in its data directory
(`~/.bitty/env`, on Windows `%USERPROFILE%\.bitty\env`). Put the variables
there, not in the shell: a node restarted by the updater does not get the
variables of the terminal that started it the first time.

```
BITTY_STAGE_THREADS=8
BITTY_GPU_LAYERS=12
```

A variable set in the environment wins over the file.

Without `BITTY_STAGE_THREADS` the engine gets one thread per physical core of
the machine. On the eight-core leader that is 316-329 ms per token against
362-386 with the four threads llama.cpp picks by itself.

## A network of machines

```bash
# first machine
bitty share home                      # starts the node, prints the invite

# every other machine
bitty connect 'iroh://...' --name home

# anywhere
bitty run qwen3.5-35b-a3b "hello"
```

No machine needs the whole 22 GB model. A node downloads the vocabulary and
then only the layers it is given.

### Which machines get layers

A token goes through the machines one after the other. One machine more does
not make the layers faster, it adds a trip over the network. So the planner
uses the machines that are near the leader, and takes one that is far only
while the nearer ones cannot hold the model in the memory they have free. Far
machines come in by how far they are, the nearest first.

Near means that the leader answers in less than 30 ms. Every node measures
this by itself: three times before it registers, and then with every
heartbeat. The shortest of the last eight round trips counts.
`BITTY_NETWORK_RTT_MS=120` in the `env` file of a node gives the number by
hand and turns the measuring off.

A node keeps one connection to its leader and sends every heartbeat on it, so
the number is one round trip and not a handshake. The log of a node says
which build of the engine its processor got (`bitty: engine bitty-stage-avx`).

The number is the trip to the leader, not the trip between two nodes. Two far
nodes that sit beside each other are still counted as two far trips.

A plan that holds is kept: a node is not moved because its link got slower or
faster. The measure counts the next time the leader has to plan.

### How many layers a machine gets

The layers are shared in proportion to speed, inside what the memory and the
disk of each machine can hold. A layer of the 35B model is about 0.5 GB on
disk and about the same in memory, and more than half of that (about 290 MB a
layer, measured) is memory the system cannot give back under pressure. A node
needs about 1.5 GB more for the engine and its context.

The first limit is 90% of the memory that is free on the machine, so one that
also runs a desktop and a browser keeps working. Only when the layers do not
fit anywhere inside that limit does the planner go further, up to the smaller
of 85% of the whole memory and 125% of the free memory. A machine with no
room for one layer is left out of the ring.

The node has the last word. Before it loads its layers it looks at the memory
free at that moment, and it refuses layers that are past its limit. The leader
then plans again with the number the node gave. The same holds for the disk:
a node first removes the layer files of an older plan, and when the new ones
still do not fit it says how much room it has. An engine that dies while it
loads its layers is taken as the same answer without a number: the leader
plans that machine again with three quarters of what it was given. When the
engine dies a second time the size is not the reason, and the machine gets no
layers until it comes back with another build.

`BITTY_LEAVE_OUT=bitty-1234abcd,bitty-5678ef01` on the leader gives no layers
to those machines. They stay connected. It is for tests, to see what a
machine adds, and for a machine that is known not to work.

The limits are those of the memory of the machine, also when it has a graphics
card. The engine puts on the card only the layers `BITTY_GPU_LAYERS` names,
and without that setting the card is not counted at all, neither for speed nor
for room. With it, a card larger than the memory of the machine raises the
limit and a smaller one changes nothing.

A download of layers that is cut, by a restart, an update or a line that
drops, goes on from where it was the next time.

A node keeps its name in `node-id` inside its data directory, so after a
restart or an update the leader sees the same machine and not a new one.

The leader keeps its plan, which machine holds which layers, in the file
`plan-NAME` of its data directory, one for each model. The plan changes when a machine joins, when one
stays away, or when a range no longer fits its machine, and for nothing else:
every range that moves is a download of gigabytes. After a restart of the
leader, or when a machine drops out, the leader waits 10 minutes for the
missing machines of the plan before it plans without them
(`BITTY_PLAN_WAIT_SECS`); an update restarts a node in far less. Requests in
that time are answered with what the leader is waiting for. When the leader
drops a plan it writes why in its log (`the kept plan is dropped: ...`).

The leader names the model. A node that registers is told the model of the
leader, by its name in the registry or by the URL of the GGUF. A background node
that runs another model saves that name, ends, and is started again with it by
`bitty start --if-started`, which the updater runs every minute. So the whole
network goes to another model with one command on the leader, for example
`bitty restart --model qwen3.5-0.8b`, and comes back the same way. The layers of
the first model stay on the disks, and the plan of each model has its own file
(`plan-NAME`), so coming back fetches nothing again. A node started with a file
of its own, a path outside the model directory, keeps that file.

Each node tells the leader, with every heartbeat, what it holds of the model and
what it is still fetching. The leader writes it in its log when it changes, and
the progress of a download every five minutes, for example
`bitty-c5e9968c: fetching 10..34, 5120 MiB so far` and later
`bitty-c5e9968c: holds 10..34`. So the state of a far machine can be read on the
leader without going there. A node on an older build says nothing.
`stopped fetching 29..40 at 2381 MiB` is a download that was given up, for
example because the plan changed: nothing has written to it for minutes.

When the plan gives a machine fewer layers than a file it already holds, for
example 36..40 when it has 25..40, the new file is cut out of the old one on
the disk and nothing comes over the network.

### Straight or through a relay

Each node writes in its log how it reaches the next one, and again when that
changes:

```
bitty: link to 90a1b52b49: direct (Ip(192.168.1.20:55424)), round trip 2 ms
bitty: link to 1d0da13c2f: through a relay (Relay(https://use1-1.relay...)), round trip 240 ms
```

A link through a relay makes every token cross a server far away. When the
time per token is far above the time the workers report, look for these lines.

Two machines in different homes often cannot reach each other straight, and
then the relay stays. So each node times a connection to every relay when it
starts and sits behind the nearest one:

```
bitty: relay https://euc1-1.relay.n0.iroh-canary.iroh.link./, 34 ms away
```

`BITTY_RELAY=default` in `~/.bitty/env` leaves the choice to iroh, which has
been seen to pick a relay on the other side of the ocean.

## Reading and restarting a far machine from the leader

On the leader:

```
bitty ask bitty-1234abcd logs
bitty ask bitty-1234abcd restart
bitty ask bitty-1234abcd update
```

`logs`: with its next heartbeat the node sends its build, the last lines of its
own log and of the log of its updater, each kind of line once and without the
token of the cluster. The leader keeps them in
`~/.bitty/logs/node-bitty-1234abcd.log`. The leader asks by itself when a node
could not load its layers.

`restart`: a node that runs in the background ends, and its updater starts it
again within a minute. A node somebody started in a terminal stays.

`update`: the node runs the updater of its installation at once, so a machine
whose timer or scheduled task does not work is still kept current. The leader
gives this order by itself to a node that registers with a released build
other than its own.

When the engine of a node dies, the error the leader logs says
how it ended (exit code or signal), what the processor of that machine has
(`avx`, `avx2`, `fma`, `f16c`) and the last lines the engine wrote.

Both need this build on the node. An older node ignores them.

To give a machine less than it would get, put one of these in its `env` file
and restart the node:

```
BITTY_AVAILABLE_RAM_MB=6000
BITTY_MAX_LAYERS=6
```

## Options

| Linux | Windows | |
| --- | --- | --- |
| `install.sh --no-auto-update` | `install.ps1 -NoAutoUpdate`, or `$env:BITTY_NO_AUTO_UPDATE=1` before the one-line form | install without the updater |
| `bitty-update --update` | `bitty-update.ps1 -Update` | check now |
| `install.sh --uninstall` | `install.ps1 -Uninstall` | remove binaries and updater; models and settings stay |

Environment: `BITTY_HOME` (where the binaries go), `BITTY_BIN_DIR` (Linux,
where the `bitty` link goes), `BITTY_CHANNEL` (release tag to follow, default
`nightly`).

To pass an option to the one-line form on Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/askscience/bitty/qwen-llama-cpp/scripts/install.sh | bash -s -- --no-auto-update
```

## What the prebuilt binaries are

- x86-64 only. The engine comes in three builds: `bitty-stage` for processors
  with AVX2 (Intel from 2013, AMD from 2015), `bitty-stage-avx` for older ones
  with AVX, `bitty-stage-sse` for the ones with SSE4.2 only. The node looks at
  its processor and starts the one that fits. The older the build the slower
  a layer. `BITTY_STAGE_BIN` names one by hand.
- A node keeps its machine awake while it runs (Windows and Linux with
  systemd), so a machine left alone stays in the ring. `BITTY_ALLOW_SLEEP=1`
  in the `env` file lets it sleep.
- CPU only. For a GPU build the engine yourself with Vulkan as described in
  [engine/README.md](../engine/README.md) and point `BITTY_STAGE_BIN` at it in
  the `env` file. The updater keeps updating `bitty` and leaves your engine
  alone.
- The Linux binaries are built on Ubuntu 22.04 and need glibc 2.35 or later.

Other platforms build from source: `cargo build --release -p bitty-cli`, and
the engine as in [engine/README.md](../engine/README.md).

## Checked and not checked

Checked on Linux, on a real machine: install, update, restart of a running
node on update, restart after the node was killed, nothing started after
`bitty stop`, a refused archive with a wrong checksum, uninstall. All of this
by running `bitty-update --update` by hand. Not checked: the systemd timer
itself firing, and the cron fallback.

Checked on Windows only on the GitHub runner, never on a real desktop: the
build, two nodes generating through each other, the leader restart, start,
restart and stop of the background node, install over a `bitty.exe` that is
in use, registering and removing the scheduled task. Not checked there: the
task actually firing in a logged-in session, and whether a node started by
the task survives the end of the task.
