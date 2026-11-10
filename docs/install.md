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
was stopped with `bitty stop`: stopping by hand is remembered.

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
still do not fit it says how much room it has.

A node keeps its name in `node-id` inside its data directory, so after a
restart or an update the leader sees the same machine and not a new one.

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

- x86-64 only, and the processor needs AVX2 (Intel from 2013, AMD from 2015).
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
