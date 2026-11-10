#!/usr/bin/env bash
# End-to-end check of a built bitty and stage engine, used by the release workflow on Linux
# and on Windows (Git Bash) before anything is published.
#
#   scripts/ci-smoke.sh <bitty> <bitty-stage> <small model.gguf>
#
# It starts two nodes on this machine, generates through both, restarts the leader and
# generates again (the worker has to register again by itself), then drives the background
# runtime the way the updater does: start, restart, stop.
set -euo pipefail

native() { if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }

BITTY="$1"
MODEL="$(native "$3")"
BITTY_STAGE_BIN="$(native "$2")"
export BITTY_STAGE_BIN BITTY_STAGE_THREADS=2
export BITTY_DRAFT_MODEL="$MODEL" BITTY_DRAFT_THREADS=2 BITTY_TIMING=1
WORK="$(mktemp -d)"
DATA="$(native "$WORK")"
PIDS=()

cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2> /dev/null || true; done
    "$BITTY" stop --data-dir "$DATA/runtime" > /dev/null 2>&1 || true
    sleep 1
    rm -rf "$WORK" 2> /dev/null || true
}
trap cleanup EXIT

show_logs() { for log in "$WORK"/*.log; do echo "--- $log"; tail -n 40 "$log"; done; }
die() { echo "FAILED: $*" >&2; show_logs >&2; exit 1; }

"$BITTY" version

leader() {
    "$BITTY" node --no-iroh --model "$MODEL" --layers 24 --listen 127.0.0.1:50551 --worker-listen 127.0.0.1:50561 \
        --node-id n0 --cluster-token tok --data-dir "$DATA/n0" >> "$WORK/n0.log" 2>&1 &
    LEADER=$!
    PIDS+=("$LEADER")
}
generate() {
    "$BITTY" generate --node 127.0.0.1:50551 --cluster-token tok --data-dir "$DATA/client" \
        --prompt "Name three colours." --seed 7 --max-tokens 24 2> "$WORK/client.log"
}
# the nodes need a moment to load and to find each other: ask until both layer ranges answer
generate_when_ready() {
    local text
    for _ in $(seq 1 30); do
        if text="$(generate)" && [ -n "$text" ] && grep -q "joined scheduler" "$WORK/n1.log"; then
            echo "$text"
            return 0
        fi
        sleep 2
    done
    return 1
}

echo "== two nodes"
leader
"$BITTY" node --no-iroh --model "$MODEL" --layers 24 --join 127.0.0.1:50551 --listen 127.0.0.1:50571 \
    --node-id n1 --cluster-token tok --data-dir "$DATA/n1" > "$WORK/n1.log" 2>&1 &
PIDS+=("$!")
first="$(generate_when_ready)" || die "no text from two nodes"
echo "$first"

echo "== draft model"
# the model is its own draft here: nearly every guess is right, so a pass yields several tokens
drafted="$("$BITTY" generate --node 127.0.0.1:50551 --cluster-token tok --data-dir "$DATA/client" \
    --prompt "Name three colours." --seed 7 --max-tokens 24 --draft 4 2>> "$WORK/client.log")" \
    || die "no text with the draft model"
echo "$drafted"
[ -n "$drafted" ] || die "empty text with the draft model"
grep -q "draft model loaded" "$WORK/n0.log" || die "the leader did not load the draft model"
kept="$(grep -o "[0-9]* of [0-9]* draft tokens kept" "$WORK/n0.log" | tail -n 1)"
echo "$kept"
[ "${kept%% *}" -ge 8 ] || die "the workers kept too few draft tokens: $kept"

echo "== leader restart"
kill "$LEADER"
sleep 3
leader
sleep 10
second="$(generate_when_ready)" || die "no text after the leader restarted"
[ "$(grep -c "joined scheduler" "$WORK/n1.log")" -ge 2 ] || die "the worker did not register again"
[ "$first" = "$second" ] || die "the text changed after the restart: '$first' / '$second'"
for pid in "${PIDS[@]}"; do kill "$pid" 2> /dev/null || true; done
PIDS=()
sleep 3

echo "== background runtime"
state="$WORK/runtime/runtime/runtime.toml"
pid_of() { grep -o 'pid = [0-9]*' "$state" | grep -o '[0-9]*'; }
"$BITTY" start --if-started --data-dir "$DATA/runtime"
[ ! -f "$state" ] || die "start --if-started started a node that was never asked for"
"$BITTY" start --model "$MODEL" --data-dir "$DATA/runtime" || die "bitty start"
one="$(pid_of)"
"$BITTY" start --if-started --data-dir "$DATA/runtime" | grep -q "running" || die "the node is not running after start"
[ "$(pid_of)" = "$one" ] || die "start replaced a node that was running"
"$BITTY" restart --if-started --data-dir "$DATA/runtime" || die "bitty restart"
two="$(pid_of)"
[ "$one" != "$two" ] || die "restart kept the old node"
"$BITTY" start --if-started --data-dir "$DATA/runtime" | grep -q "running" || die "the node is not running after restart"
"$BITTY" stop --data-dir "$DATA/runtime" || die "bitty stop"
[ ! -f "$state" ] || die "stop left the saved state behind"
echo "ok"
