#!/usr/bin/env bash
# Installs bitty on Linux from the prebuilt binaries and keeps it up to date.
#
#   curl -fsSL https://raw.githubusercontent.com/askscience/bitty/qwen-llama-cpp/scripts/install.sh | bash
#
# What it does:
#   - downloads `bitty` and the stage engine `bitty-stage` into ~/.local/share/bitty/bin
#   - links `bitty` into ~/.local/bin
#   - sets up a timer that looks for a new build every minute, installs it, restarts the
#     node if one was running, and starts the node again after a reboot
#
# Nothing here needs root. The same file is installed as `bitty-update`, which is what the
# timer runs.
#
#   install.sh                  install or reinstall
#   install.sh --no-auto-update install without the timer
#   install.sh --update         check once and update if there is a new build (the timer's job)
#   install.sh --uninstall      remove the binaries and the timer; models and settings stay
#
# Environment:
#   BITTY_HOME      where the binaries live      (default ~/.local/share/bitty)
#   BITTY_BIN_DIR   where the `bitty` link goes  (default ~/.local/bin)
#   BITTY_CHANNEL   release tag to follow        (default nightly)
#   BITTY_ARCHIVE   install from this local archive, no download and no checksum (for tests)
set -euo pipefail

REPO="askscience/bitty"
CHANNEL="${BITTY_CHANNEL:-nightly}"
BASE_URL="${BITTY_BASE_URL:-https://github.com/$REPO/releases/download/$CHANNEL}"
PREFIX="${BITTY_HOME:-$HOME/.local/share/bitty}"
BIN_DIR="${BITTY_BIN_DIR:-$HOME/.local/bin}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
INTERVAL_MINUTES=1
CRON_MARK="# bitty-update"

quiet=0
say() { [ "$quiet" = 1 ] || printf '%s\n' "$*"; }
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { printf 'bitty install: %s\n' "$*" >&2; exit 1; }

platform() {
    [ "$(uname -s)" = "Linux" ] || fail "this script is for Linux; on Windows use scripts/install.ps1"
    case "$(uname -m)" in
        x86_64 | amd64) echo "linux-x86_64" ;;
        *) fail "there is no prebuilt bitty for $(uname -m) yet; build it from source (see engine/README.md)" ;;
    esac
}

need() {
    for tool in "$@"; do
        command -v "$tool" > /dev/null 2>&1 || fail "'$tool' is needed and was not found; install it and run this again"
    done
}

fetch() { curl --fail --silent --show-error --location --retry 3 --retry-delay 2 "$1" --output "$2"; }

installed_version() { cat "$PREFIX/VERSION" 2> /dev/null || true; }

# Downloads the current build when it differs from the installed one and swaps the files in.
# Prints nothing and changes nothing when the installed build is the current one.
update_binaries() {
    local name archive work current latest
    name="bitty-$(platform)"
    work="$(mktemp -d)"
    WORK_DIR="$work"
    trap 'rm -rf "$WORK_DIR"' EXIT
    current="$(installed_version)"

    if [ -n "${BITTY_ARCHIVE:-}" ]; then
        archive="$BITTY_ARCHIVE"
    else
        fetch "$BASE_URL/VERSION" "$work/VERSION" || fail "cannot reach $BASE_URL"
        latest="$(tr -d '[:space:]' < "$work/VERSION")"
        [ -n "$latest" ] || fail "the release has an empty VERSION file"
        if [ "$latest" = "$current" ] && [ -x "$PREFIX/bin/bitty" ] && [ -x "$PREFIX/bin/bitty-stage" ] \
            && [ -x "$PREFIX/bin/bitty-stage-sse" ]; then
            return 1
        fi
        archive="$work/$name.tar.gz"
        fetch "$BASE_URL/$name.tar.gz" "$archive"
        fetch "$BASE_URL/SHA256SUMS" "$work/SHA256SUMS"
        # a build that is being published can have a new archive next to old checksums: refuse
        # it now, the next run sees the finished release
        (cd "$work" && grep " $name.tar.gz\$" SHA256SUMS | sha256sum --check --status) \
            || fail "the checksum of $name.tar.gz does not match; nothing was changed"
    fi

    mkdir -p "$work/new" "$PREFIX/bin"
    tar -xzf "$archive" -C "$work/new"
    for file in bitty bitty-stage install.sh VERSION; do
        [ -f "$work/new/$name/$file" ] || fail "the archive has no $file"
    done
    chmod +x "$work/new/$name/bitty" "$work/new/$name"/bitty-stage* "$work/new/$name/install.sh"
    "$work/new/$name/bitty" version > /dev/null || fail "the new bitty does not run on this machine; nothing was changed"

    # copy next to the target first, then rename: a running node keeps its old file and nobody
    # ever sees half a binary
    local pair source target
    local pairs=("bitty:bin/bitty" "bitty-stage:bin/bitty-stage" "install.sh:bin/bitty-update" "VERSION:VERSION")
    # the engines for older processors: bitty picks the one its machine can run
    local variant
    for variant in avx sse; do
        if [ -f "$work/new/$name/bitty-stage-$variant" ]; then
            pairs+=("bitty-stage-$variant:bin/bitty-stage-$variant")
        fi
    done
    for pair in "${pairs[@]}"; do
        source="$work/new/$name/${pair%%:*}"
        target="$PREFIX/${pair#*:}"
        cp -p "$source" "$target.new"
        mv -f "$target.new" "$target"
    done
    return 0
}

link_command() {
    mkdir -p "$BIN_DIR"
    ln -sfn "$PREFIX/bin/bitty" "$BIN_DIR/bitty"
}

have_user_systemd() {
    command -v systemctl > /dev/null 2>&1 && systemctl --user show-environment > /dev/null 2>&1
}

remove_cron() {
    command -v crontab > /dev/null 2>&1 || return 0
    crontab -l 2> /dev/null | grep -q "$CRON_MARK" || return 0
    crontab -l 2> /dev/null | grep -v "$CRON_MARK" | crontab -
}

install_timer() {
    if have_user_systemd; then
        mkdir -p "$UNIT_DIR"
        cat > "$UNIT_DIR/bitty-update.service" << EOF
[Unit]
Description=Update bitty and keep its node running
After=network-online.target

[Service]
Type=oneshot
ExecStart=$PREFIX/bin/bitty-update --update
# the node this starts must stay up when the update check is over
KillMode=process
EOF
        cat > "$UNIT_DIR/bitty-update.timer" << EOF
[Unit]
Description=Look for a new bitty build every $INTERVAL_MINUTES min

[Timer]
OnBootSec=1min
OnUnitActiveSec=${INTERVAL_MINUTES}min
AccuracySec=10s

[Install]
WantedBy=timers.target
EOF
        systemctl --user daemon-reload
        # restart, not only start: a timer that is already running keeps its old interval
        if ! { systemctl --user enable bitty-update.timer && systemctl --user restart bitty-update.timer; } > /dev/null 2>&1; then
            say "auto update: the systemd timer could not be enabled"
            say "  run '$PREFIX/bin/bitty-update --update' whenever you want the current build"
            return 0
        fi
        remove_cron
        # without lingering the timer only runs while this user is logged in
        if loginctl enable-linger "$(id -un)" > /dev/null 2>&1; then
            say "auto update: systemd timer every $INTERVAL_MINUTES min, also when nobody is logged in"
        else
            say "auto update: systemd timer every $INTERVAL_MINUTES min while you are logged in"
            say "  to keep it running after logout: sudo loginctl enable-linger $(id -un)"
        fi
    elif command -v crontab > /dev/null 2>&1; then
        remove_cron
        (
            crontab -l 2> /dev/null || true
            echo "*/$INTERVAL_MINUTES * * * * $PREFIX/bin/bitty-update --update $CRON_MARK"
            echo "@reboot $PREFIX/bin/bitty-update --update $CRON_MARK"
        ) | crontab -
        say "auto update: cron job every $INTERVAL_MINUTES min"
    else
        say "auto update: not set up, this machine has neither a systemd user session nor cron"
        say "  run '$PREFIX/bin/bitty-update --update' whenever you want the current build"
    fi
}

# An installation made by an older build keeps that build's timer. The updater brings it to
# the current interval; an installation without auto update is left alone.
refresh_timer() {
    if [ -f "$UNIT_DIR/bitty-update.timer" ]; then
        grep -q "^OnUnitActiveSec=${INTERVAL_MINUTES}min\$" "$UNIT_DIR/bitty-update.timer" && return 0
    elif command -v crontab > /dev/null 2>&1 && crontab -l 2> /dev/null | grep -q "$CRON_MARK"; then
        crontab -l 2> /dev/null | grep -q "^\*/$INTERVAL_MINUTES .*$CRON_MARK" && return 0
    else
        return 0
    fi
    log "setting the update check to every $INTERVAL_MINUTES minute(s)"
    install_timer || log "the timer could not be changed"
}

remove_timer() {
    if command -v systemctl > /dev/null 2>&1; then
        systemctl --user disable --now bitty-update.timer > /dev/null 2>&1 || true
        rm -f "$UNIT_DIR/bitty-update.service" "$UNIT_DIR/bitty-update.timer"
        systemctl --user daemon-reload > /dev/null 2>&1 || true
    fi
    remove_cron
}

run_install() {
    local auto_update="$1"
    need curl tar sha256sum
    if update_binaries; then
        say "installed bitty $(installed_version) in $PREFIX/bin"
        # a node that was running keeps the old binary until it is restarted
        "$PREFIX/bin/bitty" restart --if-started > /dev/null 2>&1 || true
    else
        say "bitty $(installed_version) is already the current build"
    fi
    link_command
    if [ "$auto_update" = 1 ]; then
        install_timer
    else
        remove_timer
        say "auto update: off; run '$PREFIX/bin/bitty-update --update' to update by hand"
    fi
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) say "note: $BIN_DIR is not in your PATH; add it, or call $BIN_DIR/bitty" ;;
    esac
    say ""
    say "Next:"
    say "  first machine:   bitty share home        (prints the invite)"
    say "  other machines:  bitty connect 'INVITE' --name home"
    say "  then, anywhere:  bitty run qwen3.5-35b-a3b \"hello\""
}

# What the timer runs. Quiet unless something happens: its output is the update log.
run_update() {
    quiet=1
    need curl tar sha256sum
    mkdir -p "$PREFIX"
    exec >> "$PREFIX/update.log" 2>&1
    # one run at a time: a slow download must not meet the next tick
    exec 9> "$PREFIX/update.lock"
    flock -n 9 2> /dev/null || exit 0
    if [ "$(wc -c < "$PREFIX/update.log")" -gt 1000000 ]; then
        tail -n 200 "$PREFIX/update.log" > "$PREFIX/update.log.tmp" && cat "$PREFIX/update.log.tmp" > "$PREFIX/update.log"
        rm -f "$PREFIX/update.log.tmp"
    fi
    local before
    before="$(installed_version)"
    # in a subshell: a failed download must not stop the node from being looked after below
    if (update_binaries); then
        log "updated ${before:-nothing} -> $(installed_version)"
        "$PREFIX/bin/bitty" restart --if-started 9>&- || log "restart failed"
    fi
    # after a reboot or a crash: bring back the node that was running. Does nothing when
    # bitty was stopped with `bitty stop` or never started.
    "$PREFIX/bin/bitty" start --if-started > /dev/null 9>&- || log "start failed"
    refresh_timer 9>&-
}

run_uninstall() {
    if [ -x "$PREFIX/bin/bitty" ]; then
        "$PREFIX/bin/bitty" stop > /dev/null 2>&1 || true
    fi
    remove_timer
    if [ "$(readlink "$BIN_DIR/bitty" 2> /dev/null)" = "$PREFIX/bin/bitty" ]; then
        rm -f "$BIN_DIR/bitty"
    fi
    rm -rf "$PREFIX"
    say "bitty was removed. Models and settings are still in ${BITTY_DATA_DIR:-$HOME/.bitty}."
}

main() {
    local action=install auto_update=1
    while [ $# -gt 0 ]; do
        case "$1" in
            --update) action=update ;;
            --uninstall) action=uninstall ;;
            --no-auto-update) auto_update=0 ;;
            -h | --help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
            *) fail "unknown argument: $1" ;;
        esac
        shift
    done
    case "$action" in
        install) run_install "$auto_update" ;;
        update) run_update ;;
        uninstall) run_uninstall ;;
    esac
}

main "$@"
