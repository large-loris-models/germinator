#!/usr/bin/env bash
# Germinator — clean shutdown for the fuzzing pipeline started by start.sh.
#
# Usage: ./scripts/run/stop.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../build/env.sh" >/dev/null

RUN_STATE="$BUILD_OUT/run_state"
PIDS_FILE="$RUN_STATE/pids"

log() { echo "[$(date -Is)] [stop] $*" >&2; }

if [[ ! -f "$PIDS_FILE" ]]; then
    log "No pids file at $PIDS_FILE — nothing to stop."
    exit 0
fi

START_TIME="$(stat -c %Y "$PIDS_FILE" 2>/dev/null || date +%s)"

# ── SIGTERM all tracked pids ────────────────────────────────────────────────

declare -a LIVE_PIDS=()
while IFS=: read -r name pid; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
        log "Sending SIGTERM to $name (PID $pid)"
        kill -TERM "$pid" 2>/dev/null || true
        LIVE_PIDS+=("$name:$pid")
    else
        log "$name (PID $pid) already exited"
    fi
done < "$PIDS_FILE"

# ── Wait up to 10s, then SIGKILL ────────────────────────────────────────────

if (( ${#LIVE_PIDS[@]} > 0 )); then
    deadline=$(( $(date +%s) + 10 ))
    for entry in "${LIVE_PIDS[@]}"; do
        name="${entry%%:*}"
        pid="${entry#*:}"
        while kill -0 "$pid" 2>/dev/null && (( $(date +%s) < deadline )); do
            sleep 0.5
        done
        if kill -0 "$pid" 2>/dev/null; then
            log "Force-killing $name (PID $pid)"
            kill -KILL "$pid" 2>/dev/null || true
        else
            log "$name (PID $pid) exited"
        fi
    done
fi

# ── Summary ────────────────────────────────────────────────────────────────

now="$(date +%s)"
elapsed=$(( now - START_TIME ))
h=$(( elapsed / 3600 )); m=$(( (elapsed % 3600) / 60 )); s=$(( elapsed % 60 ))

corpus_count=0
if [[ -d "$CORPUS_DIR" ]]; then
    corpus_count="$(find "$CORPUS_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)"
fi

crash_count=0
workdir="$(ls -1dt "$BUILD_OUT"/workdir_*/crashes* 2>/dev/null | head -n1)"
if [[ -n "$workdir" ]]; then
    crash_count="$(find "$workdir" -maxdepth 1 -type f 2>/dev/null | wc -l)"
fi

log "────────────────────────────────────────"
log "Runtime:            ${h}h ${m}m ${s}s"
log "Corpus entries:     $corpus_count"
log "Crashes found:      $crash_count"
log "────────────────────────────────────────"

rm -f "$PIDS_FILE"
log "Cleaned up $RUN_STATE"
log "Stopped."
