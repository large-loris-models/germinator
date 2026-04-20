#!/usr/bin/env bash
# Germinator — single entry point for the full fuzzing pipeline:
#   Centipede (sancov cuda-tile-opt)  +  ASAN oracle (asan cuda-tile-opt)
#
# Usage:
#   nohup ./scripts/run/start.sh > build/run.log 2>&1 &
#   cat build/run_state/pids
#   ./scripts/run/stop.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../build/env.sh" >/dev/null

# ── State dir ───────────────────────────────────────────────────────────────

RUN_STATE="$BUILD_OUT/run_state"
PIDS_FILE="$RUN_STATE/pids"
RUN_LOG="$RUN_STATE/run.log"
START_TIME="$(date +%s)"

mkdir -p "$RUN_STATE"
: > "$PIDS_FILE"
: > "$RUN_LOG"

log() { echo "[$(date -Is)] [start] $*" | tee -a "$RUN_LOG" >&2; }

# ── Detect available cores ──────────────────────────────────────────────────

detect_cores() {
    local cpus=""
    if [[ -r /sys/fs/cgroup/cpuset.cpus.effective ]]; then
        cpus="$(cat /sys/fs/cgroup/cpuset.cpus.effective)"
    elif [[ -r /sys/fs/cgroup/cpuset/cpuset.cpus ]]; then
        cpus="$(cat /sys/fs/cgroup/cpuset/cpuset.cpus)"
    fi
    if [[ -z "$cpus" ]]; then
        cpus="0-$(($(nproc) - 1))"
    fi
    local out=() part lo hi
    IFS=',' read -ra parts <<< "$cpus"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            lo="${part%-*}"; hi="${part#*-}"
            for ((i=lo; i<=hi; i++)); do out+=("$i"); done
        else
            out+=("$part")
        fi
    done
    printf '%s\n' "${out[@]}"
}

mapfile -t ALL_CORES < <(detect_cores)
FUZZER_CORES="${FUZZ_JOBS:-4}"

# Layout: FUZZER_CORES for fuzzer + 1 for ASAN oracle.
NEEDED=$(( FUZZER_CORES + 1 ))
if (( ${#ALL_CORES[@]} < NEEDED )); then
    log "WARNING: only ${#ALL_CORES[@]} cores available (need $NEEDED); oracle may share"
fi
ASAN_CORE="${ALL_CORES[$FUZZER_CORES]:-${ALL_CORES[0]}}"

# ── Check required binaries ─────────────────────────────────────────────────

FUZZ_TARGET="$BUILD_OUT/cuda_tile_opt_fuzz_target"

err=0
for pair in "fuzz target:$FUZZ_TARGET" "ASAN cuda-tile-opt:$CT_OPT_ASAN" "centipede:$CENTIPEDE_BIN"; do
    label="${pair%%:*}"
    path="${pair#*:}"
    if [[ ! -x "$path" ]]; then
        log "ERROR: $label not found or not executable: $path"
        err=1
    fi
done
if (( err )); then
    log "Aborting — missing binaries. Run the build scripts first."
    exit 1
fi

# ── Helpers ────────────────────────────────────────────────────────────────

record_pid() {
    local name="$1" pid="$2"
    echo "$name:$pid" >> "$PIDS_FILE"
}

print_summary() {
    local now elapsed h m s
    now="$(date +%s)"
    elapsed=$(( now - START_TIME ))
    h=$(( elapsed / 3600 )); m=$(( (elapsed % 3600) / 60 )); s=$(( elapsed % 60 ))

    local corpus_count=0
    if [[ -d "$CORPUS_DIR" ]]; then
        corpus_count="$(find "$CORPUS_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)"
    fi

    local crash_count=0 workdir
    workdir="$(ls -1dt "$BUILD_OUT"/workdir_*/crashes* 2>/dev/null | head -n1)"
    if [[ -n "$workdir" ]]; then
        crash_count="$(find "$workdir" -maxdepth 1 -type f 2>/dev/null | wc -l)"
    fi

    log "────────────────────────────────────────"
    log "Runtime:            ${h}h ${m}m ${s}s"
    log "Corpus entries:     $corpus_count"
    log "Crashes found:      $crash_count"
    log "────────────────────────────────────────"
}

shutdown() {
    log "Shutting down..."
    local pids=()
    if [[ -f "$PIDS_FILE" ]]; then
        while IFS=: read -r name pid; do
            [[ -n "$pid" ]] || continue
            if kill -0 "$pid" 2>/dev/null; then
                log "Sending SIGTERM to $name (PID $pid)"
                kill -TERM "$pid" 2>/dev/null || true
                pids+=("$pid")
            fi
        done < "$PIDS_FILE"
    fi
    if (( ${#pids[@]} > 0 )); then
        local deadline=$(( $(date +%s) + 5 ))
        for pid in "${pids[@]}"; do
            while kill -0 "$pid" 2>/dev/null && (( $(date +%s) < deadline )); do
                sleep 0.5
            done
            if kill -0 "$pid" 2>/dev/null; then
                log "Force-killing PID $pid"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
    print_summary
    log "Stopped."
    exit 0
}
trap shutdown INT TERM

# ── Start Centipede fuzzer ──────────────────────────────────────────────────

FUZZ_WORKDIR="$BUILD_OUT/workdir_$(date +%m%d%Y)"
mkdir -p "$FUZZ_WORKDIR" "$CORPUS_DIR"

# Copy seeds into corpus dir if corpus is empty.
if [[ -d "$SEEDS_DIR" ]] && [[ -z "$(ls -A "$CORPUS_DIR" 2>/dev/null)" ]]; then
    log "Copying seeds into corpus dir..."
    find "$SEEDS_DIR" -maxdepth 1 -type f -print0 | xargs -0 -r cp -t "$CORPUS_DIR/"
    log "Copied $(find "$CORPUS_DIR" -maxdepth 1 -type f | wc -l) seed files"
fi

FUZZER_FLAGS=(
    --binary="$FUZZ_TARGET"
    --workdir="$FUZZ_WORKDIR"
    --j="$FUZZER_CORES"
    --timeout_per_input=10
    --rss_limit_mb=8142
    --address_space_limit_mb=0
    --corpus_dir="$CORPUS_DIR"
    -crossover_level=0
    --use_counter_features
    --v=1
    --max_num_crash_reports=50000
)

log "Starting Centipede fuzzer (${FUZZER_CORES} jobs)..."
ulimit -s unlimited
"$CENTIPEDE_BIN" "${FUZZER_FLAGS[@]}" >> "$RUN_LOG" 2>&1 &
FUZZER_PID=$!
record_pid "fuzzer" "$FUZZER_PID"
log "Fuzzer PID: $FUZZER_PID"

sleep 5
if ! kill -0 "$FUZZER_PID" 2>/dev/null; then
    log "ERROR: fuzzer exited immediately — check $RUN_LOG"
    exit 1
fi

# ── Start ASAN oracle ───────────────────────────────────────────────────────

ORACLE_DIR="$SCRIPT_DIR/../oracles"

log "Starting ASAN oracle on core $ASAN_CORE..."
taskset -c "$ASAN_CORE" "$ORACLE_DIR/asan_opt.sh" "$CORPUS_DIR" >> "$RUN_LOG" 2>&1 &
ASAN_PID=$!
record_pid "asan_opt" "$ASAN_PID"
log "ASAN oracle PID: $ASAN_PID (core $ASAN_CORE)"

# ── Banner ──────────────────────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════"
log "  Germinator fuzzing pipeline running"
log "═══════════════════════════════════════════════"
log "  fuzzer      PID $FUZZER_PID   cores ${ALL_CORES[*]:0:$FUZZER_CORES}"
log "  asan_opt    PID $ASAN_PID   core  $ASAN_CORE"
log "───────────────────────────────────────────────"
log "  corpus:     $CORPUS_DIR"
log "  workdir:    $FUZZ_WORKDIR"
log "  pids file:  $PIDS_FILE"
log "  log:        $RUN_LOG"
log "═══════════════════════════════════════════════"
log ""
log "Stop with:  ./scripts/run/stop.sh"
log ""

wait
