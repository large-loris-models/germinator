#!/usr/bin/env bash
# =============================================================================
# asan_watcher.sh — Background loop that runs corpus trees through the ASAN
# build of cuda-tile-opt and saves deduped crashes.
#
# Reads inputs from:
#   - $CORPUS_DIR/*                          (Centipede-exported trees)
#   - $BUILD_OUT/workdir_*/crashes.*/ *      (Centipede's own crash inputs,
#                                             re-checked under ASAN)
#
# Saves crashes to:
#   - $BUILD_OUT/asan_crashes/<short-hash>/{input.mlir,asan.log,meta.json}
#
# State (shadow file + crash fingerprints):
#   - $BUILD_OUT/asan_watcher_state/
#
# Tuning knobs (env):
#   WATCHER_PERIOD     seconds between batches  (default 60)
#   WATCHER_TIMEOUT    per-input timeout        (default 30)
#   WATCHER_MAX_BYTES  skip inputs larger than this (default 10485760)
#
# Start it alongside the fuzzer:
#   ./scripts/run/asan_watcher.sh &
# =============================================================================
source "$(dirname "$0")/../build/env.sh"

if [[ ! -x "$CT_OPT_ASAN" ]]; then
    echo "ERROR: ASAN cuda-tile-opt not found at $CT_OPT_ASAN" >&2
    exit 1
fi

export WATCHER_PERIOD="${WATCHER_PERIOD:-60}"
export WATCHER_TIMEOUT="${WATCHER_TIMEOUT:-30}"
export WATCHER_MAX_BYTES="${WATCHER_MAX_BYTES:-10485760}"

exec uv run python "$PROJECT_ROOT/scripts/run/asan_watcher.py"
