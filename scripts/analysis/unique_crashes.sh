#!/usr/bin/env bash
# Germinator — deduplicate crashes in a run log by assertion / top line.
#
# Usage: ./scripts/analysis/unique_crashes.sh [logfile]
#   Defaults to $BUILD_OUT/run_state/run.log.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../build/env.sh" >/dev/null

LOG="${1:-$BUILD_OUT/run_state/run.log}"

if [[ ! -f "$LOG" ]]; then
    echo "Usage: $0 <logfile>   (default: $BUILD_OUT/run_state/run.log)" >&2
    exit 1
fi

echo "=== Unique crashes from $LOG ==="
echo ""

grep "CRASH LOG:.*cuda_tile_opt_fuzz_target:" "$LOG" \
    | sed 's/.*cuda_tile_opt_fuzz_target: //' \
    | sort -u \
    | while read -r msg; do
        COUNT=$(grep -cF "$msg" "$LOG")
        echo "[$COUNT hits] $msg"
    done \
    | sort -t'[' -k2 -rn

echo ""
TOTAL=$(grep -c "CRASH LOG:.*cuda_tile_opt_fuzz_target:" "$LOG" 2>/dev/null || echo 0)
UNIQUE=$(grep "CRASH LOG:.*cuda_tile_opt_fuzz_target:" "$LOG" \
    | sed 's/.*cuda_tile_opt_fuzz_target: //' | sort -u | wc -l)
echo "Total crashes:  $TOTAL"
echo "Unique crashes: $UNIQUE"
