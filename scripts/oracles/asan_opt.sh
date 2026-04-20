#!/usr/bin/env bash
# Germinator — ASAN oracle.
#
# Runs the ASAN-instrumented cuda-tile-opt on each corpus entry. Sanitizer
# diagnostics count as fail (new bugs); other non-zero exits are "error".

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ ! -x "$CT_OPT_ASAN" ]]; then
    oracle_log "ERROR: ASAN cuda-tile-opt not found: $CT_OPT_ASAN"
    exit 1
fi

CORPUS="${1:-$CORPUS_DIR}"

# Args to pass after the input file. Callers may override by exporting
# CT_OPT_ASAN_ARGS (space-separated).
read -r -a ASAN_ARGS <<< "${CT_OPT_ASAN_ARGS:---verify-each}"

oracle_init "asan_opt"

asan_check() {
    local input_file="$1"
    local output rc

    output="$(timeout "$ORACLE_TIMEOUT" "$CT_OPT_ASAN" "${ASAN_ARGS[@]}" "$input_file" -o /dev/null 2>&1)"
    rc=$?

    local verdict
    if (( rc == 0 )); then
        verdict="pass"
    elif (( rc == 124 )); then
        verdict="timeout"
    elif grep -qE 'AddressSanitizer|UndefinedBehavior|LeakSanitizer' <<< "$output"; then
        verdict="fail"
    else
        verdict="error"
    fi

    oracle_record_result "asan_opt" "$input_file" "$verdict" "$output"
}

oracle_watch_corpus "$CORPUS" asan_check
