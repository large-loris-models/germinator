#!/usr/bin/env bash
# Germinator — unified status dashboard.
#
# Shows: fuzzer health, unique crashes, ASAN oracle findings, summary.
#
# Usage: ./scripts/analysis/status.sh [run.log]
#   Default run log: $BUILD_OUT/run_state/run.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../build/env.sh" >/dev/null

RUN_LOG="${1:-$BUILD_OUT/run_state/run.log}"
PIDS_FILE="$BUILD_OUT/run_state/pids"
ORACLE_ROOT="$BUILD_OUT/oracle_results"
ASAN_DIR="$ORACLE_ROOT/asan_opt"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [[ -t 1 ]]; then
    C_BOLD=$'\e[1m'; C_DIM=$'\e[2m'; C_RESET=$'\e[0m'
    C_HDR=$'\e[1;36m'; C_GRN=$'\e[32m'
else
    C_BOLD=''; C_DIM=''; C_RESET=''; C_HDR=''; C_GRN=''
fi

bar() { printf '═%.0s' {1..72}; echo; }
hdr() { printf '\n%s%s%s\n' "$C_HDR" "$1" "$C_RESET"; bar; }

fmt_dur() {
    local s=$1 d h m r
    d=$((s/86400)); h=$(((s%86400)/3600)); m=$(((s%3600)/60)); r=$((s%60))
    if   (( d > 0 )); then printf '%dd %dh %dm' "$d" "$h" "$m"
    elif (( h > 0 )); then printf '%dh %dm %ds' "$h" "$m" "$r"
    else                   printf '%dm %ds' "$m" "$r"
    fi
}

parse_asan_log() {
    local f="$1" base summary loc frame
    base=$(basename "$f" .log)
    summary=$(grep -m1 '^SUMMARY:' "$f" 2>/dev/null || true)

    if [[ -z "$summary" ]]; then
        loc=$(grep -m1 'runtime error:' "$f" 2>/dev/null \
              | sed -E 's|.*/([^/]+:[0-9]+):[0-9]+: runtime error.*|\1|' || true)
        printf '%s\t%s\t%s\n' "unknown" "${loc:-?}" "$base"
        return
    fi

    if [[ "$summary" =~ ^SUMMARY:\ [A-Za-z]+:\ ([A-Za-z][A-Za-z0-9_-]+)\ .*\ in\ (.+)$ ]]; then
        printf '%s\t%s\t%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "$base"
        return
    fi

    if [[ "$summary" == *" leaked"* ]]; then
        frame=$(grep -E '^[[:space:]]+#[0-9]+ 0x[0-9a-f]+ in ' "$f" \
            | sed -E '
                s|^[[:space:]]+#[0-9]+ 0x[0-9a-f]+ in ||
                s| /[^ ]+:[0-9]+(:[0-9]+)?$||
                s| \(/[^)]*\) \(BuildId: [^)]*\)$||
              ' \
            | grep -vE '^(operator new|operator delete|malloc|calloc|realloc|__interceptor_)' \
            | head -1)
        frame=$(printf '%s' "${frame:-?}" | sed -E 's/\(.*$//')
        printf '%s\t%s\t%s\n' "memory-leak" "${frame:-?}" "$base"
        return
    fi

    if [[ "$summary" =~ ^SUMMARY:\ [A-Za-z]+:\ ([A-Za-z][A-Za-z0-9_-]+)\ (.+)$ ]]; then
        loc="${BASH_REMATCH[2]}"
        loc="${loc%"${loc##*[![:space:]]}"}"
        loc="${loc#*/llvm-project/}"
        printf '%s\t%s\t%s\n' "${BASH_REMATCH[1]}" "$loc" "$base"
        return
    fi

    printf '%s\t%s\t%s\n' "unknown" "?" "$base"
}

# ────────── Section 1: Fuzzer Status ──────────
hdr "Section 1: Fuzzer Status"

if [[ ! -f "$RUN_LOG" ]]; then
    echo "  run.log not found at $RUN_LOG — fuzzer not started"
else
    first_ts=$(grep -oE '^\[[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:+-]+\]' "$RUN_LOG" \
               | head -1 | tr -d '[]')

    is_running=0
    if [[ -f "$PIDS_FILE" ]]; then
        while IFS= read -r line; do
            pid="${line##*:}"
            [[ "$pid" =~ ^[0-9]+$ ]] || continue
            if kill -0 "$pid" 2>/dev/null; then is_running=1; break; fi
        done < "$PIDS_FILE"
    fi

    if (( is_running == 1 )); then
        end_epoch=$(date +%s)
        printf "  Status:           %s%s%s\n" "$C_GRN" "RUNNING" "$C_RESET"
    else
        end_epoch=$(stat -c %Y "$RUN_LOG" 2>/dev/null || date +%s)
        printf "  Status:           %s%s%s\n" "$C_DIM" "STOPPED" "$C_RESET"
    fi

    if [[ -n "$first_ts" ]]; then
        first_epoch=$(date -d "$first_ts" +%s 2>/dev/null || echo 0)
        if (( first_epoch > 0 )); then
            printf "  Runtime:          %s\n" "$(fmt_dur $((end_epoch - first_epoch)))"
        fi
    fi

    latest=$(grep -oE 'ft: [0-9]+ cov: [0-9]+ cnt: [0-9]+ cmp: [0-9]+' "$RUN_LOG" | tail -1)
    if [[ -n "$latest" ]]; then
        read -r _ ft _ cov _ cnt _ cmp <<<"$latest"
        printf "  Total features:   %s\n" "$ft"
        printf "  PC coverage:      %s\n" "$cov"
        printf "  Counter features: %s\n" "$cnt"
        printf "  Cmp features:     %s\n" "$cmp"
    fi

    avg_exec=$(awk '
        /\[S[0-9]+\..*exec\/s:/ {
            if (match($0, /\[S[0-9]+\./)) {
                shard = substr($0, RSTART+2, RLENGTH-3)
                for (i = 1; i <= NF; i++) if ($i == "exec/s:") last[shard] = $(i+1)+0
            }
        }
        END {
            n = 0; s = 0
            for (k in last) { s += last[k]; n++ }
            if (n > 0) printf "%.0f exec/s avg across %d shards", s/n, n
        }' "$RUN_LOG")
    [[ -n "$avg_exec" ]] && printf "  Throughput:       %s\n" "$avg_exec"

    if [[ -d "$CORPUS_DIR" ]]; then
        cs=$(find "$CORPUS_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
        printf "  Corpus size:      %d files\n" "$cs"
    fi
fi

# ────────── Section 2: Unique Crashes ──────────
hdr "Section 2: Unique Crashes (fuzzer)"

CRASHES_TSV="$TMP/crashes.tsv"
: > "$CRASHES_TSV"
total_crashes=0
unique_crashes=0
if [[ -f "$RUN_LOG" ]]; then
    grep "CRASH LOG:.*cuda_tile_opt_fuzz_target:" "$RUN_LOG" 2>/dev/null \
        | sed 's/.*cuda_tile_opt_fuzz_target: //' \
        | sort | uniq -c | sort -rn > "$CRASHES_TSV" || true
    total_crashes=$(grep -c "CRASH LOG:.*cuda_tile_opt_fuzz_target:" "$RUN_LOG" 2>/dev/null) \
        || total_crashes=0
    unique_crashes=$(wc -l < "$CRASHES_TSV") || unique_crashes=0
fi

if (( total_crashes == 0 )); then
    echo "  no crashes recorded"
else
    printf "  %5s  %s\n" "HITS" "ASSERTION"
    printf "  %5s  %s\n" "-----" "---------"
    while read -r count rest; do
        printf "  %5d  %s\n" "$count" "${rest:0:140}"
    done < "$CRASHES_TSV"
    echo
    printf "  Total: %d crashes, %d unique\n" "$total_crashes" "$unique_crashes"
fi

# ────────── Section 3: ASAN Findings ──────────
hdr "Section 3: ASAN Oracle Findings"

unique_asan=0
if [[ ! -d "$ASAN_DIR" ]]; then
    echo "  not started"
else
    asan_pass=$(find "$ASAN_DIR/pass"    -maxdepth 1 -type f 2>/dev/null | wc -l)
    asan_fail=$(find "$ASAN_DIR/fail"    -maxdepth 1 -type f -name '*.log' 2>/dev/null | wc -l)
    asan_to=$(find   "$ASAN_DIR/timeout" -maxdepth 1 -type f 2>/dev/null | wc -l)
    asan_err=$(find  "$ASAN_DIR/error"   -maxdepth 1 -type f 2>/dev/null | wc -l)
    asan_chk=0
    [[ -f "$ASAN_DIR/checked.log" ]] && asan_chk=$(wc -l < "$ASAN_DIR/checked.log")
    printf "  Progress: checked=%d  pass=%d  fail=%d  timeout=%d  error=%d\n\n" \
        "$asan_chk" "$asan_pass" "$asan_fail" "$asan_to" "$asan_err"

    if (( asan_fail == 0 )); then
        echo "  no failures"
    else
        ASAN_TSV="$TMP/asan.tsv"; : > "$ASAN_TSV"
        for f in "$ASAN_DIR/fail"/*.log; do
            [[ -e "$f" ]] || continue
            parse_asan_log "$f" >> "$ASAN_TSV"
        done

        ASAN_DEDUP="$TMP/asan_dedup.tsv"
        awk -F'\t' '
            { key = $1 "\t" $2; cnt[key]++; if (!(key in ex)) ex[key] = $3 }
            END { for (k in cnt) print cnt[k] "\t" k "\t" ex[k] }
        ' "$ASAN_TSV" | sort -rn > "$ASAN_DEDUP"
        unique_asan=$(wc -l < "$ASAN_DEDUP") || unique_asan=0

        printf "  %-22s  %-46s  %5s  %s\n" "TYPE" "TOP FRAME / LOCATION" "COUNT" "EXAMPLE"
        printf "  %-22s  %-46s  %5s  %s\n" "----" "--------------------" "-----" "-------"
        while IFS=$'\t' read -r count etype etop example; do
            printf "  %-22.22s  %-46.46s  %5d  %s\n" "$etype" "$etop" "$count" "$example"
        done < "$ASAN_DEDUP"
    fi
fi

# ────────── Section 4: Summary ──────────
hdr "Section 4: Summary"

total_uniq=$(( unique_crashes + unique_asan ))
printf "  %5d  fuzzer crashes  (unique by assertion)\n" "$unique_crashes"
printf "  %5d  ASAN findings   (unique by error type + top frame)\n" "$unique_asan"
echo
printf "  %sTotal unique findings: %d%s\n" "$C_BOLD" "$total_uniq" "$C_RESET"
echo
