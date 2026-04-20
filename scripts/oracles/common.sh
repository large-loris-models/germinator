#!/usr/bin/env bash
# Germinator — shared oracle library.
#
# Each oracle processes files from a corpus dir, classifies the result
# (pass / fail / timeout / error), and stores outputs under
# $BUILD_OUT/oracle_results/<oracle>/.

# shellcheck disable=SC2155

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../build/env.sh" >/dev/null

ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-30}"
ORACLE_RESULTS_ROOT="$BUILD_OUT/oracle_results"

# Sharding: each shard processes a deterministic subset of files.
ORACLE_SHARD_ID="${ORACLE_SHARD_ID:-0}"
ORACLE_TOTAL_SHARDS="${ORACLE_TOTAL_SHARDS:-1}"

# Populated by oracle_init.
ORACLE_NAME=""
ORACLE_DIR=""
ORACLE_CHECKED_LOG=""
declare -A ORACLE_CHECKED
declare -A ORACLE_FAIL_HASHES

oracle_log() {
    local name="${ORACLE_NAME:-oracle}"
    echo "[$(date -Is)] [$name] $*" >&2
}

oracle_init() {
    local name="$1"
    if (( ORACLE_TOTAL_SHARDS > 1 )); then
        name="${name}_${ORACLE_SHARD_ID}"
    fi
    ORACLE_NAME="$name"
    ORACLE_DIR="$ORACLE_RESULTS_ROOT/$name"
    ORACLE_CHECKED_LOG="$ORACLE_DIR/checked.log"

    mkdir -p "$ORACLE_DIR"/{pass,fail,timeout,error}
    touch "$ORACLE_CHECKED_LOG"

    # Load checked set into associative array for O(1) lookup.
    ORACLE_CHECKED=()
    local line
    while IFS= read -r line; do
        [[ -n "$line" ]] && ORACLE_CHECKED["$line"]=1
    done < "$ORACLE_CHECKED_LOG"

    # Load existing fail hashes for dedup.
    ORACLE_FAIL_HASHES=()
    local h
    for h in "$ORACLE_DIR/fail"/*.hash; do
        [[ -e "$h" ]] || continue
        ORACLE_FAIL_HASHES["$(basename "$h" .hash)"]=1
    done

    oracle_log "init results=$ORACLE_DIR timeout=${ORACLE_TIMEOUT}s checked=${#ORACLE_CHECKED[@]}"
}

_oracle_process_one() {
    local input_file="$1"
    local callback_fn="$2"
    local base
    base="$(basename "$input_file")"

    # Skip pre-existing seed files from the on-disk seed tree — fuzzer-
    # generated corpus entries are hash-named without an extension.
    if [[ "$base" == *.mlir ]]; then
        return 0
    fi

    if (( ORACLE_TOTAL_SHARDS > 1 )); then
        local hash
        hash=$(printf '%s' "$base" | cksum | awk '{print $1}')
        if (( hash % ORACLE_TOTAL_SHARDS != ORACLE_SHARD_ID )); then
            return 0
        fi
    fi

    if [[ -n "${ORACLE_CHECKED[$base]:-}" ]]; then
        return 0
    fi

    "$callback_fn" "$input_file" || true

    ORACLE_CHECKED["$base"]=1
    printf '%s\n' "$base" >> "$ORACLE_CHECKED_LOG"
}

oracle_iter_corpus() {
    local corpus_dir="$1"
    local callback_fn="$2"

    if [[ ! -d "$corpus_dir" ]]; then
        oracle_log "ERROR: corpus dir not found: $corpus_dir"
        return 1
    fi

    local f
    while IFS= read -r -d '' f; do
        _oracle_process_one "$f" "$callback_fn"
    done < <(find "$corpus_dir" -maxdepth 1 -type f -print0 2>/dev/null)
}

oracle_watch_corpus() {
    local corpus_dir="$1"
    local callback_fn="$2"

    oracle_iter_corpus "$corpus_dir" "$callback_fn"

    if command -v inotifywait >/dev/null 2>&1; then
        oracle_log "watching $corpus_dir (inotify)"
        local f
        while IFS= read -r f; do
            _oracle_process_one "$corpus_dir/$f" "$callback_fn"
        done < <(inotifywait -m -q -e close_write --format '%f' "$corpus_dir")
    else
        oracle_log "inotifywait not found; polling every 10s"
        while true; do
            sleep 10
            oracle_iter_corpus "$corpus_dir" "$callback_fn"
        done
    fi
}

oracle_record_result() {
    local oracle_name="$1"
    local input_file="$2"
    local verdict="$3"
    local output="${4:-}"

    local dir="$ORACLE_RESULTS_ROOT/$oracle_name/$verdict"
    local base
    base="$(basename "$input_file")"

    case "$verdict" in
        pass|timeout|error)
            cp -n "$input_file" "$dir/$base" 2>/dev/null || true
            ;;
        fail)
            local hash
            hash="$(printf '%s' "$output" | head -n 5 | sha1sum | awk '{print $1}')"
            if [[ -n "${ORACLE_FAIL_HASHES[$hash]:-}" ]]; then
                return 0
            fi
            ORACLE_FAIL_HASHES["$hash"]=1
            cp "$input_file" "$dir/$base"
            printf '%s' "$output" > "$dir/$base.log"
            : > "$dir/$hash.hash"
            oracle_log "FAIL $base hash=$hash"
            ;;
        *)
            oracle_log "unknown verdict: $verdict"
            return 1
            ;;
    esac
}
