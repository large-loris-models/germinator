#!/usr/bin/env bash
# =============================================================================
# run_fuzzer.sh — Launch Centipede against the cuda-tile fuzz target
# =============================================================================
# Prerequisites:
#   - link_fuzz_target.sh has produced $BUILD_OUT/mlir_fuzz_target
#   - setup_grammarinator.sh has populated $SEEDS_DIR/trees/ with .grtf files
# =============================================================================
source "$(dirname "$0")/../build/env.sh"

FUZZ_TARGET="$BUILD_OUT/cuda_tile_opt_fuzz_target"
TREES_DIR="$SEEDS_DIR/trees"
FUZZ_WORKDIR="$BUILD_OUT/workdir_$(date +%m%d%Y)"

echo "=== Running Centipede ==="

for f in "$FUZZ_TARGET" "$CENTIPEDE_BIN"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found" >&2
        exit 1
    fi
done

TREE_COUNT=$(find "$TREES_DIR" -maxdepth 1 -name '*.grtf' 2>/dev/null | wc -l)
if [[ "$TREE_COUNT" -eq 0 ]]; then
    echo "ERROR: no .grtf trees in $TREES_DIR" >&2
    echo "  Run setup_grammarinator.sh first." >&2
    exit 1
fi

mkdir -p "$FUZZ_WORKDIR" "$CORPUS_DIR"

# Seed the corpus from trees/ on first run only — preserves the source seeds
# while letting Centipede freely add/replace files in CORPUS_DIR over time.
if [[ -z "$(ls -A "$CORPUS_DIR" 2>/dev/null)" ]]; then
    echo "[run] seeding corpus from $TREES_DIR ..."
    cp "$TREES_DIR"/*.grtf "$CORPUS_DIR/"
    echo "[run] copied $(ls "$CORPUS_DIR" | wc -l) trees"
fi

JOBS=${FUZZ_JOBS:-4}

echo "[run] target:    $FUZZ_TARGET"
echo "[run] workdir:   $FUZZ_WORKDIR"
echo "[run] corpus:    $CORPUS_DIR ($(ls "$CORPUS_DIR" | wc -l) files)"
echo "[run] jobs:      $JOBS"

FLAGS=(
    --binary="$FUZZ_TARGET"
    --workdir="$FUZZ_WORKDIR"
    --j="$JOBS"
    --timeout_per_input=30
    --rss_limit_mb=8192
    --address_space_limit_mb=0
    --require_seeds=true
    --corpus_dir="$CORPUS_DIR"
    --use_counter_features
    --v=1
    --max_num_crash_reports=50000
    --max_len=1000000
)

echo ""
echo "[run] $CENTIPEDE_BIN ${FLAGS[*]} $*"
echo ""

ulimit -s unlimited
GRAMMARINATOR_MAX_DEPTH=50 "$CENTIPEDE_BIN" "${FLAGS[@]}" "$@"