#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# ── Paths ───────────────────────────────────────────────────────────────────
GRAMMAR_FILE="$PROJECT_ROOT/resources/mlir.g4"
CORPUS_DIR="$SEEDS_DIR/generic"
GENERATED_DIR="$PROJECT_ROOT/generated"
TREES_DIR="$SEEDS_DIR/trees"
GRTF_DIR="$SEEDS_DIR/grtf"
GRAMMARINATOR_CXX="$GRAMMARINATOR_SRC/grammarinator-cxx"
CORPUS_COUNT=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.mlir' | wc -l)
echo "[setup] corpus has $CORPUS_COUNT .mlir files"

echo "============================================================"
echo " Step 3: Fuzzer setup"
echo "   Grammar:   $GRAMMAR_FILE"
echo "   Corpus:    $CORPUS_DIR"
echo "   Generated: $GENERATED_DIR"
echo "   GRTF out:  $GRTF_DIR"
echo "============================================================"

# ── Verify prerequisites ────────────────────────────────────────────────────
if [[ ! -f "$GRAMMAR_FILE" ]]; then
    echo "ERROR: grammar not found at $GRAMMAR_FILE" >&2
    exit 1
fi

if [[ ! -d "$CORPUS_DIR" ]] || [[ -z "$(ls -A "$CORPUS_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: corpus is empty or missing at $CORPUS_DIR" >&2
    echo "  Run collect_upstream_seeds.py and organize_seeds.sh first." >&2
    exit 1
fi

if [[ ! -d "$GRAMMARINATOR_CXX" ]]; then
    echo "ERROR: grammarinator-cxx not found at $GRAMMARINATOR_CXX" >&2
    exit 1
fi

for cmd in grammarinator-process grammarinator-parse conan; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found on PATH" >&2
        exit 1
    fi
done

echo ""
echo "────────────────────────────────────────────────────────────"
echo " Building MLIR grammarinator generator"
echo "────────────────────────────────────────────────────────────"

mkdir -p "$GENERATED_DIR"

GENERATOR_HPP="$GENERATED_DIR/mlirGenerator.hpp"

if [[ -f "$GENERATOR_HPP" ]] && [[ "$GENERATOR_HPP" -nt "$GRAMMAR_FILE" ]]; then
    echo "[setup] generator up to date, skipping"
else
    grammarinator-process --language hpp "$GRAMMAR_FILE" -o "$GENERATED_DIR"
    if [[ ! -f "$GENERATOR_HPP" ]]; then
        echo "ERROR: grammarinator-process did not produce mlirGenerator.hpp" >&2
        exit 1
    fi
    echo "[setup] generated $GENERATOR_HPP"
fi


echo ""
echo "────────────────────────────────────────────────────────────"
echo " Building grammarinator C++ runtime"
echo "────────────────────────────────────────────────────────────"


GRTF_MLIR_LIB="$GRAMMARINATOR_CXX/build/lib/libgrtf-mlir.a"

if [[ -f "$GRTF_MLIR_LIB" ]] && [[ "$GRTF_MLIR_LIB" -nt "$GENERATOR_HPP" ]]; then
    echo "[setup] grammarinator runtime up to date, skipping"
else
    conan profile detect --force >/dev/null 2>&1 || true
    python3 "$GRAMMARINATOR_CXX/dev/build.py" \
        --generator mlirGenerator \
        --includedir "$GENERATED_DIR" \
        --grlf
    echo "[setup] grammarinator C++ runtime built"
fi

echo ""
echo "────────────────────────────────────────────────────────────"
echo " Parsing corpus into GRTF trees"
echo "────────────────────────────────────────────────────────────"

mkdir -p "$TREES_DIR"

EXPECTED=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.mlir' | wc -l)
ACTUAL=$(find "$TREES_DIR" -maxdepth 1 -name '*.grtf' | wc -l)

NEED_PARSE=1
if [[ "$ACTUAL" -ge "$EXPECTED" ]]; then
    NEWEST_MLIR=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.mlir' -printf '%T@\n' | sort -n | tail -1)
    OLDEST_GRTF=$(find "$TREES_DIR" -maxdepth 1 -name '*.grtf' -printf '%T@\n' | sort -n | head -1)
    if [[ -n "$NEWEST_MLIR" ]] && [[ -n "$OLDEST_GRTF" ]] && \
       awk -v a="$NEWEST_MLIR" -v b="$OLDEST_GRTF" 'BEGIN { exit !(a <= b) }'; then
        NEED_PARSE=0
    fi
fi

if [[ "$NEED_PARSE" -eq 0 ]]; then
    echo "[setup] trees up to date ($ACTUAL trees), skipping"
else
    grammarinator-parse \
        -g "$GRAMMAR_FILE" \
        -r start \
        -o "$TREES_DIR/" \
    	-j "$(nproc)" \
        "$CORPUS_DIR"/*.mlir \
        --tree-format flatbuffers
fi

TREE_COUNT=$(find "$TREES_DIR" -maxdepth 1 -name '*.grtf' | wc -l)
echo "[setup] $TREE_COUNT trees in $TREES_DIR"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete"
echo "   Corpus:    $CORPUS_DIR ($CORPUS_COUNT .mlir files)"
echo "   Trees:     $TREES_DIR ($TREE_COUNT .grtf files)"
echo "   Generator: $GENERATED_DIR/mlirGenerator.hpp"
echo "============================================================"