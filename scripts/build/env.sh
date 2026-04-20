#!/usr/bin/env bash
# Germinator — shared environment for all build/run scripts.
#
# Source this file from other scripts:
#     source "$(dirname "$0")/env.sh"
#     check_prereqs

set -euo pipefail

# ── Project layout ──────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT

export PATH="$HOME/.local/bin:$PATH"

export DEPS_DIR="$PROJECT_ROOT/deps"
export BUILD_OUT="$PROJECT_ROOT/build"
export CORPUS_DIR="$PROJECT_ROOT/corpus"
export SEEDS_DIR="$PROJECT_ROOT/seeds"

# ── LLVM / MLIR ─────────────────────────────────────────────────────────────

export LLVM_SRC="$DEPS_DIR/llvm-project"

export LLVM_BUILD_SANCOV="$DEPS_DIR/llvm-build-sancov"
export LLVM_BUILD_ASAN="$DEPS_DIR/llvm-build-asan"
export LLVM_BUILD_PLAIN="$DEPS_DIR/llvm-build-plain"

export LLVM_INSTALL_SANCOV="$DEPS_DIR/llvm-install-sancov"
export LLVM_INSTALL_ASAN="$DEPS_DIR/llvm-install-asan"
export LLVM_INSTALL_PLAIN="$DEPS_DIR/llvm-install-plain"

# ── cuda-tile ───────────────────────────────────────────────────────────────

export CUDA_TILE_REPO="https://github.com/NVIDIA/cuda-tile.git"
export CUDA_TILE_SRC="$DEPS_DIR/cuda-tile"
export CT_BUILD_SANCOV="$DEPS_DIR/cuda-tile-build-sancov"
export CT_BUILD_ASAN="$DEPS_DIR/cuda-tile-build-asan"
export CT_OPT_SANCOV="$CT_BUILD_SANCOV/bin/cuda-tile-opt"
export CT_OPT_ASAN="$CT_BUILD_ASAN/bin/cuda-tile-opt"

# ── Centipede (built from source, self-contained) ───────────────────────────

export FUZZTEST_SRC="$PROJECT_ROOT/third_party/fuzztest"
export CENTIPEDE_BIN="$FUZZTEST_SRC/bazel-bin/centipede/centipede"
export CENTIPEDE_RUNNER="$FUZZTEST_SRC/bazel-bin/centipede/centipede_runner_no_main.a"

# ── Grammarinator ───────────────────────────────────────────────────────────

export GRAMMARINATOR_SRC="$PROJECT_ROOT/third_party/grammarinator"

# ── Toolchain & flags ───────────────────────────────────────────────────────

export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"

export SANCOV_FLAGS="-fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp"

export FUZZ_CFLAGS=(
    "-g"
    "-O2"
    "-fno-omit-frame-pointer"
    "-gline-tables-only"
    "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION"
    "-fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp"
)

export FUZZ_JOBS="${FUZZ_JOBS:-4}"
export NUM_JOBS="${NUM_JOBS:-$(nproc)}"

# Symbolizer (for sanitizer stack traces)
export LLVM_SYMBOLIZER_PATH="${LLVM_SYMBOLIZER_PATH:-/usr/bin/llvm-symbolizer}"

# ── Prereq check ────────────────────────────────────────────────────────────

check_prereqs() {
    local missing=0
    for tool in cmake ninja "$CC" "$CXX" lld bazel z3; do
        if ! command -v "$tool" &>/dev/null; then
            echo "ERROR: required tool not found: $tool" >&2
            missing=1
        fi
    done
    if (( missing )); then
        echo "Run scripts/build/bootstrap.sh to install prerequisites." >&2
        return 1
    fi
    echo "[env] PROJECT_ROOT=$PROJECT_ROOT"
}
