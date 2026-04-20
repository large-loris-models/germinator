#!/usr/bin/env bash
# Germinator — collect .mlir seed files from cuda-tile tests and the MLIR
# upstream test suite. Filenames are prefixed with the source directory to
# avoid collisions across families.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=== Collecting seed .mlir files ==="

if [[ ! -d "$CUDA_TILE_SRC" ]]; then
    echo "ERROR: cuda-tile source not found at $CUDA_TILE_SRC" >&2
    echo "  Run scripts/build/setup_deps.sh first." >&2
    exit 1
fi
if [[ ! -d "$LLVM_SRC" ]]; then
    echo "ERROR: LLVM source not found at $LLVM_SRC" >&2
    echo "  Run scripts/build/setup_deps.sh first." >&2
    exit 1
fi

mkdir -p "$SEEDS_DIR"

# ── Helper: copy .mlir files under a root, prefix with <leaf>__ ─────────────
#
# Prefix uses the *immediate directory name* of each file's parent to keep
# dialect/transform provenance visible in the seed name.

copy_mlir_under() {
    local root="$1"
    local tag="$2"   # high-level tag, e.g. "cuda-tile"
    local count=0
    if [[ ! -d "$root" ]]; then
        echo "  [seeds] skipping missing: $root"
        return
    fi
    while IFS= read -r -d '' f; do
        local dir leaf base
        dir="$(dirname "$f")"
        leaf="$(basename "$dir")"
        base="$(basename "$f")"
        cp "$f" "$SEEDS_DIR/${tag}__${leaf}__${base}"
        count=$((count + 1))
    done < <(find "$root" -type f -name '*.mlir' -size -100k -print0 2>/dev/null)
    echo "  [seeds] $tag: collected $count files from $root"
}

# cuda-tile's own tests.
copy_mlir_under "$CUDA_TILE_SRC/test" "cuda-tile"

# Upstream MLIR dialect + transform test suites.
copy_mlir_under "$LLVM_SRC/mlir/test/Dialect" "dialect"
copy_mlir_under "$LLVM_SRC/mlir/test/Transforms" "transforms"

TOTAL=$(find "$SEEDS_DIR" -maxdepth 1 -type f -name '*.mlir' | wc -l)
echo ""
echo "[seeds] Collected $TOTAL total .mlir files into $SEEDS_DIR/"
