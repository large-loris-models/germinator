#!/usr/bin/env bash
# Germinator — build LLVM/MLIR (3 configs: sancov, asan, plain) and
# cuda-tile (2 configs: sancov, asan). Each LLVM/MLIR build is installed to a
# prefix that cuda-tile's CMake configure consumes.
#
# Idempotent: each stage guards on the presence of its key binary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
check_prereqs

mkdir -p "$DEPS_DIR"

JOBS="${NUM_JOBS:-$(nproc)}"

echo "=== Germinator: building dependencies ==="
echo "  Parallel jobs: $JOBS"
echo "  Deps dir:      $DEPS_DIR"
echo ""

# ── [1] Clone LLVM source ───────────────────────────────────────────────────

echo "[1/6] LLVM source..."
if [[ ! -d "$LLVM_SRC" ]]; then
    echo "  Cloning llvm-project..."
    git clone https://github.com/llvm/llvm-project.git "$LLVM_SRC"
else
    echo "  LLVM source already present, skipping clone."
fi

# ── Patch -z,defs in LLVM/MLIR CMake fragments ──────────────────────────────
#
# sancov-instrumented builds leave __sanitizer_cov_* symbols unresolved
# until the fuzzer runtime is linked. The LLVM/MLIR build uses `-Wl,-z,defs`
# which rejects unresolved symbols at link time. Strip it everywhere before
# configuring any build.

patch_zdefs() {
    echo "  Patching -z,defs out of LLVM/MLIR CMake fragments..."
    local dirs=("$LLVM_SRC/llvm" "$LLVM_SRC/mlir" "$LLVM_SRC/cmake")
    local f
    while IFS= read -r -d '' f; do
        if grep -q '\-z,defs' "$f" && ! grep -q '#PATCHED' "$f"; then
            sed -i '/PATCHED/!s/.*-z,defs.*/#PATCHED: &/' "$f"
        fi
    done < <(find "${dirs[@]}" \( -name '*.cmake' -o -name 'CMakeLists.txt' \) -print0 2>/dev/null)
}

strip_zdefs_from_ninja() {
    # CMake may still embed the flag in the generated build.ninja via linker
    # options pulled from system profiles. Scrub it post-configure.
    local build_dir="$1"
    if [[ -f "$build_dir/build.ninja" ]]; then
        sed -i 's/-Wl,-z,defs//g; s/-z,defs//g; s/-z defs//g' "$build_dir/build.ninja"
    fi
}

patch_zdefs

# ── Shared cmake flags for all LLVM/MLIR builds ─────────────────────────────

COMMON_LLVM_FLAGS=(
    -G Ninja
    -DCMAKE_C_COMPILER="$CC"
    -DCMAKE_CXX_COMPILER="$CXX"
    -DLLVM_ENABLE_PROJECTS="mlir;clang"
    -DLLVM_TARGETS_TO_BUILD="Native"
    -DLLVM_USE_LINKER=lld
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DLLVM_ENABLE_RTTI=ON
    -DLLVM_BUILD_EXAMPLES=OFF
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
    -DBUILD_SHARED_LIBS=OFF
)

# ── [2] LLVM/MLIR (sancov) ──────────────────────────────────────────────────

echo "[2/6] Building LLVM/MLIR (sancov)..."

SANCOV_CFLAGS="-g -O2 -fno-omit-frame-pointer -gline-tables-only -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION -fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp"

if [[ ! -x "$LLVM_INSTALL_SANCOV/bin/mlir-opt" ]]; then
    echo "  Configuring LLVM/MLIR (sancov)..."
    cmake -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD_SANCOV" \
        "${COMMON_LLVM_FLAGS[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="$SANCOV_CFLAGS" \
        -DCMAKE_CXX_FLAGS="$SANCOV_CFLAGS" \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_SANCOV"

    strip_zdefs_from_ninja "$LLVM_BUILD_SANCOV"

    echo "  Building LLVM/MLIR (sancov)..."
    cmake --build "$LLVM_BUILD_SANCOV" -j"$JOBS"

    echo "  Installing to $LLVM_INSTALL_SANCOV ..."
    cmake --build "$LLVM_BUILD_SANCOV" --target install -j"$JOBS"
else
    echo "  LLVM/MLIR (sancov) already installed, skipping."
fi

# ── [3] LLVM/MLIR (asan + ubsan) ────────────────────────────────────────────

echo "[3/6] Building LLVM/MLIR (asan)..."

ASAN_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"

if [[ ! -x "$LLVM_INSTALL_ASAN/bin/mlir-opt" ]]; then
    echo "  Configuring LLVM/MLIR (asan)..."
    cmake -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD_ASAN" \
        "${COMMON_LLVM_FLAGS[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="$ASAN_CFLAGS" \
        -DCMAKE_CXX_FLAGS="$ASAN_CFLAGS" \
        -DLLVM_USE_SANITIZER="Address;Undefined" \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_ASAN"

    strip_zdefs_from_ninja "$LLVM_BUILD_ASAN"

    echo "  Building LLVM/MLIR (asan)..."
    cmake --build "$LLVM_BUILD_ASAN" -j"$JOBS"

    echo "  Installing to $LLVM_INSTALL_ASAN ..."
    cmake --build "$LLVM_BUILD_ASAN" --target install -j"$JOBS"
else
    echo "  LLVM/MLIR (asan) already installed, skipping."
fi

# ── [4] LLVM/MLIR (plain) ───────────────────────────────────────────────────

echo "[4/6] Building LLVM/MLIR (plain)..."

if [[ ! -x "$LLVM_INSTALL_PLAIN/bin/mlir-opt" ]]; then
    echo "  Configuring LLVM/MLIR (plain)..."
    cmake -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD_PLAIN" \
        "${COMMON_LLVM_FLAGS[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PLAIN"

    strip_zdefs_from_ninja "$LLVM_BUILD_PLAIN"

    echo "  Building LLVM/MLIR (plain) — mlir-opt llvm-symbolizer opt..."
    cmake --build "$LLVM_BUILD_PLAIN" --target mlir-opt llvm-symbolizer opt -j"$JOBS"

    echo "  Installing to $LLVM_INSTALL_PLAIN ..."
    cmake --build "$LLVM_BUILD_PLAIN" --target install -j"$JOBS"
else
    echo "  LLVM/MLIR (plain) already installed, skipping."
fi

# ── [5] Clone cuda-tile ─────────────────────────────────────────────────────

echo "[5/6] cuda-tile source..."
if [[ ! -d "$CUDA_TILE_SRC" ]]; then
    echo "  Cloning cuda-tile from $CUDA_TILE_REPO ..."
    git clone "$CUDA_TILE_REPO" "$CUDA_TILE_SRC"
else
    echo "  cuda-tile source already present, skipping clone."
fi

# ── [6] cuda-tile (sancov) + (asan) ─────────────────────────────────────────

echo "[6/6] Building cuda-tile..."

build_cuda_tile() {
    local build_dir="$1"
    local install_prefix="$2"
    local cflags="$3"
    local bin="$build_dir/bin/cuda-tile-opt"

    if [[ -x "$bin" ]]; then
        echo "  cuda-tile at $bin already built, skipping."
        return
    fi

    echo "  Configuring cuda-tile → $build_dir (LLVM=$install_prefix)..."
    cmake -G Ninja -S "$CUDA_TILE_SRC" -B "$build_dir" \
        -DCUDA_TILE_USE_LLVM_INSTALL_DIR="$install_prefix" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_C_FLAGS="$cflags" \
        -DCMAKE_CXX_FLAGS="$cflags" \
        -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=OFF

    strip_zdefs_from_ninja "$build_dir"

    echo "  Building cuda-tile in $build_dir ..."
    cmake --build "$build_dir" -j"$JOBS"
}

echo "  cuda-tile (sancov)..."
build_cuda_tile "$CT_BUILD_SANCOV" "$LLVM_INSTALL_SANCOV" "$SANCOV_CFLAGS"

echo "  cuda-tile (asan)..."
build_cuda_tile "$CT_BUILD_ASAN" "$LLVM_INSTALL_ASAN" "$ASAN_CFLAGS"

echo ""
echo "=== All dependencies built ==="
echo "  LLVM/MLIR (sancov): $LLVM_INSTALL_SANCOV"
echo "  LLVM/MLIR (asan):   $LLVM_INSTALL_ASAN"
echo "  LLVM/MLIR (plain):  $LLVM_INSTALL_PLAIN"
echo "  cuda-tile (sancov): $CT_OPT_SANCOV"
echo "  cuda-tile (asan):   $CT_OPT_ASAN"
