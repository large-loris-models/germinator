#!/usr/bin/env bash
# =============================================================================
# test_mutations.sh — Build and run the standalone mutation test
# =============================================================================
# Builds test_mutations against the sancov LLVM/cuda-tile builds and runs it
# against two random GRTF trees from seeds/trees/.
# =============================================================================
source "$(dirname "$0")/env.sh"
check_prereqs

# ── Resolve dynamic paths ───────────────────────────────────────────────────
GRAMMARINATOR_CXX="$GRAMMARINATOR_SRC/grammarinator-cxx"
GRTF_LIB="$GRAMMARINATOR_CXX/build/lib/libgrlf-mlir.a"
GRTF_INCLUDE="$GRAMMARINATOR_CXX/libgrlf/include"
RUNTIME_INCLUDE="$GRAMMARINATOR_CXX/libgrammarinator/include"
GENERATED_DIR="$PROJECT_ROOT/generated"
TREES_DIR="$SEEDS_DIR/trees"

# Flatbuffers is conan-installed; the cache path has a hash, so glob it.
LIB_FLATBUFFERS="$(find "$HOME/.conan2/p/b" -path '*/p/lib/libflatbuffers.a' 2>/dev/null | head -1)"

echo "============================================================"
echo " Mutation test"
echo "   Build dir:  $BUILD_OUT"
echo "   LLVM build: $LLVM_BUILD_SANCOV"
echo "   CT build:   $CT_BUILD_SANCOV"
echo "   Trees dir:  $TREES_DIR"
echo "============================================================"

# ── Verify prerequisites ────────────────────────────────────────────────────
for f in "$GRTF_LIB" "$LIB_FLATBUFFERS"; do
    if [[ -z "$f" ]] || [[ ! -f "$f" ]]; then
        echo "ERROR: required library not found: $f" >&2
        exit 1
    fi
done

TREE_COUNT=$(find "$TREES_DIR" -maxdepth 1 -name '*.grtf' 2>/dev/null | wc -l)
if [[ "$TREE_COUNT" -lt 2 ]]; then
    echo "ERROR: need at least 2 .grtf trees in $TREES_DIR (found $TREE_COUNT)" >&2
    echo "  Run setup_grammarinator.sh first." >&2
    exit 1
fi
echo "[test] found $TREE_COUNT .grtf trees"

mkdir -p "$BUILD_OUT"

# ── Extract MLIR link libraries from the sancov build ───────────────────────
echo "[test] extracting MLIR link libraries..."
LINK_LIBS=()
if [[ -f "$LLVM_BUILD_SANCOV/build.ninja" ]]; then
    mapfile -t LINK_LIBS < <(
        cd "$LLVM_BUILD_SANCOV" &&
            ninja -t commands bin/mlir-opt 2>/dev/null |
            grep -E '(clang\+\+|ld\.lld).*mlir-opt' |
            grep -oP 'lib/\S+\.a' |
            sort -u |
            while read -r lib; do echo "$LLVM_BUILD_SANCOV/$lib"; done
    )
    echo "[test] found ${#LINK_LIBS[@]} LLVM/MLIR libraries"
fi

if ! printf '%s\n' "${LINK_LIBS[@]}" | grep -q 'libLLVMSupport'; then
    echo "WARNING: libLLVMSupport.a not found — expect linker errors" >&2
fi

# ── Add cuda-tile libraries ─────────────────────────────────────────────────
CT_LIBS=()
mapfile -t CT_LIBS < <(find "$CT_BUILD_SANCOV/lib" -name '*.a' 2>/dev/null | sort)
echo "[test] found ${#CT_LIBS[@]} cuda-tile libraries"

# ── Compile stubs ───────────────────────────────────────────────────────────
echo "[test] compiling stubs..."
STUBS_OBJ="$BUILD_OUT/test_stubs.o"
"$CXX" -g -O2 -std=c++20 \
    -c "$PROJECT_ROOT/src/test_stubs.cc" \
    -o "$STUBS_OBJ"

# ── Common include flags ────────────────────────────────────────────────────
INCLUDE_FLAGS=(
    -I"$GRTF_INCLUDE"
    -I"$RUNTIME_INCLUDE"
    -I"$PROJECT_ROOT"
    -I"$PROJECT_ROOT/src/mutator"
    -I"$GENERATED_DIR"
    -I"$LLVM_BUILD_SANCOV/include"
    -I"$LLVM_INSTALL_SANCOV/include"
    -I"$LLVM_SRC/llvm/include"
    -I"$LLVM_SRC/mlir/include"
    -I"$LLVM_BUILD_SANCOV/tools/mlir/include"
    -I"$CUDA_TILE_SRC/include"
    -I"$CT_BUILD_SANCOV/include"
)

# ── Compile mutator sources ─────────────────────────────────────────────────
echo "[test] compiling mutator sources..."
MUTATOR_SRCS=(
    "$PROJECT_ROOT/src/mutator/context_filter.cc"
    "$PROJECT_ROOT/src/mutator/edit.cc"
    "$PROJECT_ROOT/src/mutator/insert.cc"
)
MUTATOR_OBJS=()

for src in "${MUTATOR_SRCS[@]}"; do
    obj="$BUILD_OUT/$(basename "${src%.cc}").o"
    "$CXX" -g -O2 -std=c++20 \
        "${INCLUDE_FLAGS[@]}" \
        -c "$src" -o "$obj"
    MUTATOR_OBJS+=("$obj")
done

# ── Compile test_mutations ──────────────────────────────────────────────────
echo "[test] compiling test_mutations.cc..."
TEST_OBJ="$BUILD_OUT/test_mutations.o"
"$CXX" -g -O2 -std=c++20 \
    "${INCLUDE_FLAGS[@]}" \
    -c "$PROJECT_ROOT/src/test_mutations.cc" \
    -o "$TEST_OBJ"

# ── Link ────────────────────────────────────────────────────────────────────
echo "[test] linking test_mutations..."
TEST_BIN="$BUILD_OUT/test_mutations"

"$CXX" -std=c++20 \
    -fno-sanitize=all \
    -fuse-ld=lld \
    "$TEST_OBJ" \
    "${MUTATOR_OBJS[@]}" \
    "$STUBS_OBJ" \
    "$GRTF_LIB" \
    "$LIB_FLATBUFFERS" \
    -Wl,--start-group \
    "${CT_LIBS[@]}" \
    "${LINK_LIBS[@]}" \
    -Wl,--end-group \
    -ldl -lrt -lpthread -lm -lz -ltinfo \
    -o "$TEST_BIN"

echo "[test] built $TEST_BIN"

# ── Run ─────────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────"
echo " Running mutation test"
echo "────────────────────────────────────────────────────────────"

# Pick the first two .grtf trees deterministically.
mapfile -t TREE_FILES < <(find "$TREES_DIR" -maxdepth 1 -name '*.grtf' | sort | head -2)
SEED1="${TREE_FILES[0]}"
SEED2="${TREE_FILES[1]}"

OUTPUT_DIR="$BUILD_OUT/mutation_test_output"
mkdir -p "$OUTPUT_DIR"

echo "[test] seed 1: $(basename "$SEED1")"
echo "[test] seed 2: $(basename "$SEED2")"
echo "[test] output: $OUTPUT_DIR"
echo ""

# Hand mlir-opt path to the binary so it can parse-check outputs.
MLIR_OPT_PATH="$LLVM_BUILD_PLAIN/bin/mlir-opt" \
    "$TEST_BIN" "$SEED1" "$SEED2" 10 "$OUTPUT_DIR"

echo ""
echo "[test] mutation test complete"
echo "[test] output files:"
find "$OUTPUT_DIR" -maxdepth 1 -name '*.mlir' | head -10
echo ""
echo "============================================================"
echo " Done. Review outputs in: $OUTPUT_DIR"
echo "============================================================"