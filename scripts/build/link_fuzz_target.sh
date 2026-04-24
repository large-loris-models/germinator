#!/usr/bin/env bash
# Germinator — link the Centipede fuzz target(s) for cuda-tile.
#
# Compiles the harness + mutator and links them against:
#   - cuda-tile dialect libraries
#   - LLVM/MLIR libraries (sancov- or asan-instrumented)
#   - Grammarinator GRLF codec + flatbuffers
#   - Centipede runner (+ sancov runtime for the sancov variant)
#
# Produces:
#   build/cuda_tile_opt_fuzz_target        (sancov, primary fuzz binary)
#   build/cuda_tile_opt_fuzz_target_asan   (asan, extra_binaries for crash check)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# ── Paths not already exported by env.sh ────────────────────────────────────

GRAMMARINATOR_CXX="$GRAMMARINATOR_SRC/grammarinator-cxx"
GRLF_LIB="$GRAMMARINATOR_CXX/build/lib/libgrlf-mlir.a"
GRLF_INCLUDE="$GRAMMARINATOR_CXX/libgrlf/include"
RUNTIME_INCLUDE="$GRAMMARINATOR_CXX/libgrammarinator/include"
GENERATED_DIR="$PROJECT_ROOT/generated"
CENTIPEDE_SANCOV="$FUZZTEST_SRC/bazel-bin/centipede/libsancov_runtime.pic.a"

LIB_FLATBUFFERS="${LIB_FLATBUFFERS:-}"
if [[ -z "$LIB_FLATBUFFERS" ]]; then
    LIB_FLATBUFFERS="$(find "$HOME/.conan2" -path '*/p/lib/libflatbuffers.a' 2>/dev/null | head -1)"
fi

HARNESS_SRC="$PROJECT_ROOT/src/harness/mlir_fuzz_target.cc"
MUTATOR_SRCS=(
    "$PROJECT_ROOT/src/mutator/context_filter.cc"
    "$PROJECT_ROOT/src/mutator/registry.cc"
    "$PROJECT_ROOT/src/mutator/tree_mutations/edit_mutation.cc"
    "$PROJECT_ROOT/src/mutator/tree_mutations/insert_mutation.cc"
)

mkdir -p "$BUILD_OUT"

# ── Verify shared prerequisites ─────────────────────────────────────────────

for f in "$CENTIPEDE_RUNNER" "$GRLF_LIB" "$LIB_FLATBUFFERS" "$HARNESS_SRC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found: $f" >&2
        exit 1
    fi
done

# ── link_variant <name> <llvm_build> <llvm_install> <ct_build> ──────────────
#
# Compiles the harness + mutator with variant-specific flags, then links
# against the given LLVM/MLIR + cuda-tile builds.  Variant-specific flags
# (and whether to pull in libsancov_runtime) are selected by $name.

link_variant() {
    local variant="$1"
    local llvm_build="$2"
    local llvm_install="$3"
    local ct_build="$4"

    local tag="[link-$variant]"
    local out_suffix=""
    [[ "$variant" == "asan" ]] && out_suffix="_asan"

    echo "============================================================"
    echo " Linking $variant fuzz target"
    echo "   LLVM build:  $llvm_build"
    echo "   cuda-tile:   $ct_build"
    echo "============================================================"

    if [[ ! -f "$llvm_build/build.ninja" ]]; then
        echo "ERROR: LLVM build.ninja not found at $llvm_build" >&2
        return 1
    fi
    if [[ ! -d "$ct_build/lib" ]]; then
        echo "ERROR: cuda-tile libs not found at $ct_build/lib" >&2
        return 1
    fi

    # Extract the full archive list that mlir-opt links against.  This is the
    # simplest way to get a working, correctly-ordered set of MLIR/LLVM libs
    # without hand-maintaining a list.
    echo "$tag extracting LLVM/MLIR link libraries from $llvm_build..."
    local -a link_libs
    mapfile -t link_libs < <(
        cd "$llvm_build" &&
            ninja -t commands bin/mlir-opt 2>/dev/null |
            grep -E '(clang\+\+|ld\.lld).*mlir-opt' |
            grep -oP 'lib/\S+\.a' |
            sort -u |
            while read -r lib; do echo "$llvm_build/$lib"; done
    )
    echo "$tag found ${#link_libs[@]} LLVM/MLIR libraries"

    if ! printf '%s\n' "${link_libs[@]}" | grep -q 'libLLVMSupport'; then
        echo "WARNING: libLLVMSupport.a not in link set — expect linker errors" >&2
    fi

    local -a ct_libs
    mapfile -t ct_libs < <(find "$ct_build/lib" -name '*.a' 2>/dev/null | sort)
    echo "$tag found ${#ct_libs[@]} cuda-tile libraries"

    local -a include_flags=(
        -I"$GRLF_INCLUDE"
        -I"$RUNTIME_INCLUDE"
        -I"$PROJECT_ROOT"
        -I"$PROJECT_ROOT/src/mutator"
        -I"$GENERATED_DIR"
        -I"$llvm_build/include"
        -I"$llvm_install/include"
        -I"$LLVM_SRC/llvm/include"
        -I"$LLVM_SRC/mlir/include"
        -I"$llvm_build/tools/mlir/include"
        -I"$CUDA_TILE_SRC/include"
        -I"$ct_build/include"
    )

    # Variant-specific compile/link flags.
    local -a compile_flags link_flags
    if [[ "$variant" == "sancov" ]]; then
        compile_flags=("${FUZZ_CFLAGS[@]}")
        link_flags=(-fno-sanitize=all)
    else
        compile_flags=(
            -g -O1
            -fno-omit-frame-pointer
            -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
            -fsanitize=address,undefined
            -fno-sanitize-recover=all
        )
        link_flags=(-fsanitize=address,undefined -fno-sanitize-recover=all)
    fi

    # Mutator objects: same flags as the harness so sanitizer ABI matches.
    echo "$tag compiling mutator objects..."
    local -a mutator_objs
    for src in "${MUTATOR_SRCS[@]}"; do
        local obj="$BUILD_OUT/$(basename "${src%.cc}")${out_suffix}.o"
        "$CXX" "${compile_flags[@]}" -std=c++20 \
            "${include_flags[@]}" \
            -c "$src" -o "$obj"
        mutator_objs+=("$obj")
        echo "$tag   + $(basename "$obj")"
    done

    echo "$tag compiling fuzz harness..."
    local harness_obj="$BUILD_OUT/mlir_fuzz_target${out_suffix}.o"
    "$CXX" "${compile_flags[@]}" -std=c++20 \
        "${include_flags[@]}" \
        -c "$HARNESS_SRC" -o "$harness_obj"

    local fuzz_target="$BUILD_OUT/cuda_tile_opt_fuzz_target${out_suffix}"
    echo "$tag linking $fuzz_target..."

    # Group the archives: MLIR/LLVM + cuda-tile have circular refs; the
    # centipede runner also needs to resolve against them.
    local -a runtime_libs=("$CENTIPEDE_RUNNER")
    [[ "$variant" == "sancov" ]] && runtime_libs+=("$CENTIPEDE_SANCOV")

    "$CXX" -std=c++20 \
        "${link_flags[@]}" \
        -fuse-ld=lld \
        "$harness_obj" \
        "${mutator_objs[@]}" \
        "$GRLF_LIB" \
        "$LIB_FLATBUFFERS" \
        -Wl,--start-group \
        "${ct_libs[@]}" \
        "${link_libs[@]}" \
        "${runtime_libs[@]}" \
        -Wl,--end-group \
        -ldl -lrt -lpthread -lm -lz -ltinfo \
        -o "$fuzz_target"

    echo "$tag success: $fuzz_target ($(du -h "$fuzz_target" | cut -f1))"
}

# ── Build both variants ─────────────────────────────────────────────────────

link_variant sancov "$LLVM_BUILD_SANCOV" "$LLVM_INSTALL_SANCOV" "$CT_BUILD_SANCOV"
link_variant asan   "$LLVM_BUILD_ASAN"   "$LLVM_INSTALL_ASAN"   "$CT_BUILD_ASAN"

echo ""
echo "=== Fuzz targets linked ==="
echo "  primary: $BUILD_OUT/cuda_tile_opt_fuzz_target"
echo "  asan:    $BUILD_OUT/cuda_tile_opt_fuzz_target_asan"
