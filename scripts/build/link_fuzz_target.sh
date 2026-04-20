#!/usr/bin/env bash
# Germinator — build fuzz harnesses (placeholder).
#
# When filled in, this script should:
#   - Compile src/harness/*.cc against the sancov LLVM/MLIR install and the
#     sancov cuda-tile build, linking with $CENTIPEDE_RUNNER, to produce
#     build/cuda_tile_opt_fuzz_target.
#   - Compile an ASAN-instrumented harness against the asan install, to
#     produce build/cuda_tile_opt_fuzz_target_asan.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=== Linking fuzz targets ==="
echo "TODO: Build cuda-tile-opt fuzz harness"
echo "  - Link against sancov cuda-tile + MLIR libraries"
echo "  - Link with Centipede runner ($CENTIPEDE_RUNNER)"
echo "TODO: Build ASAN cuda-tile-opt harness"
