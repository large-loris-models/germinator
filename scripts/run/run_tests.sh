#!/usr/bin/env bash
# Germinator — placeholder for mutator / harness tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../build/env.sh"

echo "=== Germinator tests ==="
echo "TODO: Compile tests/ against the sancov LLVM/MLIR install and run."
