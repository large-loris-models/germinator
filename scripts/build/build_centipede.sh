#!/usr/bin/env bash
# Germinator — build Centipede from the vendored fuzztest submodule.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
check_prereqs

echo "=== Building Centipede ==="

if [[ ! -d "$FUZZTEST_SRC" ]]; then
    echo "ERROR: fuzztest submodule not found at $FUZZTEST_SRC" >&2
    echo "  Run: git submodule update --init --recursive" >&2
    exit 1
fi

cd "$FUZZTEST_SRC"

echo "[centipede] Building with Bazel..."
bazel build -c opt centipede:all \
    --features=-layering_check \
    --host_features=-layering_check

for f in "$CENTIPEDE_BIN" "$CENTIPEDE_RUNNER"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: expected Centipede artifact not found: $f" >&2
        exit 1
    fi
done

echo "[centipede] Engine: $CENTIPEDE_BIN"
echo "[centipede] Runner: $CENTIPEDE_RUNNER"
