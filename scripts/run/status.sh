#!/usr/bin/env bash
# =============================================================================
# status.sh — One-shot snapshot of fuzzer + ASAN watcher + corpus state.
# Read-only. Intended for `watch -n 5 ./scripts/run/status.sh` if you want
# a live view.
# =============================================================================
source "$(dirname "$0")/../build/env.sh"

exec uv run python "$PROJECT_ROOT/scripts/run/status.py"
