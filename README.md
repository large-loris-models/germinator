# Germinator

Coverage-guided fuzzer for the **cuda-tile** MLIR dialect, driven by
[Centipede](https://github.com/google/fuzztest/tree/main/centipede) and
triaged against an ASAN-instrumented build of `cuda-tile-opt`.

## What this is

`cuda-tile-opt` is the dialect driver for the cuda-tile MLIR dialect.
Germinator feeds it mutated `.mlir` inputs under coverage instrumentation,
and replays interesting inputs against a parallel ASAN/UBSAN build to
surface memory safety and undefined behavior bugs.

## Architecture

```
            seeds/ ─┐
                    ▼
                  corpus/  ◀─────────────── Centipede
                    │          (sancov cuda-tile-opt, N jobs)
                    │
                    ├──► ASAN oracle       (asan cuda-tile-opt, triage)
                    └──► grammarinator     (planned, grammar-aware gen)
```

Three LLVM/MLIR builds are produced, each installed under `deps/`:

| Build  | Purpose                         | Install prefix             |
|--------|---------------------------------|----------------------------|
| sancov | fuzz target coverage            | `deps/llvm-install-sancov` |
| asan   | ASAN/UBSAN oracle               | `deps/llvm-install-asan`   |
| plain  | `mlir-opt`, `llvm-symbolizer`   | `deps/llvm-install-plain`  |

cuda-tile is built twice — once against the sancov install (for fuzzing)
and once against the asan install (for triage).

## Quick start

```sh
# 1. One-time OS-level setup (apt, bazelisk).
./scripts/build/bootstrap.sh

# 2. Fetch submodules (fuzztest, grammarinator forks).
git submodule update --init --recursive

# 3. Build LLVM/MLIR (×3) and cuda-tile (×2). Takes a while.
./scripts/build/setup_deps.sh

# 4. Build Centipede engine + runner from the fuzztest submodule.
./scripts/build/build_centipede.sh

# 5. Collect .mlir seeds from cuda-tile + upstream MLIR tests.
./scripts/build/collect_seeds.sh

# 6. Link the fuzz harness binaries (currently a placeholder).
./scripts/build/link_fuzz_target.sh

# 7. Start the pipeline (fuzzer + ASAN oracle).
nohup ./scripts/run/start.sh > build/run.log 2>&1 &

# 8. Check progress.
./scripts/analysis/status.sh

# 9. Stop.
./scripts/run/stop.sh
```

## Layout

```
scripts/
├── build/
│   ├── env.sh                   # All paths, flags, check_prereqs
│   ├── bootstrap.sh             # OS packages (apt)
│   ├── setup_deps.sh            # LLVM/MLIR ×3 + cuda-tile ×2
│   ├── build_centipede.sh       # Centipede via Bazel
│   ├── link_fuzz_target.sh      # (placeholder) fuzz harness linking
│   ├── collect_seeds.sh         # .mlir seeds → seeds/
│   └── setup_grammarinator.sh   # (placeholder) grammar-aware generator
├── run/
│   ├── start.sh                 # Fuzzer + ASAN oracle, core-pinned
│   ├── stop.sh                  # Graceful shutdown
│   └── run_tests.sh             # (placeholder) mutator tests
├── oracles/
│   ├── common.sh                # Sharding, inotify, result bookkeeping
│   └── asan_opt.sh              # ASAN/UBSAN oracle for cuda-tile-opt
└── analysis/
    ├── status.sh                # Unified dashboard
    └── unique_crashes.sh        # Crash dedup from run.log

src/
├── harness/                     # (empty) fuzz harnesses
└── mutators/                    # (empty) structured mutators

tests/                           # (empty) unit tests
deps/                            # LLVM source/builds/installs, cuda-tile
build/                           # Build outputs, workdirs, oracle results
corpus/                          # Fuzzer corpus (lives across runs)
seeds/                           # Raw .mlir seeds collected from tests
third_party/
├── fuzztest/                    # submodule (sairam2661 fork)
└── grammarinator/               # submodule (sairam2661 fork)
```

## Bugs found

_TBD._
