# Germinator

An extensible grammar-based fuzzer for IR-family languages.

## Overview

Germinator combines grammar-based fuzzing with parameterized mutation synthesis to generate high-quality test cases for compiler intermediate representations (IRs). It builds on ideas from [SynthFuzz](https://arxiv.org/abs/2404.16947) and [Germinator](https://arxiv.org/abs/2512.05887), and extends them to support multiple IR families.

## Features

- TBD

## Supported Targets

- **MLIR** (all dialects)
- **LLVM IR** (planned)
- **WebAssembly** (planned)

## Installation

```bash
# From PyPI (when published)
pip install germinator

# From source
git clone https://github.com/large-loris-models/germinator
cd germinator
pip install -e .

# With LLM support
pip install -e ".[llm]"
```

## Quick Start

```bash
# Process a grammar
germinator-process src/germinator/grammar/grammars/mlir.g4 -o build/ --rule start_rule 

# Convert test cases to ASTs
grammarinator-parse src/germinator/grammar/grammars/mlir.g4 -i {input/*.mlir} -o {trees} -r start_rule

# Generate test cases
germinator fuzz --family ssa.mlir --seeds trees --output tests -n 1000 --grammar build --no-keep-trees -j 10
```

## Architecture

```
germinator/
├── core/           # Domain-agnostic primitives
├── families/       # Mutation strategies per domain (ssa/, etc.)
├── seeds/          # Seed corpus management + LLM generation
├── grammar/        # Grammar processing
├── engine/         # Orchestration
├── drivers/        # Test execution (pluggable)
└── cli/            # Command-line interface
```

## Adding a New Target

1. Add grammar file: `src/germinator/grammar/grammars/your_target.g4`
2. Add family config: `src/germinator/families/ssa/your_target/config.toml`
3. (Optional) Add driver: `src/germinator/drivers/your_target.py`

See [docs/adding-targets.md](docs/adding-targets.md) for details.

## License

MIT