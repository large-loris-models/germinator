# Germinator

An extensible grammar-based fuzzer for IR-family languages.

## Overview

Germinator combines grammar-based fuzzing with parameterized mutation synthesis to generate high-quality test cases for compiler intermediate representations (IRs). It builds on ideas from [SynthFuzz](https://arxiv.org/abs/2404.16947) and [Germinator](https://arxiv.org/abs/2512.05887), and extends them to support multiple IR families.

## Features

- TBD

## Supported Targets

- **MLIR**
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



## Bugs Found

### Project: LLVM MLIR
| Type | Description |
| :--- | :--- |
| **Memory Safety** | Heap buffer overflow in GPU dialect via negative workgroup attributions |
| **Memory Safety** | Memory leak in SPIR-V to LLVM conversion |
| **Stack Overflow** | Recursion in dialect conversion with circular SSA dependencies |
| **Segfault** | Null pointer dereference in Shape dialect library verification |
| **Crash** | `emitc.for` verification crash on missing block arguments |
| **Crash** | `irdl.type` verification crash on empty symbol name |
| **Crash** | `tosa.concat` verification crash accessing OOB dimension index |
| **Crash** | `memref.alloc` crash on zero-result operation |
| **Crash** | Test dialect verification crash on dynamic shapes |
| **Crash** | `tosa.table` verification crash on unranked tensor result type |
| **Crash** | SCF to SPIR-V conversion crash on zero-sized memref |
| **Crash** | Transform dialect crash on array bounds in `transform.include` |
| **Crash** | Parser crash on mixed string/integer literals in dense tensor |
| **Crash** | CallGraph crash on `func.call` with nested regions |
| **Crash** | `tosa.if` crash when region lacks terminator |
| **Crash** | Transform dialect crash when applied to non-function operations |
| **Assertion** | Failure in `scf.for` verification with malformed region args |
| **Assertion** | Failure in IRDL verification with self-referencing types |

### Project: Triton
| Type | Description |
| :--- | :--- |
| **Memory Safety** | Heap corruption in `tt.expand_dims` during error recovery |
| **Memory Safety** | Heap corruption in `ReduceOp` shape inference |
| **Crash** | Crash in AMDGPU to LLVM conversion when tensor lacks encoding |

### Project: CIRCT
| Type | Description |
| :--- | :--- |
| **Segfault** | Segmentation fault in Handshake buffer insertion |
| **Segfault** | Segmentation fault in loop schedule pipeline verifier |
| **Memory Leak** | Memory leak in HW to SystemC conversion |
| **Logic Error** | Folding rollback error during HW to LLVM conversion |
| **Assertion** | Failure in InstanceGraph analysis |
| **Assertion** | Failure in DenseArrayAttr parser |
| **Assertion** | Failure in Arc function splitting |
| **Assertion** | Failure in DeadCodeAnalysis during FSM range narrowing |
| **Assertion** | Failure when printing external modules with mismatched ports |

### Project: HEIR
| Type | Description |
| :--- | :--- |
| **Memory Safety** | Stack-use-after-return in `CaptureAmbientScope` |
| **Memory Safety** | Stack-use-after-scope in LWE Verification |
| **Memory Safety** | Heap-use-after-free during dialect conversion |
| **Logic Error** | Replacement value count mismatch in secret generic conversion |
| **Logic Error** | Invalid type cast in Arith to Mod Arith conversion |
| **Legalization** | Failure for `mgmt.modreduce` in secret generic conversion |
| **Legalization** | Failure in CGGI Quart conversion |
| **Assertion** | Use-list failure during secret to mod arith conversion |
| **Assertion** | Bitwidth mismatch in CGGI materialization |
| **Assertion** | Failure in `tensor.empty` via ElementwiseToAffine |
| **Assertion** | Failure with nested multi-result ops in `secret.generic` |
| **Assertion** | Failure in `secret.generic` verifier on operand count mismatch |

### Project: Torch MLIR

| Type | Description |
| --- | --- |
| **Assertion** | Failure in `SymbolTable` due to duplicate symbol names in global slots |
| **Assertion** | Incompatible type cast in global slot initializer due to integer attribute |
| **Assertion** | Failure in `Block::getTerminator` for missing initializer terminators |
| **Assertion** | Unexpected terminator type cast failure in global slot initializer |
| **Assertion** | Invariant failure in `ValueTensorType` during type inference from signless i64 |

### Project: IREE

| Type | Description |
| --- | --- |
| **Assertion** | Failure in DemoteF64ToF32 pass when encountering `llvm.func` operations |
| **Segfault** | Segmentation fault in the ConvertToStream pass during utility list type conversion |
| **Assertion** | Failure in function printer due to invalid tied operand index |
| **Memory Leak** | Memory leak in HAL conversion during Stream command execution legalization |
| **Memory Safety** | Memory corruption in LiftCFGToSCF pass when processing empty function regions |
| **Assertion** | Failure in Stream dialect conversion when `flow.executable.export` has results |
| **Assertion** | Failure in DialectConversion during StableHLO to IREE input conversion with zero-extent tensors |
| **Memory Safety** | Memory corruption in the ConvertToStream pass during operand type inspection |
| **Assertion** | Failure in `Util::FuncOp` printer due to out-of-bounds `tied_operands` index |
| **Assertion** | Failure in `FloatAttr::get` during constant folding of `vm.cast.ui64.f64` |
| **Assertion** | Failure in `MapScatterOp::verify` when region block is missing a terminator |
| **Segfault** | Segfault in DialectConversion rollback during `ConvertBf16ToUInt16BuffersPass` |
| **Assertion** | Failure in transform interpreter when encountering non-transform operation |
| **Assertion** | Failure in `Stream::AsyncFuncOp::build` due to result attribute mismatch |
| **Assertion** | Failure in `Stream::AsyncFuncOp::build` due to argument attribute mismatch |
| **Crash** | `hal.device.queue.dealloca` verifier crashes when fence operand is a block argument |


## License

MIT
