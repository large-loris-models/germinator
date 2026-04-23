// =============================================================================
// mlir_fuzz_target.cc — Centipede fuzz target for cuda-tile passes
// =============================================================================
//
// Per fuzz iteration:
//   1. Decode the .grlf tree input into a Grammarinator tree.
//   2. Render the tree to MLIR text.
//   3. Parse the text into an MLIR module.
//   4. Pseudo-randomly pick one cuda-tile pass and run it.
//   5. Crashes during parsing or pass execution = bug.
//
// Passes (per the .td definitions):
//   - synthesize-debug-info-scopes — pinned to cuda_tile.module
//   - fuse-fma                     — runs on FunctionOpInterface (cuda_tile.entry)
//   - loop-split                   — runs on FunctionOpInterface (cuda_tile.entry)
//
// Verifier is disabled: we want to catch process-level crashes only, not
// post-pass IR validity issues.
//
// Tree mutations are owned by MutationRegistry.  The harness has no knowledge
// of specific mutation names — adding a new mutation requires no edits here.
// =============================================================================

#include "grlf.h"
#include "grlf_codec.h"

#include "src/mutator/registry.h"

#include <grammarinator/runtime/Population.hpp>
#include <grammarinator/runtime/Rule.hpp>

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/Passes.h"

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>

using grammarinator::runtime::Rule;

// =============================================================================
// MLIR context — initialized once before main
// =============================================================================

static mlir::MLIRContext *ctx = nullptr;

__attribute__((constructor)) static void initMLIR() {
  ctx = new mlir::MLIRContext();

  // Minimal dialect set: cuda-tile + the upstream dialects its passes and ops
  // realistically interact with. registerAllDialects is avoided to keep the
  // binary small (it adds ~700MB of pass/dialect code we never invoke).
  mlir::DialectRegistry registry;
  registry.insert<mlir::cuda_tile::CudaTileDialect>();
  registry.insert<mlir::func::FuncDialect>();
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  ctx->allowUnregisteredDialects();
}

// =============================================================================
// Static configuration
// =============================================================================

namespace {

// Cheap content-derived hash. Used to pick which pass an input runs through;
// stable for a given input so coverage feedback is meaningful (same input
// always exercises the same pass).
uint32_t fnv1a(const uint8_t *data, size_t size) {
  uint32_t h = 0x811c9dc5u;
  for (size_t i = 0; i < size; ++i) {
    h ^= data[i];
    h *= 0x01000193u;
  }
  return h;
}

}  // namespace

// =============================================================================
// Initialization
// =============================================================================

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  GrammarinatorInitialize(argc, argv);
  return 0;
}

// =============================================================================
// Pass execution
// =============================================================================
//
// Choice of pass per input:
//   0 — fuse-fma                (nested in cuda_tile.entry, via FunctionOpInterface)
//   1 — loop-split              (same nesting)
//   2 — synthesize-debug-info-scopes (nested in cuda_tile.module)
//   3 — no pass (parse-only baseline; keeps coverage on the parser path)
//
// Passes are added one at a time so the failure of one doesn't drag others
// down with it. PassManager-level verification is off (enableVerifier(false)).

static void run_one_pass(mlir::ModuleOp module, uint32_t choice) {
  mlir::PassManager pm(ctx);
  pm.enableVerifier(false);

  switch (choice % 4) {
    case 0:
      pm.nest<mlir::cuda_tile::ModuleOp>()
          .nest<mlir::cuda_tile::EntryOp>()
          .addPass(mlir::cuda_tile::createFuseFMAPass());
      break;
    case 1:
      pm.nest<mlir::cuda_tile::ModuleOp>()
          .nest<mlir::cuda_tile::EntryOp>()
          .addPass(mlir::cuda_tile::createLoopSplitPass());
      break;
    case 2:
      pm.nest<mlir::cuda_tile::ModuleOp>()
          .addPass(mlir::cuda_tile::createSynthesizeDebugInfoScopesPass());
      break;
    case 3:
    default:
      // No pass — parser/verifier baseline coverage.
      return;
  }

  // Failure is fine — passes are allowed to reject inputs they can't handle.
  // We're looking for crashes, not graceful failure.
  (void)pm.run(module);
}

// =============================================================================
// Test execution
// =============================================================================

extern "C" int GrammarinatorTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size == 0) return 0;

  mlir::ScopedDiagnosticHandler handler(
      ctx, [](mlir::Diagnostic &) { return mlir::success(); });

  std::string input(reinterpret_cast<const char *>(Data), Size);

  // Parse without post-parse verification — the parser itself catches
  // syntactic problems; semantic verification is a separate concern and
  // failure there isn't a fuzzer-relevant bug.
  mlir::ParserConfig config(ctx, /*verifyAfterParse=*/false);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(input, config);
  if (!module) return 0;

  uint32_t choice = fnv1a(Data, Size);
  run_one_pass(*module, choice);

  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  GrammarinatorOneInput(&Data, &Size);
  return GrammarinatorTestOneInput(Data, Size);
}

// =============================================================================
// Mutation helpers
// =============================================================================

namespace {

size_t encode_result(Rule *root, uint8_t *Out, size_t MaxOutSize) {
  size_t out_size = grlf_encode_tree(root, Out, MaxOutSize);
  delete root;
  return out_size;
}

size_t grammarinator_mutator_fallback(uint8_t *Data, size_t Size, size_t MaxSize,
                                      unsigned int Seed) {
  return GrammarinatorMutator(Data, Size, MaxSize, Seed);
}

size_t grammarinator_crossover_fallback(const uint8_t *Data1, size_t Size1,
                                        const uint8_t *Data2, size_t Size2,
                                        uint8_t *Out, size_t MaxOutSize,
                                        unsigned int Seed) {
  return GrammarinatorCrossOver(const_cast<uint8_t *>(Data1), Size1, Data2,
                                Size2, Out, MaxOutSize, Seed);
}

}  // namespace

// =============================================================================
// Mutation — single tree
// =============================================================================

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  std::mt19937 rng(Seed);

  Rule *tree1 = grlf_decode_tree(Data, Size);
  if (!tree1) {
    return grammarinator_mutator_fallback(Data, Size, MaxSize, Seed);
  }

  mlir_fuzzer::MutationInput input{tree1, /*tree2=*/nullptr, rng};
  mlir_fuzzer::MutationResult result =
      mlir_fuzzer::MutationRegistry::instance().applyRandomSingleTree(input);

  if (result.success && result.root) {
    return encode_result(result.root, Data, MaxSize);
  }

  delete tree1;
  return grammarinator_mutator_fallback(Data, Size, MaxSize, Seed);
}

// =============================================================================
// Crossover — two trees
// =============================================================================

extern "C" size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                            const uint8_t *Data2, size_t Size2,
                                            uint8_t *Out, size_t MaxOutSize,
                                            unsigned int Seed) {
  std::mt19937 rng(Seed);

  Rule *tree1 = grlf_decode_tree(Data1, Size1);
  Rule *tree2 = grlf_decode_tree(Data2, Size2);

  if (!tree1 || !tree2) {
    delete tree1;
    delete tree2;
    return grammarinator_crossover_fallback(Data1, Size1, Data2, Size2, Out,
                                            MaxOutSize, Seed);
  }

  mlir_fuzzer::MutationInput input{tree1, tree2, rng};
  mlir_fuzzer::MutationResult result =
      mlir_fuzzer::MutationRegistry::instance().applyRandomCrossover(input);

  // tree2 is never consumed by a mutation — always delete it here.
  delete tree2;

  if (result.success && result.root) {
    // On success, tree1 was consumed by the mutation (or is now result.root).
    return encode_result(result.root, Out, MaxOutSize);
  }

  delete tree1;
  return grammarinator_crossover_fallback(Data1, Size1, Data2, Size2, Out,
                                          MaxOutSize, Seed);
}
