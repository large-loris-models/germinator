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
// =============================================================================

#include "grlf.h"
#include "grlf_codec.h"

#include "context_filter.h"
#include "edit.h"
#include "insert.h"
#include "insert_patterns.h"

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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using grammarinator::runtime::Annotations;
using grammarinator::runtime::Rule;
using NodeKey = Annotations::NodeKey;

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

int g_max_depth = 30;
mlir_fuzzer::ContextFilter g_context_filter{4, 4, 4};

const mlir_fuzzer::InsertPatterns &get_patterns() {
  static const mlir_fuzzer::InsertPatterns patterns =
      mlir_fuzzer::get_insert_patterns();
  return patterns;
}

mlir_fuzzer::EditConfig g_edit_config = []() {
  mlir_fuzzer::EditConfig cfg;
  cfg.parameter_blacklist["string_literal"] = {"generic_operation"};
  cfg.should_substitute["ssa_id"] = {"ssa_use"};
  cfg.should_substitute["non_function_type"] = {"*"};
  return cfg;
}();

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

  if (const char *env_depth = getenv("GRAMMARINATOR_MAX_DEPTH")) {
    g_max_depth = atoi(env_depth);
  }

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
// Mutation — single tree, 100% Grammarinator
// =============================================================================

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  return GrammarinatorMutator(Data, Size, MaxSize, Seed);
}

// =============================================================================
// Crossover helpers (unchanged from previous version)
// =============================================================================

namespace {

std::pair<Rule *, Rule *>
select_edit_pair(Annotations &r_annot, Annotations &d_annot, int max_depth,
                 const mlir_fuzzer::ContextFilter &context_filter,
                 std::mt19937 &rng) {
  const auto &r_by_name = r_annot.rules_by_name();
  const auto &d_by_name = d_annot.rules_by_name();
  const auto &r_info = r_annot.node_info();
  const auto &d_info = d_annot.node_info();

  std::vector<NodeKey> common_keys;
  for (const auto &[key, nodes] : r_by_name) {
    if (d_by_name.count(key)) {
      common_keys.push_back(key);
    }
  }
  std::shuffle(common_keys.begin(), common_keys.end(), rng);

  for (const auto &key : common_keys) {
    std::vector<Rule *> r_nodes = r_by_name.at(key);
    std::vector<Rule *> d_nodes = d_by_name.at(key);
    std::shuffle(r_nodes.begin(), r_nodes.end(), rng);
    std::shuffle(d_nodes.begin(), d_nodes.end(), rng);

    for (Rule *r_node : r_nodes) {
      auto r_it = r_info.find(r_node);
      if (r_it == r_info.end()) continue;
      int r_level = r_it->second.level;

      for (Rule *d_node : d_nodes) {
        auto d_it = d_info.find(d_node);
        if (d_it == d_info.end()) continue;
        int d_depth = d_it->second.depth;

        if (r_level + d_depth > max_depth) continue;
        if (!context_filter.verify(r_node, d_node)) continue;

        return {r_node, d_node};
      }
    }
  }
  return {nullptr, nullptr};
}

size_t encode_result(Rule *root, uint8_t *Out, size_t MaxOutSize) {
  size_t out_size = grlf_encode_tree(root, Out, MaxOutSize);
  delete root;
  return out_size;
}

size_t grammarinator_fallback(const uint8_t *Data1, size_t Size1,
                              const uint8_t *Data2, size_t Size2, uint8_t *Out,
                              size_t MaxOutSize, unsigned int Seed) {
  return GrammarinatorCrossOver(const_cast<uint8_t *>(Data1), Size1, Data2,
                                Size2, Out, MaxOutSize, Seed);
}

}  // namespace

// =============================================================================
// Crossover — two trees, 50/50 edit/insert
// =============================================================================

extern "C" size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                            const uint8_t *Data2, size_t Size2,
                                            uint8_t *Out, size_t MaxOutSize,
                                            unsigned int Seed) {
  std::mt19937 rng(Seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float roll = dist(rng);

  Rule *tree1 = grlf_decode_tree(Data1, Size1);
  Rule *tree2 = grlf_decode_tree(Data2, Size2);

  if (!tree1 || !tree2) {
    delete tree1;
    delete tree2;
    return grammarinator_fallback(Data1, Size1, Data2, Size2, Out, MaxOutSize,
                                  Seed);
  }

  if (roll < 0.5f) {
    Annotations r_annot(tree1);
    Annotations d_annot(tree2);

    Rule *r_node = nullptr;
    Rule *d_node = nullptr;
    std::tie(r_node, d_node) =
        select_edit_pair(r_annot, d_annot, g_max_depth, g_context_filter, rng);

    if (r_node && d_node) {
      mlir_fuzzer::EditResult result =
          mlir_fuzzer::edit(r_node, d_node, tree1, tree2, rng, g_edit_config);

      delete tree2;

      if (result.root) {
        return encode_result(result.root, Out, MaxOutSize);
      }
      return grammarinator_fallback(Data1, Size1, Data2, Size2, Out, MaxOutSize,
                                    Seed);
    }

    delete tree1;
    delete tree2;
    return grammarinator_fallback(Data1, Size1, Data2, Size2, Out, MaxOutSize,
                                  Seed);
  }

  {
    mlir_fuzzer::InsertResult result =
        mlir_fuzzer::insert(tree1, tree2, get_patterns(), g_context_filter,
                            /*max_inserts=*/1, rng, g_edit_config);

    delete tree2;
    return encode_result(result.root, Out, MaxOutSize);
  }
}