// =============================================================================
// test_mutations.cc — Standalone test & benchmark for the mutation registry
// =============================================================================
//
// Two modes:
//
//   Basic benchmark (default):
//     ./test_mutations <trees_dir> [iterations] [output_dir]
//
//     For each registered mutation, runs `iterations` attempts — each picks
//     two random .grtf trees, decodes them, and runs the mutation through
//     MutationRegistry::instance().  Renders successful results to MLIR and
//     parse-checks them via mlir-opt.  Prints a per-mutation stats table.
//
//   Composition stress test (--compose):
//     ./test_mutations <trees_dir> [trials] [output_dir] --compose
//
//     For chain lengths 5, 10, 20, runs `trials` trials.  Each trial picks a
//     base tree and applies N random mutations (via applyRandom) in sequence,
//     tracking how far the chain got before failure and whether the final
//     tree parses.
//
// Defaults:
//   trees_dir  = seeds/trees/
//   iterations = 100
//   output_dir = build/mutation_test_output
// =============================================================================

#include "src/mutator/registry.h"
#include "grlf_codec.h"
#include <grammarinator/runtime/Population.hpp>
#include <grammarinator/runtime/Rule.hpp>
#include <unistd.h>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using grammarinator::runtime::ParentRule;
using grammarinator::runtime::Rule;

namespace fs = std::filesystem;

// =============================================================================
// Tree validation (debug helper)
// =============================================================================

static void validate_tree_impl(Rule *node, std::unordered_set<Rule *> &seen) {
  assert(node != nullptr && "null node pointer");
  assert(seen.find(node) == seen.end() && "cycle detected in tree");
  seen.insert(node);
  if (node->type != Rule::UnlexerRuleType) {
    auto *p = static_cast<ParentRule *>(node);
    for (size_t i = 0; i < p->children.size(); ++i) {
      Rule *child = p->children[i];
      assert(child != nullptr && "null child pointer");
      assert(child->parent == node && "parent back-pointer mismatch");
      validate_tree_impl(child, seen);
    }
  }
}

static void validate_tree(Rule *root, const char *label) {
  std::unordered_set<Rule *> seen;
  validate_tree_impl(root, seen);
  (void)label;
}

// =============================================================================
// File helpers
// =============================================================================

static std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "ERROR: cannot open file: " << path << "\n";
    return {};
  }
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  f.read(reinterpret_cast<char *>(buf.data()), size);
  return buf;
}

static void write_file(const fs::path &path, const std::string &content) {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "WARNING: cannot write file: " << path << "\n";
    return;
  }
  f << content;
}

static std::string fmt_idx(int idx) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%03d", idx);
  return buf;
}

static std::vector<std::string> list_grtf_files(const std::string &dir) {
  std::vector<std::string> out;
  if (!fs::is_directory(dir)) return out;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".grtf") {
      out.push_back(entry.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// =============================================================================
// MLIR parse-check
// =============================================================================

static mlir::MLIRContext *g_ctx = nullptr;
static std::string g_mlir_opt_path;

static void init_mlir() {
  g_ctx = new mlir::MLIRContext();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  g_ctx->appendDialectRegistry(registry);
  g_ctx->loadAllAvailableDialects();
  g_ctx->allowUnregisteredDialects();
  mlir::registerAllPasses();
}

static bool try_parse(const std::string &mlir_text) {
  if (g_mlir_opt_path.empty()) return false;

  char tmpfile[] = "/tmp/mutation_check_XXXXXX";
  int fd = mkstemp(tmpfile);
  if (fd < 0) return false;
  write(fd, mlir_text.data(), mlir_text.size());
  close(fd);

  std::string cmd = g_mlir_opt_path + " --allow-unregistered-dialect " +
                    tmpfile + " >/dev/null 2>&1";
  int ret = system(cmd.c_str());
  unlink(tmpfile);
  return ret == 0;
}

// =============================================================================
// Tree loading
// =============================================================================

static Rule *decode_file(const std::string &path) {
  auto buf = read_file(path);
  if (buf.empty()) return nullptr;
  return grlf_decode_tree(buf.data(), buf.size());
}

// =============================================================================
// Per-mutation stats
// =============================================================================

struct MutationStats {
  std::string name;
  int attempts = 0;
  int applicable = 0;
  int succeeded = 0;
  int parse_ok = 0;
  int fit_count = 0;
  long long total_apply_us = 0;  // only across successful apply() calls
};

static void print_stats_table(const std::vector<MutationStats> &stats,
                              int iterations) {
  std::cout << "\n=== Mutation Benchmark (" << iterations
            << " iterations each) ===\n";
  std::cout << std::left << std::setw(18) << "Mutation"
            << std::setw(12) << "Applicable"
            << std::setw(11) << "Succeeded"
            << std::setw(9)  << "ParseOK"
            << std::setw(6)  << "Fit"
            << "AvgTime(us)\n";
  std::cout << std::left << std::setw(18) << "────────"
            << std::setw(12) << "──────────"
            << std::setw(11) << "─────────"
            << std::setw(9)  << "───────"
            << std::setw(6)  << "───"
            << "───────────\n";
  for (const auto &s : stats) {
    std::string applicable =
        std::to_string(s.applicable) + "/" + std::to_string(s.attempts);
    std::string succeeded =
        std::to_string(s.succeeded) + "/" + std::to_string(s.applicable);
    std::string parse_ok =
        std::to_string(s.parse_ok) + "/" + std::to_string(s.succeeded);
    long long avg_us =
        s.succeeded > 0 ? s.total_apply_us / s.succeeded : 0;

    std::cout << std::left << std::setw(18) << s.name
              << std::setw(12) << applicable
              << std::setw(11) << succeeded
              << std::setw(9)  << parse_ok
              << std::setw(6)  << s.fit_count
              << avg_us << "\n";
  }
  std::cout << "\n";
}

// =============================================================================
// Basic benchmark mode
// =============================================================================

static int run_basic(const std::string &trees_dir, int iterations,
                     const std::string &output_dir, uint32_t base_seed) {
  auto tree_files = list_grtf_files(trees_dir);
  if (tree_files.size() < 2) {
    std::cerr << "ERROR: need at least 2 .grtf files in " << trees_dir
              << " (found " << tree_files.size() << ")\n";
    return 1;
  }

  fs::create_directories(output_dir);

  std::cout << "Trees dir:   " << trees_dir << " (" << tree_files.size()
            << " files)\n";
  std::cout << "Iterations:  " << iterations << "\n";
  std::cout << "Output dir:  " << output_dir << "\n";
  std::cout << "Base seed:   " << base_seed << "\n\n";

  const auto &registry = mlir_fuzzer::MutationRegistry::instance();
  const auto &mutations = registry.mutations();

  std::vector<MutationStats> stats(mutations.size());
  for (size_t i = 0; i < mutations.size(); ++i) {
    stats[i].name = mutations[i]->name();
  }

  std::uniform_int_distribution<size_t> file_pick(0, tree_files.size() - 1);

  for (size_t mi = 0; mi < mutations.size(); ++mi) {
    const auto &mutation = *mutations[mi];
    auto &s = stats[mi];
    s.attempts = iterations;

    std::cout << "[test] running " << mutation.name() << "...\n";

    for (int i = 0; i < iterations; ++i) {
      std::mt19937 rng(base_seed + static_cast<uint32_t>(mi) * 100000u +
                       static_cast<uint32_t>(i));

      size_t i1 = file_pick(rng);
      size_t i2 = file_pick(rng);
      if (tree_files.size() > 1 && i2 == i1) {
        i2 = (i1 + 1) % tree_files.size();
      }

      Rule *tree1 = decode_file(tree_files[i1]);
      Rule *tree2 =
          mutation.needsDonor() ? decode_file(tree_files[i2]) : nullptr;

      if (!tree1 || (mutation.needsDonor() && !tree2)) {
        std::cerr << "ERROR: decode failed on iteration " << i << "\n";
        delete tree1;
        delete tree2;
        continue;
      }

      validate_tree(tree1, "tree1 post-decode");
      if (tree2) validate_tree(tree2, "tree2 post-decode");

      mlir_fuzzer::MutationInput input{tree1, tree2, rng};

      if (!mutation.canApply(input)) {
        delete tree1;
        delete tree2;
        continue;
      }
      ++s.applicable;

      auto t0 = std::chrono::high_resolution_clock::now();
      mlir_fuzzer::MutationResult result = mutation.apply(input);
      auto t1 = std::chrono::high_resolution_clock::now();

      // tree2 is never consumed by apply() per the ownership contract.
      delete tree2;

      if (!result.root) {
        // apply() failed; contract says tree1 is untouched, free it.
        delete tree1;
        continue;
      }
      // tree1 was consumed; result.root is the new owner.
      ++s.succeeded;
      if (result.success) ++s.fit_count;
      s.total_apply_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
              .count();

      std::string mlir_text = result.root->format(Rule::StrFormat);
      delete result.root;

      bool parsed = try_parse(mlir_text);
      std::string stem = mutation.name() + "_" + fmt_idx(i);
      fs::path out_path =
          fs::path(output_dir) /
          (parsed ? (stem + ".mlir") : (stem + ".fail.mlir"));
      write_file(out_path, mlir_text);
      if (parsed) ++s.parse_ok;
    }
  }

  print_stats_table(stats, iterations);
  std::cout << "Results written to: " << output_dir << "/\n";
  return 0;
}

// =============================================================================
// Composition stress test mode
// =============================================================================

struct ChainStats {
  int chain_length = 0;
  int trials = 0;
  int completed = 0;   // chains that ran all N mutations without failure
  int parsed = 0;      // chains that completed AND whose final output parsed
};

static void print_chain_stats(const std::vector<ChainStats> &stats) {
  std::cout << "\n=== Composition Stress Test ===\n";
  for (const auto &s : stats) {
    double parse_pct =
        s.completed > 0 ? (100.0 * s.parsed / s.completed) : 0.0;
    char label[64];
    std::snprintf(label, sizeof(label), "Chain length %-3d", s.chain_length);
    std::cout << label << ": " << s.completed << "/" << s.trials
              << " completed, " << s.parsed << "/" << s.completed
              << " parsed (" << std::fixed << std::setprecision(1) << parse_pct
              << "%)\n";
  }
  std::cout << "\n";
}

static int run_compose(const std::string &trees_dir, int trials,
                       const std::string &output_dir, uint32_t base_seed) {
  auto tree_files = list_grtf_files(trees_dir);
  if (tree_files.size() < 2) {
    std::cerr << "ERROR: need at least 2 .grtf files in " << trees_dir
              << " (found " << tree_files.size() << ")\n";
    return 1;
  }

  fs::create_directories(output_dir);

  std::cout << "Trees dir:   " << trees_dir << " (" << tree_files.size()
            << " files)\n";
  std::cout << "Trials:      " << trials << "\n";
  std::cout << "Output dir:  " << output_dir << "\n";
  std::cout << "Base seed:   " << base_seed << "\n\n";

  const auto &registry = mlir_fuzzer::MutationRegistry::instance();

  const std::vector<int> chain_lengths = {5, 10, 20};
  std::vector<ChainStats> all_stats;

  std::uniform_int_distribution<size_t> file_pick(0, tree_files.size() - 1);

  for (int N : chain_lengths) {
    ChainStats cs;
    cs.chain_length = N;
    cs.trials = trials;

    std::cout << "[test] running chain length " << N << "...\n";

    for (int t = 0; t < trials; ++t) {
      std::mt19937 rng(base_seed + static_cast<uint32_t>(N) * 1000000u +
                       static_cast<uint32_t>(t));

      Rule *current = decode_file(tree_files[file_pick(rng)]);
      if (!current) continue;

      int steps_done = 0;
      bool failed = false;
      for (int step = 0; step < N; ++step) {
        Rule *donor = decode_file(tree_files[file_pick(rng)]);
        if (!donor) {
          failed = true;
          break;
        }

        mlir_fuzzer::MutationInput input{current, donor, rng};
        mlir_fuzzer::MutationResult result = registry.applyRandom(input);

        delete donor;

        if (!result.root) {
          // apply() leaves tree1 untouched on failure; caller still owns it.
          failed = true;
          break;
        }
        // apply() consumed `current`; result.root is the new tree.
        current = result.root;
        validate_tree(current, "post-mutation");
        ++steps_done;
      }

      if (!failed && steps_done == N) {
        ++cs.completed;
        std::string mlir_text = current->format(Rule::StrFormat);
        bool parsed = try_parse(mlir_text);
        std::string stem = "chain" + std::to_string(N) + "_" + fmt_idx(t);
        fs::path out_path =
            fs::path(output_dir) /
            (parsed ? (stem + ".mlir") : (stem + ".fail.mlir"));
        write_file(out_path, mlir_text);
        if (parsed) ++cs.parsed;
      }

      delete current;
    }

    all_stats.push_back(cs);
  }

  print_chain_stats(all_stats);
  std::cout << "Results written to: " << output_dir << "/\n";
  return 0;
}

// =============================================================================
// main
// =============================================================================

static void usage(const char *prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " <trees_dir> [iterations] [output_dir]\n"
            << "  " << prog
            << " <trees_dir> [trials] [output_dir] --compose\n";
}

int main(int argc, char **argv) {
  // Extract --compose flag from anywhere in argv.
  bool compose_mode = false;
  std::vector<std::string> pos_args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--compose") == 0) {
      compose_mode = true;
    } else {
      pos_args.emplace_back(argv[i]);
    }
  }

  if (pos_args.empty()) {
    usage(argv[0]);
    return 1;
  }

  const std::string trees_dir = pos_args[0];
  const int count = pos_args.size() >= 2 ? std::atoi(pos_args[1].c_str())
                                         : (compose_mode ? 50 : 100);
  const std::string output_dir = pos_args.size() >= 3
                                     ? pos_args[2]
                                     : std::string("build/mutation_test_output");

  if (count <= 0) {
    std::cerr << "ERROR: count must be > 0\n";
    return 1;
  }

  g_mlir_opt_path = []() {
    const char *env = getenv("MLIR_OPT_PATH");
    return env ? std::string(env) : std::string();
  }();

  const uint32_t base_seed = []() {
    const char *env = getenv("MUTATION_TEST_SEED");
    return env ? static_cast<uint32_t>(std::atoi(env)) : 42u;
  }();

  init_mlir();

  if (compose_mode) {
    return run_compose(trees_dir, count, output_dir, base_seed);
  }
  return run_basic(trees_dir, count, output_dir, base_seed);
}
