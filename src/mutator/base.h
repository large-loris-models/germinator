// base.h
//
// Abstract interface for a tree-level mutation.
//
// A TreeMutation operates on one or two Grammarinator trees and produces a
// mutated output tree.  It has no knowledge of MLIR, Centipede, or the
// harness; it only manipulates Grammarinator Rule graphs.
//
// Ownership contract for apply():
//   - input.tree1 / input.tree2 are NOT owned by apply(); the caller keeps
//     ownership for the lifetime of the call.
//   - On success (MutationResult{root != nullptr, success = true}), apply()
//     MAY mutate input.tree1 in place and takes ownership of it.  The
//     returned root is owned by the caller from that point on.
//     input.tree2 is never consumed.
//   - On failure (MutationResult{root = nullptr, success = false}), apply()
//     must leave input.tree1 and input.tree2 unmodified and untaken; the
//     caller still owns both.
//
// To add a new mutation:
//   1. Create a header/source pair under src/mutator/tree_mutations/.
//   2. Inherit from TreeMutation and implement the four virtuals.
//   3. Add one std::make_unique<Foo>() line in MutationRegistry::instance().
//
// No harness edits, no dispatch changes.
#pragma once

#include <grammarinator/runtime/Rule.hpp>

#include <random>
#include <string>

namespace mlir_fuzzer {

// Input bundle for one mutation application.
//
// tree1 — primary tree (always present).
// tree2 — donor tree; nullptr for single-tree mutations.
// rng   — random engine; the mutation uses this for all randomness so that
//         a given seed reproduces a given outcome.
struct MutationInput {
  grammarinator::runtime::Rule *tree1;
  grammarinator::runtime::Rule *tree2;
  std::mt19937 &rng;
};

// Result of one mutation application.
//
// root        — resulting tree (owned by caller); nullptr = mutation failed.
// success     — true iff the mutation produced a usable tree.
// description — optional human-readable description of what was done.
struct MutationResult {
  grammarinator::runtime::Rule *root;
  bool success;
  std::string description;
};

class TreeMutation {
 public:
  virtual ~TreeMutation() = default;

  // Human-readable name (e.g. "edit", "insert").
  virtual std::string name() const = 0;

  // true for crossover-style mutations that require a donor tree (tree2);
  // false for single-tree mutations.
  virtual bool needsDonor() const = 0;

  // Cheap pre-check: could this mutation plausibly run on the given input?
  // Return false to skip this mutation in the registry's pool.  A true
  // result does not guarantee apply() will succeed.
  virtual bool canApply(const MutationInput &input) const = 0;

  // Run the mutation.  See ownership contract above.
  virtual MutationResult apply(const MutationInput &input) const = 0;
};

}  // namespace mlir_fuzzer
