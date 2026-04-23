// registry.h
//
// Singleton registry of tree-level mutations.
//
// Adding a new mutation requires:
//   1. One new header/source pair under src/mutator/tree_mutations/ that
//      inherits from TreeMutation.
//   2. One line in MutationRegistry::instance() that registers it.
//
// The harness only sees MutationRegistry + MutationInput/MutationResult —
// it never names a specific mutation.
#pragma once

#include "base.h"

#include <memory>
#include <vector>

namespace mlir_fuzzer {

class MutationRegistry {
 public:
  // First access constructs and registers all built-in mutations.
  static MutationRegistry &instance();

  // Register a mutation.  Normally only called from instance().
  void add(std::unique_ptr<TreeMutation> m);

  // All registered mutations, in registration order.
  const std::vector<std::unique_ptr<TreeMutation>> &mutations() const {
    return mutations_;
  }

  // Pick a random applicable mutation (any kind) and run it.
  MutationResult applyRandom(const MutationInput &input) const;

  // Pick a random applicable mutation where needsDonor() == true.
  MutationResult applyRandomCrossover(const MutationInput &input) const;

  // Pick a random applicable mutation where needsDonor() == false.
  MutationResult applyRandomSingleTree(const MutationInput &input) const;

 private:
  std::vector<std::unique_ptr<TreeMutation>> mutations_;
};

}  // namespace mlir_fuzzer
