// registry.cc
//
// Defines the singleton and the three applyRandom* dispatch variants.
//
// To add a new mutation: include its header and add one std::make_unique
// line in instance().  No other changes anywhere in the tree are required.

#include "registry.h"

#include "tree_mutations/edit_mutation.h"
#include "tree_mutations/insert_mutation.h"

#include <cstddef>
#include <random>
#include <vector>

namespace mlir_fuzzer {

namespace {

MutationResult apply_from_pool(const std::vector<TreeMutation *> &pool,
                               const MutationInput &input) {
  if (pool.empty()) return {nullptr, false, {}};
  std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
  TreeMutation *chosen = pool[dist(input.rng)];
  return chosen->apply(input);
}

}  // namespace

void MutationRegistry::add(std::unique_ptr<TreeMutation> m) {
  mutations_.push_back(std::move(m));
}

MutationResult MutationRegistry::applyRandom(const MutationInput &input) const {
  std::vector<TreeMutation *> applicable;
  for (const auto &m : mutations_) {
    if (m->canApply(input)) applicable.push_back(m.get());
  }
  return apply_from_pool(applicable, input);
}

MutationResult MutationRegistry::applyRandomCrossover(
    const MutationInput &input) const {
  std::vector<TreeMutation *> applicable;
  for (const auto &m : mutations_) {
    if (m->needsDonor() && m->canApply(input)) applicable.push_back(m.get());
  }
  return apply_from_pool(applicable, input);
}

MutationResult MutationRegistry::applyRandomSingleTree(
    const MutationInput &input) const {
  std::vector<TreeMutation *> applicable;
  for (const auto &m : mutations_) {
    if (!m->needsDonor() && m->canApply(input)) applicable.push_back(m.get());
  }
  return apply_from_pool(applicable, input);
}

MutationRegistry &MutationRegistry::instance() {
  static MutationRegistry reg;
  static bool initialized = false;
  if (!initialized) {
    reg.add(std::make_unique<EditMutation>());
    reg.add(std::make_unique<InsertMutation>());
    // Add new mutations here — one line per mutation.
    initialized = true;
  }
  return reg;
}

}  // namespace mlir_fuzzer
