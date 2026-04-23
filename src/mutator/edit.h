// edit.h
//
// The "edit" mutation: graft a donor subtree into a recipient tree with
// parameter substitution.
//
// Given two nodes of the same rule name (recipient_node, donor_node), edit:
//   1. Clones the full donor tree so context indexing sees the complete tree.
//   2. Locates the clone of donor_node within the cloned tree.
//   3. Detects "parameters" in the donor fragment — nodes that appear both
//      inside the donor subtree and outside it (in the donor's context) with
//      identical serialized text.  Nodes in the parameter blacklist are
//      skipped.
//   4. Walks common ancestors of recipient and donor (stopping when parent
//      names diverge) to find concrete values for each parameter from the
//      recipient's context.
//   5. Substitutes parameters in the cloned donor fragment.
//   6. Checks fitness: were required parameters (should_substitute) fulfilled?
//   7. Grafts the adapted donor fragment in place of the recipient node.
//
// If the donor has no children, falls back to a simple graft with no
// parameter substitution (equivalent to Python's recombine()).
//
// Callers must pass the tree roots explicitly because find_root() is
// unreliable after Individual destructors have detached trees from their
// <ROOT> wrappers.
//
// The returned root pointer is owned by the caller.
#pragma once

#include <grammarinator/runtime/Rule.hpp>

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir_fuzzer {

// Parameter blacklist: child_name → list of parent names.
// If the parent list contains "*" or is empty, the child is blacklisted
// under any parent.
// Example: {"string_literal": {"generic_operation"}} means skip
// string_literal nodes whose parent is generic_operation.
using ParameterBlacklist =
    std::unordered_map<std::string, std::vector<std::string>>;

// Should-substitute set: child_name → list of parent names.
// If a parameter node matches this set but is NOT substituted,
// the edit result is marked as unfit (is_fit = false).
// Format same as ParameterBlacklist.
using ShouldSubstituteSet =
    std::unordered_map<std::string, std::vector<std::string>>;

// Configuration for the edit mutation, loaded from mutation_config.toml.
struct EditConfig {
  ParameterBlacklist parameter_blacklist;
  ShouldSubstituteSet should_substitute;

  // Default constructor: no blacklist, no fitness requirements.
  EditConfig() = default;
};

struct EditResult {
  grammarinator::runtime::Rule *root; // root of the mutated tree (caller owns)
  bool is_fit;                        // fitness check passed
};

// Perform the edit mutation.
//
// recipient_node  — node in the recipient tree to replace
// donor_node      — node from the donor tree to graft in
//                   (must have the same rule name as recipient_node)
// recipient_root  — root of the recipient tree (tree1)
// donor_root      — root of the donor tree (tree2)
// rng             — random engine
// config          — parameter blacklist and fitness criteria
//
// Returns {root_of_mutated_tree, is_fit} on success.
// Returns {nullptr, false} if the edit cannot be performed.
EditResult edit(grammarinator::runtime::Rule *recipient_node,
                grammarinator::runtime::Rule *donor_node,
                grammarinator::runtime::Rule *recipient_root,
                grammarinator::runtime::Rule *donor_root, std::mt19937 &rng,
                const EditConfig &config = EditConfig{});

} // namespace mlir_fuzzer
