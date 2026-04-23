// edit_mutation.h
//
// The "edit" mutation: graft a donor subtree into a recipient tree with
// parameter substitution.  See edit_mutation.cc for the full algorithm.
//
// The low-level graft routine is also exposed as graft_fragment() so that
// other mutations (e.g. InsertMutation) can delegate parameter substitution
// once they've chosen a placeholder position.
#pragma once

#include "base.h"

#include <grammarinator/runtime/Rule.hpp>

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir_fuzzer {

// Parameter blacklist: child_name -> list of parent names.  If the parent
// list contains "*" or is empty, the child is blacklisted under any parent.
using ParameterBlacklist =
    std::unordered_map<std::string, std::vector<std::string>>;

// Should-substitute set: if a parameter node matches this set but is NOT
// substituted, the edit result is marked unfit.
using ShouldSubstituteSet =
    std::unordered_map<std::string, std::vector<std::string>>;

struct EditConfig {
  ParameterBlacklist parameter_blacklist;
  ShouldSubstituteSet should_substitute;
  EditConfig() = default;
};

// Shared default configuration.  Both EditMutation and InsertMutation start
// from this so that insert's delegated graft uses the same blacklist and
// fitness rules edit itself would.
EditConfig default_edit_config();

struct EditFragmentResult {
  grammarinator::runtime::Rule *root;  // mutated tree root (caller owns)
  bool is_fit;                         // fitness check passed
};

// Low-level graft: replace recipient_node with donor_node's content, running
// parameter substitution if the donor has children.  Both nodes must live in
// trees rooted at recipient_root and donor_root respectively.
//
// On success, returns the new root of the recipient tree (in place).  On
// failure, returns {nullptr, false} and leaves both trees untouched.
//
// Exposed for use by other mutations (notably InsertMutation) that need to
// graft a donor fragment after choosing an insertion placeholder.
EditFragmentResult graft_fragment(grammarinator::runtime::Rule *recipient_node,
                                  grammarinator::runtime::Rule *donor_node,
                                  grammarinator::runtime::Rule *recipient_root,
                                  grammarinator::runtime::Rule *donor_root,
                                  std::mt19937 &rng,
                                  const EditConfig &config);

class EditMutation : public TreeMutation {
 public:
  EditMutation();

  std::string name() const override { return "edit"; }
  bool needsDonor() const override { return true; }
  bool canApply(const MutationInput &input) const override;
  MutationResult apply(const MutationInput &input) const override;

 private:
  EditConfig edit_config_;
  int max_depth_;
  int k_ancestors_;
  int l_siblings_;
  int r_siblings_;
};

}  // namespace mlir_fuzzer
