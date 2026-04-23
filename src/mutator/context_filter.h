// context_filter.h
//
// Structural compatibility check used by both the edit and insert mutations.
//
// A donor node is considered compatible with a recipient node if the donor's
// surrounding context (ancestors, left siblings, right siblings) can be found
// in the recipient's context.  The check is asymmetric:
//
//   - If the donor runs out of tree before k/l/r steps are exhausted, we
//     break early and return true — the donor's context is shorter than the
//     limit, and the recipient satisfies all of it.
//
//   - If the recipient runs out of tree while the donor still has context
//     remaining, we return false — the recipient cannot satisfy the donor's
//     requirements.

#pragma once

#include <grammarinator/runtime/Rule.hpp>

namespace grammarinator {
namespace runtime {
class Rule;
class ParentRule;
} // namespace runtime
} // namespace grammarinator

namespace mlir_fuzzer {

struct ContextFilter {
  int k_ancestors; // number of ancestor levels to check
  int l_siblings;  // number of left siblings to check
  int r_siblings;  // number of right siblings to check

  // Default constructor uses the values passed via --k-ancestors, etc.
  explicit ContextFilter(int k = 4, int l = 4, int r = 4)
      : k_ancestors(k), l_siblings(l), r_siblings(r) {}

  // Returns true if donor_node is structurally compatible with recipient_node.
  bool verify(const grammarinator::runtime::Rule *recipient,
              const grammarinator::runtime::Rule *donor) const;

private:
  // Check k ancestors: walk up from both nodes simultaneously.
  bool verify_ancestors(const grammarinator::runtime::Rule *recipient,
                        const grammarinator::runtime::Rule *donor) const;

  // Check l left siblings: walk left from both nodes simultaneously.
  bool verify_left_siblings(const grammarinator::runtime::Rule *recipient,
                            const grammarinator::runtime::Rule *donor) const;

  // Check r right siblings: walk right from both nodes simultaneously.
  bool verify_right_siblings(const grammarinator::runtime::Rule *recipient,
                             const grammarinator::runtime::Rule *donor) const;
};

} // namespace mlir_fuzzer
