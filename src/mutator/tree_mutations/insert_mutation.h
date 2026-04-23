// insert_mutation.h
//
// The "insert" mutation: find quantifier positions in the recipient tree and
// inject new elements drawn from the donor tree, delegating parameter
// substitution to the edit mutation's graft_fragment().
//
// Pattern data model (emitted by scripts/build/generate_insert_patterns.py):
//   Symbol         — one literal or one rule reference
//   AltBranch      — a flat sequence of Symbols; one alternative
//   QuantifierSpec — min/max + vector<AltBranch>; min=max=1 with multiple
//                    alternatives represents a "pick one branch" position
//   MatchElement   — variant<Symbol, AltBranch, QuantifierSpec>
//
// InsertMutation requires a donor: candidate child rules are looked up in
// the donor's rules_by_name index before being grafted.
#pragma once

#include "base.h"
#include "context_filter.h"
#include "edit_mutation.h"

#include <grammarinator/runtime/Rule.hpp>

#include <climits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace mlir_fuzzer {

// One literal or rule-reference position in a pattern.
struct Symbol {
  bool is_rule;
  std::string value;

  bool operator==(const Symbol &o) const = default;
  bool operator<(const Symbol &o) const {
    if (is_rule != o.is_rule) return is_rule < o.is_rule;
    return value < o.value;
  }
};

// One alternative in an alternation — a flat sequence of symbols that must
// all match in order.
struct AltBranch {
  std::vector<Symbol> symbols;

  bool operator==(const AltBranch &o) const = default;
  bool operator<(const AltBranch &o) const { return symbols < o.symbols; }
};

// A quantified position.  min == max == 1 with multiple alternatives means
// "pick one branch here" (unquantified alternation).
struct QuantifierSpec {
  int min;
  int max;  // INT_MAX for unbounded
  std::vector<AltBranch> alternatives;

  bool operator==(const QuantifierSpec &o) const = default;
  bool operator<(const QuantifierSpec &o) const {
    if (min != o.min) return min < o.min;
    if (max != o.max) return max < o.max;
    return alternatives < o.alternatives;
  }
};

using MatchElement = std::variant<Symbol, AltBranch, QuantifierSpec>;

struct InsertPattern {
  std::vector<MatchElement> match_pattern;
  std::unordered_set<std::string> child_rules;
};

using InsertPatterns = std::unordered_map<std::string, InsertPattern>;

class InsertMutation : public TreeMutation {
 public:
  InsertMutation();

  std::string name() const override { return "insert"; }
  bool needsDonor() const override { return true; }
  bool canApply(const MutationInput &input) const override;
  MutationResult apply(const MutationInput &input) const override;

 private:
  InsertPatterns patterns_;
  EditConfig edit_config_;
  int k_ancestors_;
  int l_siblings_;
  int r_siblings_;
  int max_inserts_;
};

}  // namespace mlir_fuzzer
