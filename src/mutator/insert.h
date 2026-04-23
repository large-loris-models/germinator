// insert.h
//
// The "insert" mutation: find quantifier positions in the recipient tree
// and inject new elements from the donor tree.
//
// Algorithm:
//   1. Find parent rules in the recipient that match insert patterns.
//   2. For each match, find valid insertion positions via greedy matching
//      of the pattern against the recipient's flattened children.
//   3. At each position, insert the placeholder structure demanded by the
//      pattern (possibly multiple nodes — comma-list inserts need both the
//      separator literal and the donor placeholder).
//   4. Verify structural compatibility via context filter.
//   5. Delegate to edit() for parameter substitution and graft.
//
// Data model:
//   Symbol         — one literal or one rule reference
//   AltBranch      — a flat sequence of Symbols; one alternative
//   QuantifierSpec — min/max + vector<AltBranch>
//                    min=max=1 with multiple alternatives means "pick one
//                    branch at this position" (unquantified alternation)
//   MatchElement   — variant<Symbol, AltBranch, QuantifierSpec>
//
// On failure, returns a clone of the recipient unchanged.
#pragma once

#include "context_filter.h"
#include "edit.h"

#include <grammarinator/runtime/Rule.hpp>

#include <climits>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace mlir_fuzzer {

// A single element in a pattern: either a literal token ("(", ",", etc.)
// or a reference to a rule name that must appear at this position.
struct Symbol {
  bool is_rule;        // true = rule reference; false = literal token
  std::string value;   // rule name or literal text

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

// A quantified position. `alternatives` is the set of branches the repeated
// unit can take; at each repetition, any branch may match. For plain
// single-rule quantifiers like `X+` there's one branch containing one Symbol.
// For comma-lists like `X (',' X)*` the repeated-unit quantifier has one
// branch containing the comma literal + rule ref. For `(X | Y)+` there are
// multiple branches, each with one Symbol.
//
// min == max == 1 with multiple alternatives represents an unquantified
// alternation at a single position ("pick exactly one branch here").
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

struct InsertResult {
  grammarinator::runtime::Rule *root;  // mutated tree root (caller owns)
  bool is_fit;
};

// Perform the insert mutation.
InsertResult insert(grammarinator::runtime::Rule *recipient_root,
                    grammarinator::runtime::Rule *donor_root,
                    const InsertPatterns &patterns,
                    const ContextFilter &context_filter, int max_inserts,
                    std::mt19937 &rng,
                    const EditConfig &edit_config = EditConfig{});

}  // namespace mlir_fuzzer