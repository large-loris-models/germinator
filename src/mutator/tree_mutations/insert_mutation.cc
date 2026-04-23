// insert_mutation.cc
//
// Implementation of the InsertMutation.  See insert_mutation.h for the data
// model.
//
// Algorithm:
//   1. Find parent rules in the recipient that match one of the registered
//      insert patterns.
//   2. For each match, find valid insertion positions via greedy matching
//      of the pattern against the recipient's flattened children.
//   3. At each position, insert the placeholder structure demanded by the
//      pattern (a comma-list insert needs both the separator literal and
//      the donor placeholder).
//   4. Verify structural compatibility with the candidate donor via the
//      context filter.
//   5. Delegate to graft_fragment() for parameter substitution and graft.

#include "insert_mutation.h"

#include "insert_patterns.h"

#include <grammarinator/runtime/Population.hpp>
#include <grammarinator/runtime/Rule.hpp>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace mlir_fuzzer {

using grammarinator::runtime::Annotations;
using grammarinator::runtime::ParentRule;
using grammarinator::runtime::Rule;
using grammarinator::runtime::UnparserRule;
using NodeKey = Annotations::NodeKey;

namespace {

// ---------------------------------------------------------------------------
// flatten_children
// ---------------------------------------------------------------------------
//
// Collect the "effective" children of a parent, unwrapping the structural
// wrapper nodes Grammarinator's C++ tree uses that are transparent in the
// Python tree's nodes_by_name index.

void flatten_children_impl(Rule *node, std::vector<Rule *> &out) {
  if (node->type == Rule::UnlexerRuleType ||
      node->type == Rule::UnparserRuleType) {
    out.push_back(node);
    return;
  }
  auto *p = static_cast<ParentRule *>(node);
  for (auto *child : p->children) {
    flatten_children_impl(child, out);
  }
}

std::vector<Rule *> flatten_children(ParentRule *parent) {
  std::vector<Rule *> result;
  for (auto *child : parent->children) {
    flatten_children_impl(child, result);
  }
  return result;
}

bool name_matches_literal(Rule *node, const std::string &literal) {
  if (node->name == literal) return true;
  if (node->type == Rule::UnlexerRuleType) {
    auto *unlexer = static_cast<grammarinator::runtime::UnlexerRule *>(node);
    if (unlexer->src == literal) return true;
  }
  return false;
}

bool symbol_matches(Rule *node, const Symbol &sym) {
  if (sym.is_rule) return node->name == sym.value;
  return name_matches_literal(node, sym.value);
}

// Match every symbol in `branch` in order starting from `start`.  Returns
// the number of children consumed, or nullopt on failure.
std::optional<int> try_match_branch(const std::vector<Rule *> &flat, int start,
                                    const AltBranch &branch) {
  if (start + static_cast<int>(branch.symbols.size()) >
      static_cast<int>(flat.size())) {
    return std::nullopt;
  }
  for (size_t i = 0; i < branch.symbols.size(); ++i) {
    if (!symbol_matches(flat[start + i], branch.symbols[i])) {
      return std::nullopt;
    }
  }
  return static_cast<int>(branch.symbols.size());
}

struct RepetitionMatch {
  int start_index;
  int branch_index;
  int span;
};

struct QuantifierMatch {
  const QuantifierSpec *spec;
  std::vector<RepetitionMatch> reps;  // empty allowed if spec->min == 0
};

// Walk the pattern and the flattened children in lockstep.  On success,
// return one QuantifierMatch per QuantifierSpec in the pattern (in order).
std::optional<std::vector<QuantifierMatch>> match_pattern(
    ParentRule *recipient_parent, const InsertPattern &pattern) {
  auto flat = flatten_children(recipient_parent);
  int idx = 0;
  std::vector<QuantifierMatch> out;

  for (const auto &elem : pattern.match_pattern) {
    if (const auto *sym = std::get_if<Symbol>(&elem)) {
      if (idx >= static_cast<int>(flat.size())) return std::nullopt;
      if (!symbol_matches(flat[idx], *sym)) return std::nullopt;
      ++idx;
      continue;
    }

    if (const auto *branch = std::get_if<AltBranch>(&elem)) {
      auto n = try_match_branch(flat, idx, *branch);
      if (!n) return std::nullopt;
      idx += *n;
      continue;
    }

    const auto &spec = std::get<QuantifierSpec>(elem);
    QuantifierMatch qm{&spec, {}};

    if (spec.min == 1 && spec.max == 1) {
      bool matched = false;
      for (size_t bi = 0; bi < spec.alternatives.size(); ++bi) {
        auto n = try_match_branch(flat, idx, spec.alternatives[bi]);
        if (n) {
          qm.reps.push_back({idx, static_cast<int>(bi), *n});
          idx += *n;
          matched = true;
          break;
        }
      }
      if (!matched) return std::nullopt;
      out.push_back(std::move(qm));
      continue;
    }

    while (true) {
      if (spec.max != INT_MAX &&
          static_cast<int>(qm.reps.size()) >= spec.max) {
        break;
      }
      bool any_branch_matched = false;
      for (size_t bi = 0; bi < spec.alternatives.size(); ++bi) {
        auto n = try_match_branch(flat, idx, spec.alternatives[bi]);
        if (n) {
          qm.reps.push_back({idx, static_cast<int>(bi), *n});
          idx += *n;
          any_branch_matched = true;
          break;
        }
      }
      if (!any_branch_matched) break;
    }

    if (static_cast<int>(qm.reps.size()) < spec.min) return std::nullopt;
    out.push_back(std::move(qm));
  }

  return out;
}

bool donor_has_required_rules(Annotations &donor_annot,
                              const InsertPattern &pattern) {
  const auto &by_name = donor_annot.rules_by_name();
  for (const auto &rule_name : pattern.child_rules) {
    if (by_name.find(NodeKey(rule_name)) == by_name.end()) return false;
  }
  return true;
}

// Pick the first branch containing at least one rule-ref Symbol.
std::optional<int> pick_insertable_branch(const QuantifierSpec &spec) {
  for (size_t bi = 0; bi < spec.alternatives.size(); ++bi) {
    for (const auto &sym : spec.alternatives[bi].symbols) {
      if (sym.is_rule) return static_cast<int>(bi);
    }
  }
  return std::nullopt;
}

// Emit the symbols of `branch` as children of `parent` starting at child
// index `idx`.  Returns the placeholder for the branch's first insertable
// rule ref.
Rule *insert_branch_at(ParentRule *parent, int idx, const AltBranch &branch) {
  Rule *first_placeholder = nullptr;
  int offset = 0;
  for (const auto &sym : branch.symbols) {
    if (sym.is_rule) {
      auto *placeholder = new UnparserRule(sym.value);
      parent->insert_child(idx + offset, placeholder);
      if (!first_placeholder) first_placeholder = placeholder;
    } else {
      auto *lit = new UnparserRule(sym.value);
      parent->insert_child(idx + offset, lit);
    }
    ++offset;
  }
  return first_placeholder;
}

std::pair<ParentRule *, int> find_real_parent_and_index(Rule *anchor) {
  ParentRule *real_parent = anchor->parent;
  auto &siblings = real_parent->children;
  auto it = std::find(siblings.begin(), siblings.end(), anchor);
  int idx = static_cast<int>(std::distance(siblings.begin(), it));
  return {real_parent, idx};
}

}  // namespace

// ---------------------------------------------------------------------------
// InsertMutation
// ---------------------------------------------------------------------------

InsertMutation::InsertMutation()
    : patterns_(get_insert_patterns()),
      edit_config_(default_edit_config()),
      k_ancestors_(4),
      l_siblings_(4),
      r_siblings_(4),
      max_inserts_(1) {}

bool InsertMutation::canApply(const MutationInput &input) const {
  return input.tree1 != nullptr && input.tree2 != nullptr && !patterns_.empty();
}

MutationResult InsertMutation::apply(const MutationInput &input) const {
  if (!input.tree1 || !input.tree2) return {nullptr, false, {}};

  ContextFilter context_filter{k_ancestors_, l_siblings_, r_siblings_};

  Rule *recipient_root = input.tree1;
  Rule *donor_root = input.tree2;

  Annotations recipient_annot(recipient_root);
  Annotations donor_annot(donor_root);

  const auto &donor_by_name = donor_annot.rules_by_name();

  // Snapshot parent-rule candidates before any tree mutation.
  std::vector<std::string> valid_parents;
  {
    const auto &recipient_by_name = recipient_annot.rules_by_name();
    for (const auto &[parent_name, pattern] : patterns_) {
      if (recipient_by_name.find(NodeKey(parent_name)) !=
          recipient_by_name.end()) {
        valid_parents.push_back(parent_name);
      }
    }
  }
  std::shuffle(valid_parents.begin(), valid_parents.end(), input.rng);

  for (const auto &parent_name : valid_parents) {
    const InsertPattern &pattern = patterns_.at(parent_name);

    if (!donor_has_required_rules(donor_annot, pattern)) continue;

    recipient_annot.reset();
    const auto &recipient_by_name = recipient_annot.rules_by_name();
    auto rbn_it = recipient_by_name.find(NodeKey(parent_name));
    if (rbn_it == recipient_by_name.end()) continue;

    std::vector<Rule *> recipient_parents = rbn_it->second;

    for (Rule *rp_rule : recipient_parents) {
      if (rp_rule->type == Rule::UnlexerRuleType) continue;
      auto *recipient_parent = static_cast<ParentRule *>(rp_rule);

      auto qmatches = match_pattern(recipient_parent, pattern);
      if (!qmatches) continue;

      for (auto &qm : *qmatches) {
        if (qm.spec->max != INT_MAX &&
            static_cast<int>(qm.reps.size()) >= qm.spec->max) {
          continue;
        }
        if (qm.spec->min == 1 && qm.spec->max == 1) continue;

        auto branch_idx = pick_insertable_branch(*qm.spec);
        if (!branch_idx) continue;
        const AltBranch &branch = qm.spec->alternatives[*branch_idx];

        auto flat = flatten_children(recipient_parent);
        int target_flat_idx;
        if (!qm.reps.empty()) {
          const auto &last = qm.reps.back();
          target_flat_idx = last.start_index + last.span;
        } else {
          if (flat.empty()) continue;
          target_flat_idx = 0;
        }

        Rule *anchor = nullptr;
        if (target_flat_idx < static_cast<int>(flat.size())) {
          anchor = flat[target_flat_idx];
        } else if (!flat.empty()) {
          anchor = flat.back();
        }
        if (!anchor) continue;

        auto [actual_parent, actual_idx] = find_real_parent_and_index(anchor);
        if (target_flat_idx >= static_cast<int>(flat.size())) ++actual_idx;

        Rule *placeholder = insert_branch_at(actual_parent, actual_idx, branch);
        if (!placeholder) continue;

        auto donor_it = donor_by_name.find(NodeKey(placeholder->name));
        if (donor_it == donor_by_name.end()) continue;
        const auto &donor_candidates = donor_it->second;
        if (donor_candidates.empty()) continue;

        int attempts = std::min(static_cast<int>(donor_candidates.size()),
                                max_inserts_);
        std::vector<Rule *> shuffled(donor_candidates.begin(),
                                     donor_candidates.end());
        std::shuffle(shuffled.begin(), shuffled.end(), input.rng);

        for (int a = 0; a < attempts; ++a) {
          Rule *donor_node = shuffled[a];
          if (!context_filter.verify(placeholder, donor_node)) continue;

          EditFragmentResult result =
              graft_fragment(placeholder, donor_node, recipient_root,
                             donor_root, input.rng, edit_config_);
          if (result.root) {
            return {result.root, /*success=*/true, "insert"};
          }
          // graft_fragment failed; tree may be inconsistent.  Bail out of
          // this parent.
          goto next_parent;
        }
      }
    }
  next_parent:;
  }

  return {nullptr, false, {}};
}

}  // namespace mlir_fuzzer
