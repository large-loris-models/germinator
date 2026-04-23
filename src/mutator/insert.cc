// insert.cc
//
// Implementation of the insert mutation.
// See insert.h for the algorithm overview.

#include "insert.h"

#include <grammarinator/runtime/Population.hpp>
#include <grammarinator/runtime/Rule.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <random>
#include <vector>

namespace mlir_fuzzer {

using grammarinator::runtime::Annotations;
using grammarinator::runtime::ParentRule;
using grammarinator::runtime::Rule;
using grammarinator::runtime::UnparserRule;
using NodeKey = Annotations::NodeKey;

// ---------------------------------------------------------------------------
// flatten_children
// ---------------------------------------------------------------------------
//
// Collect the "effective" children of a parent, unwrapping the structural
// nodes Grammarinator's C++ tree uses (quantifier/quantified/alternative
// wrappers) that are transparent in the Python tree's nodes_by_name index.

static void flatten_children_impl(Rule *node, std::vector<Rule *> &out) {
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

static std::vector<Rule *> flatten_children(ParentRule *parent) {
  std::vector<Rule *> result;
  for (auto *child : parent->children) {
    flatten_children_impl(child, result);
  }
  return result;
}

// ---------------------------------------------------------------------------
// name_matches_literal
// ---------------------------------------------------------------------------

static bool name_matches_literal(Rule *node, const std::string &literal) {
  if (node->name == literal) return true;
  if (node->type == Rule::UnlexerRuleType) {
    auto *unlexer = static_cast<grammarinator::runtime::UnlexerRule *>(node);
    if (unlexer->src == literal) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// symbol_matches
// ---------------------------------------------------------------------------

static bool symbol_matches(Rule *node, const Symbol &sym) {
  if (sym.is_rule) {
    return node->name == sym.value;
  }
  return name_matches_literal(node, sym.value);
}

// ---------------------------------------------------------------------------
// try_match_branch
// ---------------------------------------------------------------------------
//
// Given the flattened children and a start index, attempt to match every
// symbol in `branch` in order. On success, returns the number of children
// consumed (always equal to branch.symbols.size()); on failure, returns
// nullopt and `start` is untouched.

static std::optional<int> try_match_branch(const std::vector<Rule *> &flat,
                                           int start, const AltBranch &branch) {
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

// ---------------------------------------------------------------------------
// QuantifierMatch — one match of a quantifier element against the children
// ---------------------------------------------------------------------------
//
// Each repetition records which branch matched and where. For insert, we
// need to know both so we can re-emit the branch's symbols (comma + donor
// for comma-lists, or just the donor for simple quantifiers).

struct RepetitionMatch {
  int start_index;        // index into flat children where this rep begins
  int branch_index;       // which alternative matched
  int span;               // number of flat children consumed
};

struct QuantifierMatch {
  const QuantifierSpec *spec;
  std::vector<RepetitionMatch> reps;  // empty allowed if spec->min == 0
};

// ---------------------------------------------------------------------------
// match_pattern
// ---------------------------------------------------------------------------
//
// Walks the pattern and the flattened children in lockstep, greedily
// matching each element. Returns nullopt if any element can't be satisfied.
// On success, returns one QuantifierMatch per QuantifierSpec in the pattern
// (in order). Bare Symbols and bare AltBranches still advance the child
// cursor but don't produce insertable positions.
//
// For QuantifierSpec with (min,max)==(1,1): treated as "pick one branch" —
// must match exactly one branch; no insertable repetitions.

static std::optional<std::vector<QuantifierMatch>>
match_pattern(ParentRule *recipient_parent, const InsertPattern &pattern) {
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

    // Pick-one case: (1,1) with multiple alts, or (1,1) with one alt.
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

    // General quantifier: try to match each repetition greedily.
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

// ---------------------------------------------------------------------------
// donor_has_required_rules
// ---------------------------------------------------------------------------

static bool donor_has_required_rules(Annotations &donor_annot,
                                     const InsertPattern &pattern) {
  const auto &by_name = donor_annot.rules_by_name();
  for (const auto &rule_name : pattern.child_rules) {
    if (by_name.find(NodeKey(rule_name)) == by_name.end()) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// pick_insertable_branch
// ---------------------------------------------------------------------------
//
// Given a QuantifierSpec, pick a branch we can actually insert. We need a
// branch that contains at least one rule-ref Symbol (otherwise there's
// nothing for edit() to fill). Returns the first such branch index, or
// nullopt if the quantifier has no insertable branches.

static std::optional<int> pick_insertable_branch(const QuantifierSpec &spec) {
  for (size_t bi = 0; bi < spec.alternatives.size(); ++bi) {
    for (const auto &sym : spec.alternatives[bi].symbols) {
      if (sym.is_rule) return static_cast<int>(bi);
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// insert_branch_at
// ---------------------------------------------------------------------------
//
// Emit the symbols of `branch` as children of `parent` starting at child
// index `idx`. Literal symbols become unlexer-ish leaves; the first rule
// reference becomes a placeholder that edit() will fill — that placeholder
// is returned to the caller. Later rule refs in the same branch become
// their own placeholders too (rare; only matters for patterns like
// `('key' '=' rule)` where only the rule position is insertable).
//
// Returns the placeholder for the branch's first insertable rule ref.

static Rule *insert_branch_at(ParentRule *parent, int idx,
                              const AltBranch &branch) {
  Rule *first_placeholder = nullptr;
  int offset = 0;
  for (const auto &sym : branch.symbols) {
    if (sym.is_rule) {
      auto *placeholder = new UnparserRule(sym.value);
      parent->insert_child(idx + offset, placeholder);
      if (!first_placeholder) first_placeholder = placeholder;
    } else {
      // Literal: insert a leaf whose `src` matches the literal. We reuse
      // UnparserRule with the literal as its name — the mutator's name
      // comparison treats this consistently with how flatten_children sees
      // fixed-text tokens emitted by the generator.
      auto *lit = new UnparserRule(sym.value);
      parent->insert_child(idx + offset, lit);
    }
    ++offset;
  }
  return first_placeholder;
}

// ---------------------------------------------------------------------------
// find_real_parent_and_index
// ---------------------------------------------------------------------------
//
// flatten_children walks through wrapper nodes. When we want to insert at a
// flat-child position, we need to locate the *actual* parent in the tree
// and the index within that parent's children list. Use an existing node at
// the target flat-index as an anchor: its real parent and its index within
// that parent's children are the right insertion point.

static std::pair<ParentRule *, int> find_real_parent_and_index(Rule *anchor) {
  ParentRule *real_parent = anchor->parent;
  auto &siblings = real_parent->children;
  auto it = std::find(siblings.begin(), siblings.end(), anchor);
  int idx = static_cast<int>(std::distance(siblings.begin(), it));
  return {real_parent, idx};
}

// ---------------------------------------------------------------------------
// insert (top-level)
// ---------------------------------------------------------------------------

InsertResult insert(Rule *recipient_root, Rule *donor_root,
                    const InsertPatterns &patterns,
                    const ContextFilter &context_filter, int max_inserts,
                    std::mt19937 &rng, const EditConfig &edit_config) {
  Annotations recipient_annot(recipient_root);
  Annotations donor_annot(donor_root);

  const auto &donor_by_name = donor_annot.rules_by_name();

  // Snapshot parent-rule candidates before any tree mutation.
  std::vector<std::string> valid_parents;
  {
    const auto &recipient_by_name = recipient_annot.rules_by_name();
    for (const auto &[parent_name, pattern] : patterns) {
      if (recipient_by_name.find(NodeKey(parent_name)) !=
          recipient_by_name.end()) {
        valid_parents.push_back(parent_name);
      }
    }
  }
  std::shuffle(valid_parents.begin(), valid_parents.end(), rng);

  for (const auto &parent_name : valid_parents) {
    const InsertPattern &pattern = patterns.at(parent_name);

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

      // For each matched quantifier, try inserting a new repetition at a
      // valid branch. We prefer quantifiers that (a) can still accept more
      // repetitions (below max) and (b) have an insertable branch.
      for (auto &qm : *qmatches) {
        if (qm.spec->max != INT_MAX &&
            static_cast<int>(qm.reps.size()) >= qm.spec->max) {
          continue;
        }
        // Skip (1,1) pick-one quantifiers: they don't represent insertion
        // slots, just positional variance.
        if (qm.spec->min == 1 && qm.spec->max == 1) continue;

        auto branch_idx = pick_insertable_branch(*qm.spec);
        if (!branch_idx) continue;
        const AltBranch &branch = qm.spec->alternatives[*branch_idx];

        // Choose a flat-child position to insert at: append after the last
        // existing rep if any, else at the start of this quantifier's slot.
        // We need a concrete flat index plus an anchor node to locate the
        // real insertion point in the actual tree.
        auto flat = flatten_children(recipient_parent);
        int target_flat_idx;
        if (!qm.reps.empty()) {
          const auto &last = qm.reps.back();
          target_flat_idx = last.start_index + last.span;
        } else {
          // Quantifier matched zero reps; we don't have an anchor from it.
          // Fall back to the start of the quantifier's slot, which is the
          // position the matcher left the cursor at. Approximate: reuse the
          // first real flat child as anchor. If parent is empty, skip.
          if (flat.empty()) continue;
          target_flat_idx = 0;
        }

        // Find the anchor node at that flat index (or just before end).
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
        if (!placeholder) continue;  // branch had no rule refs (shouldn't happen)

        // Find a compatible donor and delegate to edit().
        auto donor_it = donor_by_name.find(NodeKey(placeholder->name));
        if (donor_it == donor_by_name.end()) continue;
        const auto &donor_candidates = donor_it->second;
        if (donor_candidates.empty()) continue;

        int attempts = std::min(
            static_cast<int>(donor_candidates.size()), max_inserts);
        std::vector<Rule *> shuffled(donor_candidates.begin(),
                                     donor_candidates.end());
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        for (int a = 0; a < attempts; ++a) {
          Rule *donor_node = shuffled[a];
          if (!context_filter.verify(placeholder, donor_node)) continue;

          EditResult result = edit(placeholder, donor_node, recipient_root,
                                   donor_root, rng, edit_config);
          if (result.root) {
            return InsertResult{result.root, result.is_fit};
          }
          // edit() failed; tree may be inconsistent. Move to next parent.
          goto next_parent;
        }
      }
    }
  next_parent:;
  }

  // Nothing worked — return a clone of the recipient unchanged.
  return InsertResult{recipient_root->clone(), false};
}

}  // namespace mlir_fuzzer