// edit_mutation.cc
//
// Implementation of the EditMutation (plus the shared graft_fragment()
// routine used by other mutations).
//
// Algorithm for a full edit:
//   1. Select a (recipient_node, donor_node) pair of the same rule name,
//      where grafting donor_node under recipient_node won't exceed
//      max_depth and passes the context filter.
//   2. Clone the full donor tree so context indexing sees the complete tree.
//   3. Locate the clone of donor_node within the cloned tree.
//   4. Detect "parameters" in the donor fragment — nodes appearing both
//      inside the donor subtree and outside it (in the donor's context)
//      with identical serialized text.  Blacklisted names are skipped.
//   5. Walk common ancestors of recipient_node and donor_node to collect
//      concrete parameter values from the recipient's context.
//   6. Substitute parameters in the cloned donor fragment.
//   7. Check fitness: were required parameters (should_substitute)
//      fulfilled?
//   8. Graft the adapted fragment in place of recipient_node.
//
// If the donor has no children, falls back to a simple graft.

#include "edit_mutation.h"

#include "context_filter.h"

#include <grammarinator/runtime/Population.hpp>
#include <grammarinator/runtime/Rule.hpp>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlir_fuzzer {

using grammarinator::runtime::Annotations;
using grammarinator::runtime::ParentRule;
using grammarinator::runtime::Rule;
using grammarinator::runtime::UnlexerRule;
using NodeKey = Annotations::NodeKey;

// ---------------------------------------------------------------------------
// Shared config
// ---------------------------------------------------------------------------

EditConfig default_edit_config() {
  EditConfig cfg;
  cfg.parameter_blacklist["string_literal"] = {"generic_operation"};
  cfg.should_substitute["ssa_id"] = {"ssa_use"};
  cfg.should_substitute["non_function_type"] = {"*"};
  return cfg;
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

namespace {

using NodeIndex = std::unordered_map<std::string, std::vector<Rule *>>;
using ParameterMap = std::unordered_map<Rule *, std::vector<Rule *>>;
using ParameterValues = std::unordered_map<Rule *, std::vector<Rule *>>;

// ---------------------------------------------------------------------------
// index_nodes
// ---------------------------------------------------------------------------
//
// Recursively collect all named Rule* nodes (UnlexerRule and UnparserRule)
// in the subtree rooted at `current`, grouped by name.  The subtree rooted
// at `exclude` is skipped entirely.  Nodes matching the parameter blacklist
// are skipped.

void index_nodes(Rule *current, NodeIndex &index, const Rule *exclude,
                 const ParameterBlacklist &blacklist) {
  if (!current) return;
  if (current == exclude) return;

  auto bl_it = blacklist.find(current->name);
  if (bl_it != blacklist.end()) {
    const auto &parents = bl_it->second;
    if (parents.empty()) return;  // wildcard: always blacklisted
    if (current->parent) {
      for (const auto &p : parents) {
        if (p == "*" || p == current->parent->name) return;
      }
    }
  }

  if (current->type == Rule::UnlexerRuleType ||
      current->type == Rule::UnparserRuleType) {
    index[current->name].push_back(current);
  }

  if (current->type != Rule::UnlexerRuleType) {
    auto *parent = static_cast<ParentRule *>(current);
    for (auto *child : parent->children) {
      index_nodes(child, index, exclude, blacklist);
    }
  }
}

// ---------------------------------------------------------------------------
// collect_parameters
// ---------------------------------------------------------------------------
//
// Find "parameters": nodes appearing in both the fragment (children of
// donor_node) and the context (donor tree outside donor_node) with identical
// serialized text.  donor_root and donor_node must be in the same cloned
// tree so pointer identity is stable.

ParameterMap collect_parameters(Rule *donor_root, Rule *donor_node,
                                const ParameterBlacklist &blacklist) {
  NodeIndex fragment_index, context_index;

  if (donor_node->type != Rule::UnlexerRuleType) {
    auto *p = static_cast<ParentRule *>(donor_node);
    for (auto *child : p->children) {
      index_nodes(child, fragment_index, /*exclude=*/nullptr, blacklist);
    }
  }

  index_nodes(donor_root, context_index, /*exclude=*/donor_node, blacklist);

  ParameterMap parameters;
  for (const auto &[name, fragment_nodes] : fragment_index) {
    auto it = context_index.find(name);
    if (it == context_index.end()) continue;
    const auto &context_nodes = it->second;

    for (auto *frag_node : fragment_nodes) {
      for (auto *ctx_node : context_nodes) {
        if (frag_node->equalTokens(*ctx_node)) {
          parameters[ctx_node].push_back(frag_node);
        }
      }
    }
  }
  return parameters;
}

// ---------------------------------------------------------------------------
// match_nodes (forward declaration for mutual recursion)
// ---------------------------------------------------------------------------

void match_nodes(const std::vector<Rule *> &abstract_nodes,
                 const std::vector<Rule *> &concrete_nodes,
                 const ParameterMap &parameters,
                 ParameterValues &parameter_values);

void recursively_match_nodes(Rule *abstract_node, Rule *concrete_node,
                             const ParameterMap &parameters,
                             ParameterValues &parameter_values) {
  if (abstract_node->type == Rule::UnlexerRuleType ||
      concrete_node->type == Rule::UnlexerRuleType) {
    return;
  }
  auto *a_parent = static_cast<ParentRule *>(abstract_node);
  auto *c_parent = static_cast<ParentRule *>(concrete_node);
  match_nodes(a_parent->children, c_parent->children, parameters,
              parameter_values);
}

// Greedy left-to-right name-based matching of abstract (donor) nodes against
// concrete (recipient) nodes.  On name match, if abstract is a parameter,
// record concrete as a candidate value; otherwise recurse into children.
void match_nodes(const std::vector<Rule *> &abstract_nodes,
                 const std::vector<Rule *> &concrete_nodes,
                 const ParameterMap &parameters,
                 ParameterValues &parameter_values) {
  size_t c_idx = 0;
  for (size_t a_idx = 0; a_idx < abstract_nodes.size(); ++a_idx) {
    Rule *a_node = abstract_nodes[a_idx];
    size_t old_idx = c_idx;
    bool found = false;

    while (c_idx < concrete_nodes.size()) {
      Rule *c_node = concrete_nodes[c_idx];
      ++c_idx;
      if (a_node->name == c_node->name) {
        found = true;
        if (parameters.count(a_node)) {
          parameter_values[a_node].push_back(c_node);
        } else {
          recursively_match_nodes(a_node, c_node, parameters, parameter_values);
        }
        break;
      }
    }
    if (!found) c_idx = old_idx;
  }
}

// Walk up from recipient_node and donor_node simultaneously, collecting
// sibling lists at each level.  Stops when parent names diverge.
ParameterValues collect_parameter_values(Rule *recipient_node, Rule *donor_node,
                                         const ParameterMap &parameters) {
  ParameterValues parameter_values;

  Rule *r = recipient_node;
  Rule *d = donor_node;

  while (r && d && r->parent && d->parent &&
         r->parent->name == d->parent->name) {
    {
      std::vector<Rule *> left_abstract, left_concrete;
      for (Rule *s = d->left_sibling(); s; s = s->left_sibling())
        left_abstract.push_back(s);
      for (Rule *s = r->left_sibling(); s; s = s->left_sibling())
        left_concrete.push_back(s);
      std::reverse(left_abstract.begin(), left_abstract.end());
      std::reverse(left_concrete.begin(), left_concrete.end());
      if (!left_abstract.empty() && !left_concrete.empty())
        match_nodes(left_abstract, left_concrete, parameters, parameter_values);
    }
    {
      std::vector<Rule *> right_abstract, right_concrete;
      for (Rule *s = d->right_sibling(); s; s = s->right_sibling())
        right_abstract.push_back(s);
      for (Rule *s = r->right_sibling(); s; s = s->right_sibling())
        right_concrete.push_back(s);
      if (!right_abstract.empty() && !right_concrete.empty())
        match_nodes(right_abstract, right_concrete, parameters,
                    parameter_values);
    }
    r = r->parent;
    d = d->parent;
  }
  return parameter_values;
}

// Equivalent to Python's recombine(): deep-copy donor, graft into recipient.
// Used when donor has no children.
EditFragmentResult simple_graft(Rule *recipient_node, Rule *donor_node) {
  Rule *cloned_donor = donor_node->clone();
  recipient_node->replace(cloned_donor);
  delete recipient_node;

  Rule *root = cloned_donor;
  while (root->parent && root->parent->name != "<ROOT>") {
    root = root->parent;
  }
  return {root, /*is_fit=*/true};
}

// Fitness check: verify that parameter nodes matching should_substitute
// criteria were actually substituted.
bool check_should_substitute(const ParameterMap &parameters,
                             const std::unordered_set<Rule *> &replaced_nodes,
                             const ShouldSubstituteSet &should_substitute) {
  if (should_substitute.empty()) return true;

  for (const auto &[ctx_node, frag_nodes] : parameters) {
    for (auto *frag_node : frag_nodes) {
      auto it = should_substitute.find(frag_node->name);
      if (it == should_substitute.end()) continue;

      const auto &parents = it->second;
      bool requires_sub = false;
      if (parents.empty() || (parents.size() == 1 && parents[0] == "*")) {
        requires_sub = true;
      } else if (frag_node->parent) {
        for (const auto &p : parents) {
          if (p == frag_node->parent->name) {
            requires_sub = true;
            break;
          }
        }
      }

      if (requires_sub &&
          replaced_nodes.find(frag_node) == replaced_nodes.end()) {
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// select_edit_pair
// ---------------------------------------------------------------------------
//
// Pick matching (recipient_node, donor_node) names that can be grafted
// without blowing past max_depth and that pass the context filter.

std::pair<Rule *, Rule *> select_edit_pair(Annotations &r_annot,
                                           Annotations &d_annot, int max_depth,
                                           const ContextFilter &context_filter,
                                           std::mt19937 &rng) {
  const auto &r_by_name = r_annot.rules_by_name();
  const auto &d_by_name = d_annot.rules_by_name();
  const auto &r_info = r_annot.node_info();
  const auto &d_info = d_annot.node_info();

  std::vector<NodeKey> common_keys;
  for (const auto &[key, nodes] : r_by_name) {
    if (d_by_name.count(key)) common_keys.push_back(key);
  }
  std::shuffle(common_keys.begin(), common_keys.end(), rng);

  for (const auto &key : common_keys) {
    std::vector<Rule *> r_nodes = r_by_name.at(key);
    std::vector<Rule *> d_nodes = d_by_name.at(key);
    std::shuffle(r_nodes.begin(), r_nodes.end(), rng);
    std::shuffle(d_nodes.begin(), d_nodes.end(), rng);

    for (Rule *r_node : r_nodes) {
      auto r_it = r_info.find(r_node);
      if (r_it == r_info.end()) continue;
      int r_level = r_it->second.level;

      for (Rule *d_node : d_nodes) {
        auto d_it = d_info.find(d_node);
        if (d_it == d_info.end()) continue;
        int d_depth = d_it->second.depth;

        if (r_level + d_depth > max_depth) continue;
        if (!context_filter.verify(r_node, d_node)) continue;

        return {r_node, d_node};
      }
    }
  }
  return {nullptr, nullptr};
}

}  // namespace

// ---------------------------------------------------------------------------
// graft_fragment — shared low-level graft
// ---------------------------------------------------------------------------

EditFragmentResult graft_fragment(Rule *recipient_node, Rule *donor_node,
                                  Rule *recipient_root, Rule *donor_root,
                                  std::mt19937 &rng, const EditConfig &config) {
  (void)recipient_root;

  if (recipient_node->name != donor_node->name) {
    return {nullptr, false};
  }

  // If donor has no children, fall back to simple graft (no parameter sub).
  if (donor_node->type == Rule::UnlexerRuleType) {
    return simple_graft(recipient_node, donor_node);
  }
  {
    auto *dp = static_cast<ParentRule *>(donor_node);
    if (dp->children.empty()) {
      return simple_graft(recipient_node, donor_node);
    }
  }

  // Step 1: Clone the entire donor tree.
  Rule *cloned_donor_root = donor_root->clone();

  // Step 2: Locate the clone of donor_node via parallel walk.
  Rule *cloned_donor_node = nullptr;
  {
    std::vector<std::pair<Rule *, Rule *>> stack;
    stack.push_back({donor_root, cloned_donor_root});
    while (!stack.empty()) {
      auto [orig, clone] = stack.back();
      stack.pop_back();
      if (orig == donor_node) {
        cloned_donor_node = clone;
        break;
      }
      if (orig->type != Rule::UnlexerRuleType) {
        auto *op = static_cast<ParentRule *>(orig);
        auto *cp = static_cast<ParentRule *>(clone);
        for (size_t i = 0; i < op->children.size(); ++i) {
          stack.push_back({op->children[i], cp->children[i]});
        }
      }
    }
  }

  if (!cloned_donor_node) {
    delete cloned_donor_root;
    return {nullptr, false};
  }

  // Step 3: Detect parameters (with blacklist filtering).
  ParameterMap parameters = collect_parameters(
      cloned_donor_root, cloned_donor_node, config.parameter_blacklist);

  // Step 4: Collect parameter values from the recipient's context.
  ParameterValues parameter_values =
      collect_parameter_values(recipient_node, cloned_donor_node, parameters);

  // Step 5: Substitute parameters in the cloned fragment.
  std::vector<Rule *> detached_nodes;
  std::unordered_set<Rule *> already_replaced;

  for (const auto &[ctx_node, frag_orig_nodes] : parameters) {
    auto val_it = parameter_values.find(ctx_node);
    if (val_it == parameter_values.end() || val_it->second.empty()) continue;

    const auto &candidates = val_it->second;
    Rule *chosen = candidates[std::uniform_int_distribution<size_t>(
        0, candidates.size() - 1)(rng)];

    for (auto *frag_node : frag_orig_nodes) {
      if (already_replaced.count(frag_node)) continue;
      already_replaced.insert(frag_node);

      Rule *replacement = chosen->clone();
      frag_node->replace(replacement);
      detached_nodes.push_back(frag_node);
    }
  }

  for (auto *n : detached_nodes) delete n;

  // Step 6: Fitness check — did required parameters get substituted?
  bool is_fit = check_should_substitute(parameters, already_replaced,
                                        config.should_substitute);

  // Step 7: Graft — detach cloned_donor_node and replace recipient_node.
  cloned_donor_node->remove();
  recipient_node->replace(cloned_donor_node);
  delete recipient_node;

  // Step 8: Delete the remainder of the cloned donor tree.
  delete cloned_donor_root;

  // Step 9: Find and return the root of the mutated recipient tree.
  Rule *root = cloned_donor_node;
  while (root->parent && root->parent->name != "<ROOT>") {
    root = root->parent;
  }
  return {root, is_fit};
}

// ---------------------------------------------------------------------------
// EditMutation
// ---------------------------------------------------------------------------

EditMutation::EditMutation() : edit_config_(default_edit_config()) {
  const char *env_depth = std::getenv("GRAMMARINATOR_MAX_DEPTH");
  max_depth_ = env_depth ? std::atoi(env_depth) : 30;
  k_ancestors_ = 4;
  l_siblings_ = 4;
  r_siblings_ = 4;
}

bool EditMutation::canApply(const MutationInput &input) const {
  return input.tree1 != nullptr && input.tree2 != nullptr;
}

MutationResult EditMutation::apply(const MutationInput &input) const {
  if (!input.tree1 || !input.tree2) return {nullptr, false, {}};

  ContextFilter context_filter{k_ancestors_, l_siblings_, r_siblings_};

  Annotations r_annot(input.tree1);
  Annotations d_annot(input.tree2);

  auto [r_node, d_node] =
      select_edit_pair(r_annot, d_annot, max_depth_, context_filter, input.rng);

  if (!r_node || !d_node) return {nullptr, false, {}};

  EditFragmentResult result = graft_fragment(
      r_node, d_node, input.tree1, input.tree2, input.rng, edit_config_);

  if (!result.root) return {nullptr, false, {}};

  return {result.root, /*success=*/true, "edit"};
}

}  // namespace mlir_fuzzer
