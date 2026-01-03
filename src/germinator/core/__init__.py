"""
Core primitives for tree manipulation, mutations, and context matching.
"""

from germinator.core.tree import (
    get_root,
    left_sibling,
    right_sibling,
    ancestors,
    walk_tree,
    index_by_name,
    find_common_names,
    NodePath,
)
from germinator.core.context import (
    ContextRequirements,
    ContextMatcher,
)
from germinator.core.mutation import (
    FitnessViolation,
    Parameter,
    ParameterizedMutation,
    MutationResult,
    MutationSynthesizer,
    MutationApplicator,
)

__all__ = [
    # tree
    "get_root",
    "left_sibling",
    "right_sibling",
    "ancestors",
    "walk_tree",
    "index_by_name",
    "find_common_names",
    "NodePath",
    # context
    "ContextRequirements",
    "ContextMatcher",
    # mutation
    "FitnessViolation",
    "Parameter",
    "ParameterizedMutation",
    "MutationResult",
    "MutationSynthesizer",
    "MutationApplicator",
]