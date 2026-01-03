"""
Parameterized mutations and substitution logic.

Implements SYNTH (extract mutation), MATCH (bind parameters), 
and INSTANTIATE (apply substitutions).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any

from germinator.core.tree import index_by_name, find_common_names, get_root
from germinator.core.context import ContextRequirements


class FitnessViolation(Flag):
    """Reasons a mutation might fail fitness criteria."""
    NONE = 0
    MISSING_SUBSTITUTION = auto()  # Required param wasn't substituted
    DUPLICATE_DEFINITION = auto()  # SSA violation: same value defined twice
    NO_VALID_LOCATION = auto()     # Couldn't find matching context


@dataclass
class Parameter:
    """A parameter identified in a mutation fragment."""
    name: str                    # Rule name (e.g., "value_use")
    fragment_nodes: list[Any]    # Nodes in fragment that use this param
    context_nodes: list[Any]     # Nodes in donor context that define this param
    string_value: str            # The concrete string value in donor


@dataclass
class ParameterizedMutation:
    """
    A mutation extracted from a donor, with parameters identified.
    
    The fragment is a subtree to be inserted. Parameters are nodes
    within the fragment whose values should be substituted based on
    the recipient context.
    """
    fragment: Any                           # The subtree to insert (deepcopy of donor node)
    parameters: dict[str, Parameter]        # Identified parameters
    context_requirements: ContextRequirements
    donor_node: Any                         # Original donor node (for reference)


@dataclass
class MutationResult:
    """Result of applying a mutation."""
    mutant: Any                             # The mutated tree
    donor: Any | None = None                # Original donor node
    recipient: Any | None = None            # Original recipient node
    bindings: dict[str, Any] = field(default_factory=dict)  # What got substituted
    fitness_violations: FitnessViolation = FitnessViolation.NONE
    
    @property
    def is_fit(self) -> bool:
        return self.fitness_violations == FitnessViolation.NONE


class MutationSynthesizer:
    """
    Synthesizes parameterized mutations from donor trees.
    
    This implements the SYNTH step: given a donor node, extract a
    parameterized mutation by identifying which parts of the fragment
    correspond to values defined in the surrounding context.
    """
    
    def __init__(self, parameter_blacklist: dict[str, list[str]] | None = None):
        """
        Args:
            parameter_blacklist: Rules to exclude from parameterization.
                Format: {"child_rule": ["parent_rule", ...]} or {"child_rule": "*"}
        """
        self.parameter_blacklist = parameter_blacklist or {}
    
    def extract(
        self,
        donor_node,
        k: int = 4,
        l: int = 4,
        r: int = 4,
    ) -> ParameterizedMutation:
        """
        Extract a parameterized mutation from a donor node.
        
        Args:
            donor_node: The node to extract as a mutation
            k, l, r: Context matching parameters
        
        Returns:
            ParameterizedMutation with identified parameters
        """
        fragment = deepcopy(donor_node)
        donor_root = get_root(donor_node)
        
        # Index fragment nodes (excluding nothing - we want all of them)
        fragment_index = self._index_with_blacklist(fragment, exclude_subtree=None)
        
        # Index context nodes (excluding the donor subtree itself)
        context_index = self._index_with_blacklist(donor_root, exclude_subtree=donor_node)
        
        # Find parameters: nodes with same name AND same string value
        # appearing in both fragment and context
        common_names = find_common_names(fragment_index, context_index)
        parameters = {}
        
        for name in common_names:
            fragment_nodes = fragment_index[name]
            context_nodes = context_index[name]
            
            # Group by string value
            fragment_by_value = self._group_by_string(fragment_nodes)
            context_by_value = self._group_by_string(context_nodes)
            
            # Parameters are values that appear in both
            common_values = set(fragment_by_value.keys()) & set(context_by_value.keys())
            
            for value in common_values:
                param_key = f"{name}:{value}"
                parameters[param_key] = Parameter(
                    name=name,
                    fragment_nodes=fragment_by_value[value],
                    context_nodes=context_by_value[value],
                    string_value=value,
                )
        
        context_requirements = ContextRequirements.from_node(donor_node, k, l, r)
        
        return ParameterizedMutation(
            fragment=fragment,
            parameters=parameters,
            context_requirements=context_requirements,
            donor_node=donor_node,
        )
    
    def _index_with_blacklist(self, node, exclude_subtree) -> dict[str, list]:
        """Index nodes, respecting the parameter blacklist."""
        index = {}
        
        from germinator.core.tree import walk_tree
        
        for n in walk_tree(node):
            # Skip excluded subtree
            if exclude_subtree is not None and n is exclude_subtree:
                continue
            
            # Check blacklist
            if self._is_blacklisted(n):
                continue
            
            if n.name not in index:
                index[n.name] = []
            index[n.name].append(n)
        
        return index
    
    def _is_blacklisted(self, node) -> bool:
        """Check if a node should be excluded from parameterization."""
        if node.name in self.parameter_blacklist:
            allowed_parents = self.parameter_blacklist[node.name]
            if allowed_parents == "*":
                return True
            if node.parent and node.parent.name in allowed_parents:
                return True
        
        # Also check wildcard
        if "*" in self.parameter_blacklist:
            if node.name in self.parameter_blacklist["*"]:
                return True
        
        return False
    
    def _group_by_string(self, nodes) -> dict[str, list]:
        """Group nodes by their string representation."""
        groups = {}
        for node in nodes:
            s = str(node)
            if s not in groups:
                groups[s] = []
            groups[s].append(node)
        return groups


class MutationApplicator:
    """
    Applies parameterized mutations to recipient trees.
    
    This implements MATCH (bind parameters from recipient context)
    and INSTANTIATE (substitute and insert).
    """
    
    def __init__(
        self,
        must_substitute: dict[str, list[str]] | None = None,
        no_duplicates: dict[str, list[str]] | None = None,
    ):
        """
        Args:
            must_substitute: Rules that must have their params substituted.
                Format: {"child_rule": ["parent_rule", ...]} or {"child_rule": "*"}
            no_duplicates: Rules that cannot appear with duplicate values.
                Format: same as must_substitute
        """
        self.must_substitute = must_substitute or {}
        self.no_duplicates = no_duplicates or {}
    
    def apply(
        self,
        mutation: ParameterizedMutation,
        recipient_node,
    ) -> MutationResult:
        """
        Apply a parameterized mutation to a recipient location.
        
        Args:
            mutation: The parameterized mutation to apply
            recipient_node: Where to insert in the recipient tree
        
        Returns:
            MutationResult with the mutated tree and fitness info
        """
        fragment = deepcopy(mutation.fragment)
        recipient = deepcopy(recipient_node)
        
        # MATCH: Extract parameter bindings from recipient context
        bindings = self._extract_bindings(mutation, recipient)
        
        # Track which params must be substituted
        required_subs = self._find_required_substitutions(mutation)
        
        # Substitute parameters in fragment
        for param_key, param in mutation.parameters.items():
            if param_key not in bindings:
                continue
            
            bound_value = bindings[param_key]
            
            # Replace all occurrences in fragment
            for fragment_node in param.fragment_nodes:
                replacement = deepcopy(bound_value)
                fragment_node.replace(replacement)
                
                # Track if required sub was made
                if fragment_node in required_subs:
                    required_subs.remove(fragment_node)
        
        # Insert fragment at recipient location
        mutant_node = recipient.replace(fragment)
        mutant_root = get_root(mutant_node)
        
        # Check fitness
        violations = FitnessViolation.NONE
        
        if required_subs:
            violations |= FitnessViolation.MISSING_SUBSTITUTION
        
        if self._has_duplicates(mutant_root):
            violations |= FitnessViolation.DUPLICATE_DEFINITION
        
        return MutationResult(
            mutant=mutant_root,
            donor=mutation.donor_node,
            recipient=recipient_node,
            bindings=bindings,
            fitness_violations=violations,
        )
    
    def _extract_bindings(
        self,
        mutation: ParameterizedMutation,
        recipient_node,
    ) -> dict[str, Any]:
        """
        MATCH step: Walk recipient context to find values for parameters.
        """
        bindings = {}
        recipient_root = get_root(recipient_node)
        
        # Get ancestors of both donor and recipient for matching
        donor_ancestors = self._get_ancestor_chain(mutation.donor_node)
        recipient_ancestors = self._get_ancestor_chain(recipient_node)
        
        # Match along ancestor chains
        for d_ancestor, r_ancestor in zip(donor_ancestors, recipient_ancestors):
            if d_ancestor.name != r_ancestor.name:
                break
            
            # Get siblings at this level
            d_idx = d_ancestor.parent.children.index(d_ancestor) if d_ancestor.parent else 0
            r_idx = r_ancestor.parent.children.index(r_ancestor) if r_ancestor.parent else 0
            
            if d_ancestor.parent and r_ancestor.parent:
                d_siblings = d_ancestor.parent.children
                r_siblings = r_ancestor.parent.children
                
                # Match left siblings
                self._match_siblings(
                    d_siblings[:d_idx], r_siblings[:r_idx],
                    mutation.parameters, bindings
                )
                
                # Match right siblings
                self._match_siblings(
                    d_siblings[d_idx+1:], r_siblings[r_idx+1:],
                    mutation.parameters, bindings
                )
        
        return bindings
    
    def _get_ancestor_chain(self, node) -> list:
        """Get list of ancestors from node up to root."""
        chain = []
        current = node
        while current.parent:
            chain.append(current)
            current = current.parent
        chain.append(current)  # Include root
        return chain
    
    def _match_siblings(
        self,
        donor_siblings: list,
        recipient_siblings: list,
        parameters: dict[str, Parameter],
        bindings: dict[str, Any],
    ):
        """Recursively match sibling nodes to extract parameter bindings."""
        d_idx = 0
        
        for r_node in recipient_siblings:
            while d_idx < len(donor_siblings):
                d_node = donor_siblings[d_idx]
                d_idx += 1
                
                if d_node.name == r_node.name:
                    # Check if this is a parameter
                    param_key = f"{d_node.name}:{str(d_node)}"
                    if param_key in parameters:
                        if param_key not in bindings:
                            bindings[param_key] = r_node
                    
                    # Recurse into children
                    if d_node.children and r_node.children:
                        self._match_siblings(
                            d_node.children, r_node.children,
                            parameters, bindings
                        )
                    break
    
    def _find_required_substitutions(self, mutation: ParameterizedMutation) -> set:
        """Find fragment nodes that must be substituted per fitness criteria."""
        required = set()
        
        for param in mutation.parameters.values():
            for node in param.fragment_nodes:
                if self._must_substitute(node):
                    required.add(node)
        
        return required
    
    def _must_substitute(self, node) -> bool:
        """Check if a node must have its parameter substituted."""
        if node.name in self.must_substitute:
            allowed = self.must_substitute[node.name]
            if allowed == "*":
                return True
            if node.parent and node.parent.name in allowed:
                return True
        return False
    
    def _has_duplicates(self, root) -> bool:
        """Check if tree violates no-duplicate rules."""
        from germinator.core.tree import walk_tree
        
        seen = set()
        
        for node in walk_tree(root):
            if node.name in self.no_duplicates:
                allowed = self.no_duplicates[node.name]
                should_check = (
                    allowed == "*" or
                    (node.parent and node.parent.name in allowed)
                )
                
                if should_check:
                    node_str = str(node)
                    if node_str in seen:
                        return True
                    seen.add(node_str)
        
        return False