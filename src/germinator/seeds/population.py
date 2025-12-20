"""
Population management for evolutionary fuzzing.

Extends SeedStore with selection strategies for mutation and recombination.
Uses Grammarinator's tree infrastructure for efficient node indexing.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grammarinator.tool.default_population import DefaultPopulation, DefaultTree

from germinator.core import ContextMatcher

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of selecting nodes for mutation."""
    donor_node: Any
    recipient_node: Any
    donor_tree: DefaultTree
    recipient_tree: DefaultTree


class Population:
    """
    Population of trees for evolutionary fuzzing.
    
    Wraps Grammarinator's DefaultPopulation with:
    - Context-aware selection (k-ancestor, l/r-sibling matching)
    - Support for family-specific selection criteria
    """
    
    def __init__(
        self,
        directory: Path | str,
        context_matcher: ContextMatcher | None = None,
        valid_targets: list[str] | None = None,
        min_depths: dict[str, int] | None = None,
    ):
        """
        Args:
            directory: Path to population directory (with .grt tree files)
            context_matcher: Matcher for context-aware selection
            valid_targets: Node types valid for mutation (None = all)
            min_depths: Minimum depths per rule (from grammar processing)
        """
        self.directory = Path(directory)
        self.context_matcher = context_matcher or ContextMatcher()
        self.valid_targets = set(valid_targets) if valid_targets else None
        
        # Use Grammarinator's population for tree management
        self._population = DefaultPopulation(
            directory=str(self.directory),
            min_depths=min_depths or {},
        )
        
        self._rng = random.Random()
        
        logger.info(f"Population initialized with {len(self._population._files)} trees from {self.directory}")
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = random.Random(seed)
    
    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        context_matcher: ContextMatcher | None = None,
        valid_targets: list[str] | None = None,
    ) -> "Population":
        """Create a population from a directory of tree files."""
        return cls(
            directory=directory,
            context_matcher=context_matcher,
            valid_targets=valid_targets,
        )
    
    def can_mutate(self) -> bool:
        """Check if population has enough trees for mutation."""
        return len(self._population._files) >= 1
    
    def can_recombine(self) -> bool:
        """Check if population has enough trees for recombination."""
        return len(self._population._files) >= 2
    
    def add_tree(self, tree: Any, path: Path | str | None = None):
        """Add a tree to the population."""
        self._population.add_individual(tree, path=str(path) if path else None)
    
    def select_for_mutation(self, max_depth: int = 100) -> Any | None:
        """
        Select a node for mutation (re-generation).

        Returns a node that can be mutated by regenerating its subtree.
        """
        if not self.can_mutate():
            return None
        
        return self._population.select_to_mutate(max_depth)
    
    def select_for_recombine(self, max_depth: int = 100) -> SelectionResult | None:
        """
        Select donor and recipient nodes for recombination.

        Uses context matching to find compatible locations.

        Returns:
            SelectionResult with matched donor/recipient nodes, or None
        """
        if not self.can_recombine():
            return None
        
        # Get random tree files
        tree_files = list(self._population._files)
        self._rng.shuffle(tree_files)
        
        # Try pairs until we find a match
        for i in range(0, min(len(tree_files) - 1, 20), 2):  # Limit attempts
            try:
                recipient_tree = DefaultTree.load(tree_files[i])
                donor_tree = DefaultTree.load(tree_files[i + 1])
                
                result = self._find_compatible_nodes(
                    recipient_tree, donor_tree, max_depth
                )
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Failed to load tree pair: {e}")
                continue
        
        return None
    
    def select_for_edit(self, max_depth: int = 100) -> SelectionResult | None:
        """
        Select donor and recipient for SynthFuzz-style edit.

        Same as recombine but with context matching enforced.
        """
        return self.select_for_recombine(max_depth)
    
    def select_for_insert(self, max_depth: int = 100) -> tuple[DefaultTree, DefaultTree] | None:
        """
        Select trees for insertion mutation.

        Returns two trees; insertion logic handled by generator.
        """
        if not self.can_recombine():
            return None
        
        tree_files = list(self._population._files)
        if len(tree_files) < 2:
            return None
            
        selected = self._rng.sample(tree_files, 2)
        
        try:
            recipient_tree = DefaultTree.load(selected[0])
            donor_tree = DefaultTree.load(selected[1])
            return recipient_tree, donor_tree
        except Exception as e:
            logger.debug(f"Failed to load trees for insert: {e}")
            return None
    
    def _find_compatible_nodes(
        self,
        recipient_tree: DefaultTree,
        donor_tree: DefaultTree,
        max_depth: int,
    ) -> SelectionResult | None:
        """Find compatible donor/recipient nodes with context matching."""
        
        # Find common rule names
        common_names = set(recipient_tree.nodes_by_name.keys()) & \
                       set(donor_tree.nodes_by_name.keys())
        
        # Filter by valid targets if specified
        if self.valid_targets:
            common_names &= self.valid_targets
        
        if not common_names:
            return None
        
        # Build candidate list from recipient
        recipient_candidates = []
        for name in common_names:
            for node in recipient_tree.nodes_by_name[name]:
                # Check depth constraint
                level = recipient_tree.node_levels.get(node, 0)
                if level <= max_depth:
                    recipient_candidates.append(node)
        
        if not recipient_candidates:
            return None
        
        # Shuffle and try to find matches
        self._rng.shuffle(recipient_candidates)
        
        for recipient_node in recipient_candidates[:50]:  # Limit attempts
            donor_candidates = list(donor_tree.nodes_by_name.get(recipient_node.name, []))
            self._rng.shuffle(donor_candidates)
            
            for donor_node in donor_candidates[:10]:  # Limit attempts
                # Check context matching
                if not self.context_matcher.matches(recipient_node, donor_node):
                    continue
                
                # Check depth constraint
                donor_depth = donor_tree.node_depths.get(donor_node, 0)
                recipient_level = recipient_tree.node_levels.get(recipient_node, 0)
                
                if recipient_level + donor_depth <= max_depth:
                    return SelectionResult(
                        donor_node=donor_node,
                        recipient_node=recipient_node,
                        donor_tree=donor_tree,
                        recipient_tree=recipient_tree,
                    )
        
        return None