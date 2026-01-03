"""
Abstract base class for mutation families.

A mutation family defines domain-specific behavior for:
- How to identify parameters in mutations
- What fitness criteria to enforce
- How to configure context matching
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from germinator.core import (
    ContextMatcher,
    MutationSynthesizer,
    MutationApplicator,
    MutationResult,
    ParameterizedMutation,
)


@dataclass
class FamilyConfig:
    """Configuration for a mutation family."""
    
    # Context matching
    k_ancestors: int = 4
    l_siblings: int = 4
    r_siblings: int = 4
    
    # Parameterization
    parameter_blacklist: dict[str, list[str] | str] = field(default_factory=dict)
    
    # Fitness criteria
    must_substitute: dict[str, list[str] | str] = field(default_factory=dict)
    no_duplicates: dict[str, list[str] | str] = field(default_factory=dict)
    
    # Grammar
    grammar_path: Path | None = None
    
    # Valid node types for mutation (empty = all)
    valid_mutation_targets: list[str] = field(default_factory=list)


class MutationFamily(ABC):
    """
    Abstract base for mutation families.
    
    A family encapsulates domain-specific knowledge about how to
    mutate a particular class of languages/IRs.
    
    Subclasses should:
    1. Provide a default config appropriate for their domain
    2. Optionally override methods if the default algorithms don't fit
    """
    
    def __init__(self, config: FamilyConfig | None = None):
        self.config = config or self.default_config()
        
        # Initialize core components with family-specific config
        self._context_matcher = ContextMatcher(
            k_ancestors=self.config.k_ancestors,
            l_siblings=self.config.l_siblings,
            r_siblings=self.config.r_siblings,
        )
        
        self._synthesizer = MutationSynthesizer(
            parameter_blacklist=self._normalize_blacklist(self.config.parameter_blacklist),
        )
        
        self._applicator = MutationApplicator(
            must_substitute=self._normalize_blacklist(self.config.must_substitute),
            no_duplicates=self._normalize_blacklist(self.config.no_duplicates),
        )
    
    @classmethod
    @abstractmethod
    def default_config(cls) -> FamilyConfig:
        """Return the default configuration for this family."""
        ...
    
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the family name (e.g., 'ssa.mlir')."""
        ...
    
    @property
    def context_matcher(self) -> ContextMatcher:
        return self._context_matcher
    
    @property
    def synthesizer(self) -> MutationSynthesizer:
        return self._synthesizer
    
    @property
    def applicator(self) -> MutationApplicator:
        return self._applicator
    
    def extract_mutation(self, donor_node) -> ParameterizedMutation:
        """
        SYNTH: Extract a parameterized mutation from a donor node.
        
        Override this if your domain needs custom parameter identification.
        """
        return self._synthesizer.extract(
            donor_node,
            k=self.config.k_ancestors,
            l=self.config.l_siblings,
            r=self.config.r_siblings,
        )
    
    def find_locations(self, recipient_tree, donor_node) -> list:
        """
        LOCATE: Find valid insertion points in recipient.
        
        Override this if your domain needs custom location matching.
        """
        return self._context_matcher.find_matching_locations(recipient_tree, donor_node)
    
    def apply_mutation(
        self,
        mutation: ParameterizedMutation,
        recipient_node,
    ) -> MutationResult:
        """
        MATCH + INSTANTIATE: Apply mutation to recipient.
        
        Override this if your domain needs custom application logic.
        """
        return self._applicator.apply(mutation, recipient_node)
    
    def is_valid_target(self, node) -> bool:
        """Check if a node is a valid mutation target for this family."""
        if not self.config.valid_mutation_targets:
            return True
        return node.name in self.config.valid_mutation_targets
    
    def _normalize_blacklist(self, blacklist: dict) -> dict[str, list[str]]:
        """Convert string wildcards to consistent format."""
        normalized = {}
        for key, value in blacklist.items():
            if value == "*":
                normalized[key] = "*"
            elif isinstance(value, str):
                normalized[key] = [value]
            else:
                normalized[key] = list(value)
        return normalized