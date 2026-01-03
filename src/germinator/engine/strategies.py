"""
Generation strategies.

Each strategy defines a different way to create test cases:
- Generate: Pure grammar-based generation
- Mutate: Re-generate a subtree
- Recombine: Splice trees together
- Edit: SynthFuzz-style parameterized mutation
- Insert: Add new nodes at valid locations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from germinator.core import (
    MutationResult,
    MutationSynthesizer,
    MutationApplicator,
    FitnessViolation,
)
from germinator.seeds.population import Population, SelectionResult


class StrategyType(Enum):
    """Available generation strategies."""
    GENERATE = auto()
    MUTATE = auto()
    RECOMBINE = auto()
    EDIT = auto()
    INSERT = auto()


@dataclass
class GenerationResult:
    """Result from any generation strategy."""
    tree: Any
    strategy: StrategyType
    success: bool
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Strategy(ABC):
    """Base class for generation strategies."""
    
    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        ...
    
    @abstractmethod
    def can_execute(self) -> bool:
        """Check if this strategy can be executed given current state."""
        ...
    
    @abstractmethod
    def execute(self) -> GenerationResult:
        """Execute the strategy and return a result."""
        ...


class GenerateStrategy(Strategy):
    """
    Pure grammar-based generation.

    Creates new trees from scratch using the grammar.
    No seeds required.
    """

    def __init__(self, generator_factory, rule: str | None = None, max_depth: int = 100):
        self.generator_factory = generator_factory
        self.rule = rule
        self.max_depth = max_depth

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.GENERATE

    def can_execute(self) -> bool:
        return self.generator_factory is not None

    def execute(self) -> GenerationResult:
        try:
            generator = self.generator_factory(max_depth=self.max_depth)

            rule_name = self.rule or generator._default_rule.__name__
            rule_fn = getattr(generator, rule_name)

            tree = rule_fn()

            return GenerationResult(
                tree=tree,
                strategy=self.strategy_type,
                success=True,
                metadata={"rule": rule_name},
            )
        except Exception as e:
            return GenerationResult(
                tree=None,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": str(e)},
            )
            
class MutateStrategy(Strategy):
    """
    Mutation via subtree regeneration.

    Selects a node and regenerates its subtree using the grammar.
    """

    def __init__(
        self,
        population: Population,
        generator_factory,
        max_depth: int = 100,
    ):
        self.population = population
        self.generator_factory = generator_factory
        self.max_depth = max_depth

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.MUTATE

    def can_execute(self) -> bool:
        return self.population.can_mutate()

    def execute(self) -> GenerationResult:
        from copy import deepcopy
        from germinator.core.tree import get_root

        node = self.population.select_for_mutation(self.max_depth)
        if node is None:
            return GenerationResult(
                tree=None,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": "no node selected"},
            )

        # Calculate depth budget
        level = 0
        current = node
        while current.parent:
            current = current.parent
            level += 1

        # Generate replacement - but only if the rule exists in the generator
        generator = self.generator_factory(max_depth=self.max_depth - level)
        
        # Check if this rule exists as a method in the generator
        if not hasattr(generator, node.name):
            return GenerationResult(
                tree=None,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": f"rule '{node.name}' not in generator (likely a lexer rule)"},
            )

        original = deepcopy(node)
        
        try:
            rule_fn = getattr(generator, node.name)
            replacement = rule_fn()

            # Replace and get root
            node.replace(replacement)
            root = get_root(node)

            return GenerationResult(
                tree=root,
                strategy=self.strategy_type,
                success=True,
                metadata={
                    "mutated_node": node.name,
                    "original": original,
                },
            )
        except Exception as e:
            return GenerationResult(
                tree=None,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": str(e)},
            )
            
class RecombineStrategy(Strategy):
    """
    Simple tree recombination.
    
    Splices a subtree from donor into recipient at a compatible location.
    No parameterization - just direct replacement.
    """
    
    def __init__(self, population: Population, max_depth: int = 100):
        self.population = population
        self.max_depth = max_depth
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.RECOMBINE
    
    def can_execute(self) -> bool:
        return self.population.can_recombine()
    
    def execute(self) -> GenerationResult:
        from copy import deepcopy
        from germinator.core.tree import get_root
        
        selection = self.population.select_for_recombine(self.max_depth)
        if selection is None:
            return GenerationResult(
                tree=None,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": "no compatible nodes found"},
            )
        
        donor = deepcopy(selection.donor_node)
        recipient = deepcopy(selection.recipient_node)
        
        recipient.replace(donor)
        root = get_root(recipient)
        
        return GenerationResult(
            tree=root,
            strategy=self.strategy_type,
            success=True,
            metadata={
                "donor_node": selection.donor_node.name,
                "recipient_node": selection.recipient_node.name,
            },
        )


class EditStrategy(Strategy):
    """
    SynthFuzz-style parameterized edit.
    
    Extracts a parameterized mutation from donor, finds matching
    context in recipient, binds parameters, and applies.
    """
    
    def __init__(
        self,
        population: Population,
        synthesizer: MutationSynthesizer,
        applicator: MutationApplicator,
        max_depth: int = 100,
        max_retries: int = 20,
    ):
        self.population = population
        self.synthesizer = synthesizer
        self.applicator = applicator
        self.max_depth = max_depth
        self.max_retries = max_retries
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.EDIT
    
    def can_execute(self) -> bool:
        return self.population.can_recombine()
    
    def execute(self) -> GenerationResult:
        selection = self.population.select_for_edit(self.max_depth)
        if selection is None:
            return GenerationResult(
                tree=None,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": "no compatible nodes found"},
            )
        
        # SYNTH: Extract parameterized mutation
        mutation = self.synthesizer.extract(selection.donor_node)
        
        # MATCH + INSTANTIATE: Apply to recipient
        result = self.applicator.apply(mutation, selection.recipient_node)
        
        return GenerationResult(
            tree=result.mutant,
            strategy=self.strategy_type,
            success=result.is_fit,
            metadata={
                "donor_node": selection.donor_node.name,
                "recipient_node": selection.recipient_node.name,
                "bindings": len(result.bindings),
                "fitness_violations": result.fitness_violations.name,
            },
        )


class InsertStrategy(Strategy):
    """
    Insert new nodes at valid quantifier locations.
    
    Finds locations in recipient where grammar allows insertion
    (e.g., operation* in a block), then inserts from donor.
    """
    
    def __init__(
        self,
        population: Population,
        synthesizer: MutationSynthesizer,
        applicator: MutationApplicator,
        insert_patterns: dict,
        max_depth: int = 100,
        max_inserts_per_location: int = 20,
    ):
        self.population = population
        self.synthesizer = synthesizer
        self.applicator = applicator
        self.insert_patterns = insert_patterns
        self.max_depth = max_depth
        self.max_inserts_per_location = max_inserts_per_location
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.INSERT
    
    def can_execute(self) -> bool:
        return self.population.can_recombine() and bool(self.insert_patterns)
    
    def execute(self) -> GenerationResult:
        import random
        from copy import deepcopy
        from germinator.core.tree import get_root
        
        trees = self.population.select_for_insert(self.max_depth)
        if trees is None:
            return GenerationResult(
                tree=None,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": "not enough trees"},
            )
        
        recipient_tree, donor_tree = trees
        
        # Find valid insertion parents in recipient
        valid_parents = set(self.insert_patterns.keys()) & \
                        set(recipient_tree.nodes_by_name.keys())
        
        if not valid_parents:
            return GenerationResult(
                tree=recipient_tree.root,
                strategy=self.strategy_type,
                success=False,
                metadata={"error": "no valid insertion locations"},
            )
        
        # Try each parent type
        parent_list = list(valid_parents)
        random.shuffle(parent_list)
        
        for parent_name in parent_list:
            pattern = self.insert_patterns[parent_name]
            
            # Check if donor has required nodes
            if not pattern.child_rules <= set(donor_tree.nodes_by_name.keys()):
                continue
            
            # Try to find and apply insertion
            for parent_node in recipient_tree.nodes_by_name[parent_name]:
                location = self._find_insert_location(parent_node, pattern)
                if location is None:
                    continue
                
                # Get donor node to insert
                rule_name = location["rule_name"]
                if rule_name not in donor_tree.nodes_by_name:
                    continue
                
                donor_node = random.choice(list(donor_tree.nodes_by_name[rule_name]))
                
                # Check context matching
                if not self.population.context_matcher.matches(location["node"], donor_node):
                    continue
                
                # Apply as edit
                mutation = self.synthesizer.extract(donor_node)
                result = self.applicator.apply(mutation, location["node"])
                
                return GenerationResult(
                    tree=result.mutant,
                    strategy=self.strategy_type,
                    success=result.is_fit,
                    metadata={
                        "parent": parent_name,
                        "inserted": rule_name,
                        "fitness_violations": result.fitness_violations.name,
                    },
                )
        
        return GenerationResult(
            tree=recipient_tree.root,
            strategy=self.strategy_type,
            success=False,
            metadata={"error": "no valid insertion found"},
        )
    
    def _find_insert_location(self, parent_node, pattern) -> dict | None:
        """Find a valid insertion location within a parent node."""
        # Simplified version - real implementation would do full pattern matching
        # as in SynthFuzz's greedy_quantifier_match
        
        from germinator.grammar.patterns import QuantifierSpec
        
        children = parent_node.children or []
        
        for i, spec in enumerate(pattern.match_pattern):
            if isinstance(spec, QuantifierSpec):
                # Found a quantifier - this is a potential insertion point
                # Create a placeholder node
                from grammarinator.runtime.rule import UnparserRule
                placeholder = UnparserRule(name=spec.rule_name, parent=None)
                
                # Find where to insert (after existing nodes of this type)
                insert_idx = 0
                for j, child in enumerate(children):
                    if child.name == spec.rule_name:
                        insert_idx = j + 1
                
                parent_node.insert_child(idx=insert_idx, node=placeholder)
                
                return {
                    "node": placeholder,
                    "rule_name": spec.rule_name,
                    "parent": parent_node,
                }
        
        return None