"""
Main generation engine.

Orchestrates test case generation using configurable strategies.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from germinator.core import ContextMatcher
from germinator.families import MutationFamily
from germinator.seeds import Population, SeedStore
from germinator.engine.strategies import (
    Strategy,
    StrategyType,
    GenerationResult,
    GenerateStrategy,
    MutateStrategy,
    RecombineStrategy,
    EditStrategy,
    InsertStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for the generator."""
    
    # Output
    output_dir: Path = Path("./output")
    output_pattern: str = "test_%06d"
    
    # Strategies to enable
    enable_generate: bool = True
    enable_mutate: bool = True
    enable_recombine: bool = True
    enable_edit: bool = True
    enable_insert: bool = True
    
    # Strategy weights (for random selection)
    weights: dict[StrategyType, float] = field(default_factory=lambda: {
        StrategyType.GENERATE: 1.0,
        StrategyType.MUTATE: 1.0,
        StrategyType.RECOMBINE: 1.0,
        StrategyType.EDIT: 2.0,  # Prefer SynthFuzz-style edits
        StrategyType.INSERT: 1.0,
    })
    
    # Generation limits
    max_depth: int = 100
    max_retries: int = 20
    
    # Fitness
    require_fitness: bool = True
    
    # Persistence
    keep_trees: bool = True
    save_metadata: bool = False


class Generator:
    """
    Main test case generator.
    
    Combines multiple strategies to generate diverse test cases.
    Integrates with mutation families for domain-specific behavior.
    """
    
    def __init__(
        self,
        family: MutationFamily,
        population: Population | None = None,
        generator_factory: Any = None,
        insert_patterns: dict | None = None,
        config: GeneratorConfig | None = None,
        transformers: list[Callable] | None = None,
        serializer: Callable[[Any], str] | None = None,
    ):
        """
        Args:
            family: Mutation family defining domain-specific behavior
            population: Population of seed trees
            generator_factory: Factory for grammar-based generators
            insert_patterns: Patterns for insertion strategy
            config: Generator configuration
            transformers: Post-processing transforms for trees
            serializer: Function to convert tree to string
        """
        self.family = family
        self.population = population
        self.generator_factory = generator_factory
        self.insert_patterns = insert_patterns or {}
        self.config = config or GeneratorConfig()
        self.transformers = transformers or []
        self.serializer = serializer or str
        
        self._strategies: list[Strategy] = []
        self._rng = random.Random()
        self._index = 0
        
        self._setup_strategies()
        self._setup_output()
    
    def _setup_strategies(self):
        """Initialize enabled strategies."""
        cfg = self.config
        
        if cfg.enable_generate and self.generator_factory:
            self._strategies.append(GenerateStrategy(
                generator_factory=self.generator_factory,
                max_depth=cfg.max_depth,
            ))
        
        if cfg.enable_mutate and self.population and self.generator_factory:
            self._strategies.append(MutateStrategy(
                population=self.population,
                generator_factory=self.generator_factory,
                max_depth=cfg.max_depth,
            ))
        
        if cfg.enable_recombine and self.population:
            self._strategies.append(RecombineStrategy(
                population=self.population,
                max_depth=cfg.max_depth,
            ))
        
        if cfg.enable_edit and self.population:
            self._strategies.append(EditStrategy(
                population=self.population,
                synthesizer=self.family.synthesizer,
                applicator=self.family.applicator,
                max_depth=cfg.max_depth,
                max_retries=cfg.max_retries,
            ))
        
        if cfg.enable_insert and self.population and self.insert_patterns:
            self._strategies.append(InsertStrategy(
                population=self.population,
                synthesizer=self.family.synthesizer,
                applicator=self.family.applicator,
                insert_patterns=self.insert_patterns,
                max_depth=cfg.max_depth,
            ))
    
    def _setup_output(self):
        """Setup output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = random.Random(seed)
        if self.population:
            self.population.set_seed(seed)
    
    def _select_strategy(self) -> Strategy | None:
        """Select a strategy based on weights and availability."""
        available = [s for s in self._strategies if s.can_execute()]
        
        if not available:
            return None
        
        # Weight-based selection
        weights = [self.config.weights.get(s.strategy_type, 1.0) for s in available]
        total = sum(weights)
        
        if total == 0:
            return self._rng.choice(available)
        
        r = self._rng.random() * total
        cumulative = 0
        
        for strategy, weight in zip(available, weights):
            cumulative += weight
            if r <= cumulative:
                return strategy
        
        return available[-1]
    
    def generate_one(self) -> tuple[str | None, GenerationResult]:
        """
        Generate a single test case.
        
        Returns:
            Tuple of (output_path, generation_result)
        """
        strategy = self._select_strategy()
        
        if strategy is None:
            logger.warning("No strategies available")
            return None, GenerationResult(
                tree=None,
                strategy=StrategyType.GENERATE,
                success=False,
                metadata={"error": "no strategies available"},
            )
        
        # Execute with retries for fitness
        result = None
        attempts = 0
        
        while attempts < self.config.max_retries:
            attempts += 1
            result = strategy.execute()
            
            if result.success or not self.config.require_fitness:
                break
            
            # For edit/insert, retry on fitness failure
            if strategy.strategy_type not in (StrategyType.EDIT, StrategyType.INSERT):
                break
        
        if result is None or result.tree is None:
            return None, result
        
        # Apply transformers
        tree = deepcopy(result.tree)
        for transformer in self.transformers:
            tree = transformer(tree)
        
        # Serialize
        output = self.serializer(tree)
        
        # Save to file
        filename = self.config.output_pattern % self._index
        output_path = self.config.output_dir / filename
        output_path.write_text(output)
        
        # Add to population if keeping trees
        if self.config.keep_trees and self.population:
            self.population.add_tree(tree, output_path)
        
        self._index += 1
        
        return str(output_path), result
    
    def generate(self, n: int) -> list[tuple[str, GenerationResult]]:
        """
        Generate n test cases.
        
        Args:
            n: Number of test cases to generate
        
        Returns:
            List of (output_path, result) tuples
        """
        results = []
        
        for i in range(n):
            path, result = self.generate_one()
            if path:
                results.append((path, result))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{n} test cases")
        
        return results
    
    def generate_iter(self, n: int | None = None):
        """
        Generate test cases as an iterator.
        
        Args:
            n: Number to generate (None = infinite)
        
        Yields:
            (output_path, result) tuples
        """
        count = 0
        while n is None or count < n:
            path, result = self.generate_one()
            if path:
                yield path, result
                count += 1
    
    def stats(self) -> dict:
        """Return generation statistics."""
        return {
            "total_generated": self._index,
            "strategies_available": [s.strategy_type.name for s in self._strategies if s.can_execute()],
            "population_size": len(self.population._population._files) if self.population else 0,
        }