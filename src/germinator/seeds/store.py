"""
Seed corpus storage and management.

Handles persistence, sampling, and organization of seed test cases
and their parsed tree representations.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import dill

logger = logging.getLogger(__name__)


@dataclass
class Seed:
    """A single seed test case."""
    path: Path
    tree: any = None  # Parsed tree (lazy loaded)
    
    def load_tree(self) -> any:
        """Load the tree from disk if not already loaded."""
        if self.tree is None:
            tree_path = self.path.with_suffix(self.path.suffix + ".tree")
            if tree_path.exists():
                with open(tree_path, "rb") as f:
                    self.tree = dill.load(f)
            else:
                raise FileNotFoundError(f"No tree file for {self.path}")
        return self.tree
    
    def save_tree(self, tree: any):
        """Save tree to disk."""
        self.tree = tree
        tree_path = self.path.with_suffix(self.path.suffix + ".tree")
        with open(tree_path, "wb") as f:
            dill.dump(tree, f)


@dataclass
class SeedStore:
    """
    Manages a corpus of seed test cases.
    
    Seeds are stored as:
    - Original text file: seeds/test_001.mlir
    - Parsed tree (pickle): seeds/test_001.mlir.tree
    
    The store supports:
    - Loading existing seeds from a directory
    - Adding new seeds (e.g., from LLM generation)
    - Random sampling for mutation
    - Persistence across runs
    """
    
    directory: Path
    seeds: list[Seed] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random)
    
    def __post_init__(self):
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load(cls, directory: Path | str, extension: str = ".mlir") -> "SeedStore":
        """
        Load seeds from a directory.
        
        Args:
            directory: Path to seeds directory
            extension: File extension to look for
        
        Returns:
            Populated SeedStore
        """
        directory = Path(directory)
        store = cls(directory=directory)
        
        if not directory.exists():
            logger.warning(f"Seeds directory does not exist: {directory}")
            return store
        
        for path in sorted(directory.glob(f"*{extension}")):
            # Skip tree files
            if ".tree" in path.suffixes:
                continue
            store.seeds.append(Seed(path=path))
        
        logger.info(f"Loaded {len(store.seeds)} seeds from {directory}")
        return store
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = random.Random(seed)
    
    def __len__(self) -> int:
        return len(self.seeds)
    
    def __iter__(self) -> Iterator[Seed]:
        return iter(self.seeds)
    
    def sample(self, n: int = 1) -> list[Seed]:
        """Sample n random seeds."""
        if n > len(self.seeds):
            return list(self.seeds)
        return self._rng.sample(self.seeds, n)
    
    def sample_one(self) -> Seed | None:
        """Sample a single random seed."""
        if not self.seeds:
            return None
        return self._rng.choice(self.seeds)
    
    def sample_pair(self) -> tuple[Seed, Seed] | None:
        """Sample two different seeds (for donor/recipient)."""
        if len(self.seeds) < 2:
            return None
        return tuple(self._rng.sample(self.seeds, 2))
    
    def add(self, content: str, name: str | None = None) -> Seed:
        """
        Add a new seed to the store.
        
        Args:
            content: The test case content
            name: Optional name (auto-generated if not provided)
        
        Returns:
            The created Seed
        """
        if name is None:
            name = f"seed_{len(self.seeds):06d}"
        
        # Determine extension from existing seeds or default
        ext = ".mlir"
        if self.seeds:
            ext = self.seeds[0].path.suffix
        
        path = self.directory / f"{name}{ext}"
        path.write_text(content)
        
        seed = Seed(path=path)
        self.seeds.append(seed)
        
        logger.debug(f"Added seed: {path}")
        return seed
    
    def add_with_tree(self, content: str, tree: any, name: str | None = None) -> Seed:
        """Add a seed with its pre-parsed tree."""
        seed = self.add(content, name)
        seed.save_tree(tree)
        return seed
    
    def remove(self, seed: Seed):
        """Remove a seed from the store."""
        if seed in self.seeds:
            self.seeds.remove(seed)
            if seed.path.exists():
                seed.path.unlink()
            tree_path = seed.path.with_suffix(seed.path.suffix + ".tree")
            if tree_path.exists():
                tree_path.unlink()
    
    def clear(self):
        """Remove all seeds."""
        for seed in list(self.seeds):
            self.remove(seed)
    
    def stats(self) -> dict:
        """Return statistics about the seed corpus."""
        total_size = sum(s.path.stat().st_size for s in self.seeds if s.path.exists())
        trees_loaded = sum(1 for s in self.seeds if s.tree is not None)
        
        return {
            "count": len(self.seeds),
            "total_size_bytes": total_size,
            "trees_loaded": trees_loaded,
        }