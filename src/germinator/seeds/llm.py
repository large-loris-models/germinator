"""
LLM-based seed generation.

Uses an LLM to generate initial seeds for cold-start scenarios
or to augment the corpus when coverage stalls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM seed generation."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7


class LLMSeeder:
    """
    Generates seed test cases using an LLM.
    
    The LLM is prompted with:
    - Grammar description or examples
    - Target domain info (e.g., MLIR dialects)
    - Constraints to follow
    
    Generated seeds are validated before being added to the corpus.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert at generating test cases for compiler intermediate representations.
Generate syntactically valid test cases that exercise interesting code paths.
Focus on edge cases, unusual combinations, and potential corner cases.
Output only the test case, no explanations."""
    
    def __init__(
        self,
        config: LLMConfig | None = None,
        validator: Callable[[str], bool] | None = None,
        system_prompt: str | None = None,
    ):
        """
        Args:
            config: LLM configuration
            validator: Function to validate generated seeds (returns True if valid)
            system_prompt: Custom system prompt (uses default if not provided)
        """
        self.config = config or LLMConfig()
        self.validator = validator
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic()
            except ImportError:
                raise ImportError(
                    "anthropic package required for LLM seeding. "
                    "Install with: pip install germinator[llm]"
                )
        return self._client
    
    def generate(
        self,
        prompt: str,
        n: int = 1,
        examples: list[str] | None = None,
    ) -> list[str]:
        """
        Generate seed test cases.
        
        Args:
            prompt: Description of what to generate
            n: Number of seeds to generate
            examples: Optional example seeds to guide generation
        
        Returns:
            List of generated (and validated) seeds
        """
        seeds = []
        attempts = 0
        max_attempts = n * 3  # Allow some failures
        
        while len(seeds) < n and attempts < max_attempts:
            attempts += 1
            
            # Build prompt with examples if provided
            full_prompt = prompt
            if examples:
                examples_text = "\n---\n".join(examples[:3])  # Limit examples
                full_prompt = f"Examples:\n{examples_text}\n\n{prompt}"
            
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                
                content = response.content[0].text.strip()
                
                # Validate if validator provided
                if self.validator is None or self.validator(content):
                    seeds.append(content)
                    logger.debug(f"Generated valid seed ({len(seeds)}/{n})")
                else:
                    logger.debug(f"Generated seed failed validation (attempt {attempts})")
                    
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
        
        if len(seeds) < n:
            logger.warning(f"Only generated {len(seeds)}/{n} valid seeds")
        
        return seeds
    
    def generate_for_dialect(
        self,
        dialect: str,
        n: int = 10,
        examples: list[str] | None = None,
    ) -> list[str]:
        """
        Generate seeds for a specific MLIR dialect.
        
        Args:
            dialect: Dialect name (e.g., "arith", "scf", "linalg")
            n: Number of seeds to generate
            examples: Optional examples
        
        Returns:
            List of generated seeds
        """
        prompt = f"""Generate {n} different MLIR test cases using the '{dialect}' dialect.
Each test case should be a complete, valid MLIR module.
Focus on:
- Different operations from the {dialect} dialect
- Various type combinations
- Edge cases and boundary conditions
- Interesting control flow patterns (if applicable)

Generate only the MLIR code, one complete module."""
        
        return self.generate(prompt, n=n, examples=examples)
    
    def generate_diverse(
        self,
        dialects: list[str],
        n_per_dialect: int = 5,
        examples: list[str] | None = None,
    ) -> list[str]:
        """
        Generate diverse seeds across multiple dialects.
        
        Args:
            dialects: List of dialect names
            n_per_dialect: Seeds per dialect
            examples: Optional examples
        
        Returns:
            Combined list of seeds
        """
        all_seeds = []
        
        for dialect in dialects:
            logger.info(f"Generating seeds for dialect: {dialect}")
            seeds = self.generate_for_dialect(dialect, n=n_per_dialect, examples=examples)
            all_seeds.extend(seeds)
        
        return all_seeds


def create_grammar_validator(grammar_path: Path) -> Callable[[str], bool]:
    """
    Create a validator that checks if content parses under a grammar.
    
    Args:
        grammar_path: Path to the .g4 grammar file
    
    Returns:
        Validator function
    """
    # This would integrate with Grammarinator's parser
    # For now, return a permissive validator
    def validator(content: str) -> bool:
        # Basic sanity checks
        if not content.strip():
            return False
        # Could add grammar-based parsing here
        return True
    
    return validator
