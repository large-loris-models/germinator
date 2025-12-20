"""
Insert patterns for mutation placement.

These patterns describe where insertions can happen in a grammar,
derived from quantifier rules (e.g., operation*, block+).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass
class QuantifierSpec:
    """
    Specification for a quantified element in a grammar rule.
    
    Examples:
        operation* -> QuantifierSpec(min=0, max=inf, rule_name="operation")
        block+     -> QuantifierSpec(min=1, max=inf, rule_name="block")
        arg?       -> QuantifierSpec(min=0, max=1, rule_name="arg")
    """
    min: int
    max: int | float  # Can be inf
    rule_name: str

    def allows_insertion(self, current_count: int) -> bool:
        """Check if another element can be inserted."""
        if self.max == inf:
            return True
        return current_count < self.max

    def is_optional(self) -> bool:
        return self.min == 0

    def is_repeating(self) -> bool:
        return self.max > 1 or self.max == inf
    
    def __hash__(self):
        return hash((self.min, self.max if self.max != inf else "inf", self.rule_name))
    
    def __eq__(self, other):
        if not isinstance(other, QuantifierSpec):
            return False
        return self.min == other.min and self.max == other.max and self.rule_name == other.rule_name


class InsertPattern:
    """
    Pattern describing valid insertion points within a parent rule.
    
    The match_pattern is a sequence of expected children:
    - str: A literal rule name that must match exactly
    - QuantifierSpec: A quantified element where insertion may occur
    
    Example for MLIR block:
        match_pattern = [
            "block_label",
            QuantifierSpec(min=0, max=inf, rule_name="operation"),
            "block_terminator",
        ]
        child_rules = {"operation"}
    """
    
    def __init__(self, match_pattern: list, child_rules: set[str]):
        self.match_pattern = tuple(match_pattern)
        self.child_rules = frozenset(child_rules)

    def get_quantifiers(self) -> list[QuantifierSpec]:
        """Get all quantifier specs in the pattern."""
        return [p for p in self.match_pattern if isinstance(p, QuantifierSpec)]

    def has_insertable_quantifier(self) -> bool:
        """Check if pattern has any quantifier that allows insertion."""
        return any(
            isinstance(p, QuantifierSpec) and p.max != 1
            for p in self.match_pattern
        )
    
    def __repr__(self):
        return f"InsertPattern(match_pattern={self.match_pattern}, child_rules={self.child_rules})"


InsertMatchPattern = InsertPattern