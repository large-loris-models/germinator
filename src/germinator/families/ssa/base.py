"""
Base class for SSA-style IR mutation families.

SSA IRs share common patterns:
- Values are defined exactly once (e.g., %x = ...)
- Values must be defined before use
- Types must be consistent
- Hierarchical structure (module > func > block > op)

This base class provides sensible defaults that work across
MLIR, LLVM IR, WASM, SPIR-V, etc. Specific targets override
only what's different.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from germinator.families.base import MutationFamily, FamilyConfig


class SSAMutationFamily(MutationFamily):
    """
    Base family for SSA-based IRs.
    
    Provides defaults tuned for SSA semantics:
    - Parameter blacklist excludes structural tokens
    - Fitness rules enforce def-use and no-duplicate-def
    - Context matching uses k=4, l=4, r=4
    
    Subclass this for specific IRs (MLIR, LLVM, etc.)
    """
    
    # Subclasses should override these
    GRAMMAR_FILE: ClassVar[str] = ""
    TARGET_NAME: ClassVar[str] = "ssa"
    
    # Common SSA patterns - subclasses can extend
    DEFAULT_PARAMETER_BLACKLIST: ClassVar[dict] = {
        # Structural tokens typically shouldn't be parameterized
    }
    
    DEFAULT_MUST_SUBSTITUTE: ClassVar[dict] = {
        # SSA values that must be substituted to maintain def-use
    }
    
    DEFAULT_NO_DUPLICATES: ClassVar[dict] = {
        # SSA values that cannot be defined twice
    }
    
    DEFAULT_VALID_TARGETS: ClassVar[list[str]] = [
        # Node types that are valid mutation targets
    ]
    
    @classmethod
    def name(cls) -> str:
        return f"ssa.{cls.TARGET_NAME}"
    
    @classmethod
    def default_config(cls) -> FamilyConfig:
        grammar_path = None
        if cls.GRAMMAR_FILE:
            # Look for grammar in package
            grammar_path = Path(__file__).parent / cls.TARGET_NAME / cls.GRAMMAR_FILE
            if not grammar_path.exists():
                grammar_path = None
        
        return FamilyConfig(
            k_ancestors=4,
            l_siblings=4,
            r_siblings=4,
            parameter_blacklist=cls.DEFAULT_PARAMETER_BLACKLIST.copy(),
            must_substitute=cls.DEFAULT_MUST_SUBSTITUTE.copy(),
            no_duplicates=cls.DEFAULT_NO_DUPLICATES.copy(),
            grammar_path=grammar_path,
            valid_mutation_targets=cls.DEFAULT_VALID_TARGETS.copy(),
        )
    
    @classmethod
    def from_config_file(cls, config_path: Path) -> "SSAMutationFamily":
        """
        Load family configuration from a TOML file.
        
        Expected format:
```toml
        [context]
        k_ancestors = 4
        l_siblings = 4
        r_siblings = 4
        
        [parameterization]
        blacklist = ["op_name.parent", "some_token"]
        
        [fitness]
        must_substitute = ["value_use.operation"]
        no_duplicates = ["value_def.block"]
        
        [grammar]
        path = "mlir.g4"
        
        [targets]
        valid = ["operation", "block"]
```
        """
        import tomllib
        
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        
        config = cls.default_config()
        
        # Context
        if "context" in data:
            ctx = data["context"]
            config.k_ancestors = ctx.get("k_ancestors", config.k_ancestors)
            config.l_siblings = ctx.get("l_siblings", config.l_siblings)
            config.r_siblings = ctx.get("r_siblings", config.r_siblings)
        
        # Parameterization
        if "parameterization" in data:
            config.parameter_blacklist = cls._parse_rule_list(
                data["parameterization"].get("blacklist", [])
            )
        
        # Fitness
        if "fitness" in data:
            fit = data["fitness"]
            config.must_substitute = cls._parse_rule_list(fit.get("must_substitute", []))
            config.no_duplicates = cls._parse_rule_list(fit.get("no_duplicates", []))
        
        # Grammar
        if "grammar" in data:
            grammar_file = data["grammar"].get("path")
            if grammar_file:
                config.grammar_path = config_path.parent / grammar_file
        
        # Targets
        if "targets" in data:
            config.valid_mutation_targets = data["targets"].get("valid", [])
        
        return cls(config)
    
    @staticmethod
    def _parse_rule_list(rules: list[str]) -> dict[str, list[str] | str]:
        """
        Parse a list of rules into the blacklist format.
        
        Formats:
        - "rule_name" -> {"rule_name": "*"}
        - "child.parent" -> {"child": ["parent"]}
        """
        result = {}
        for rule in rules:
            if "." in rule:
                child, parent = rule.split(".", 1)
                if child not in result:
                    result[child] = []
                if result[child] != "*":
                    result[child].append(parent)
            else:
                result[rule] = "*"
        return result