"""
MLIR-specific test driver.

Handles running tests against mlir-opt or similar MLIR tools.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from germinator.drivers.base import Driver, ExecutionResult, ResultType
from germinator.drivers.subprocess_driver import SubprocessDriver, SubprocessDriverConfig


@dataclass
class MLIRDriverConfig:
    """Configuration for MLIR driver."""
    
    # Path to mlir-opt or similar binary
    binary: Path | str
    
    # Dialect -> pass associations
    # Maps dialect names to list of passes that work with that dialect
    dialect_passes: dict[str, list[str]] = field(default_factory=dict)
    
    # Maximum number of passes to run per test
    max_passes: int = 5
    
    # Whether to randomly select passes or use dialect associations
    random_passes: bool = False
    
    # Additional flags to always include
    extra_flags: list[str] = field(default_factory=lambda: ["-split-input-file"])
    
    # Return codes to filter out
    filter_codes: set[int] = field(default_factory=lambda: {1})
    
    # Error patterns to filter out (expected MLIR errors)
    filter_patterns: list[str] = field(default_factory=lambda: [
        r"error: expected",
        r"error: use of undeclared",
        r"error: redefinition",
    ])
    
    # Timeout per test
    timeout: float = 30.0
    
    # Random seed
    seed: int | None = None


class MLIRDriver(Driver):
    """
    Driver for MLIR-based compiler tools.
    
    Intelligently selects passes based on dialects present in the test,
    and filters out expected validation errors.
    """
    
    def __init__(self, config: MLIRDriverConfig):
        self.config = config
        self._rng = random.Random(config.seed)
        
        # Build subprocess config
        self._subprocess_config = SubprocessDriverConfig(
            command=[str(config.binary)],  # Passes added dynamically
            timeout=config.timeout,
            pass_codes={0},
            filter_codes=config.filter_codes,
            filter_patterns=config.filter_patterns,
            env={"LLVM_PROFILE_FILE": "/dev/null"},
        )
    
    @classmethod
    def from_config_file(cls, path: Path | str) -> "MLIRDriver":
        """Load driver from a TOML config file."""
        import tomllib
        
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)
        
        # Load dialect associations if specified
        dialect_passes = {}
        if "dialect_associations" in data:
            assoc_path = path.parent / data["dialect_associations"]
            with open(assoc_path) as f:
                dialect_passes = json.load(f)
        
        config = MLIRDriverConfig(
            binary=data.get("target_binary", "mlir-opt"),
            dialect_passes=dialect_passes,
            max_passes=data.get("max_options", 5),
            random_passes=data.get("use_random_options", False),
            filter_codes=set(data.get("retcode_filter", [1])),
            filter_patterns=data.get("error_filter_patterns", []),
            seed=data.get("seed"),
        )
        
        return cls(config)
    
    def execute(self, test_input: str) -> ExecutionResult:
        # Select passes for this test
        passes = self._select_passes(test_input)
        
        # Build command
        command = [
            str(self.config.binary),
            *passes,
            *self.config.extra_flags,
        ]
        
        # Create subprocess driver with this command
        subprocess_config = SubprocessDriverConfig(
            command=command,
            timeout=self.config.timeout,
            pass_codes={0},
            filter_codes=self.config.filter_codes,
            filter_patterns=self.config.filter_patterns,
            env={"LLVM_PROFILE_FILE": "/dev/null"},
        )
        
        driver = SubprocessDriver(subprocess_config)
        result = driver.execute(test_input)
        
        # Add pass info to metadata
        result.metadata["passes"] = passes
        
        return result
    
    def classify_result(self, return_code: int, stderr: str) -> ResultType:
        # Delegate to subprocess driver's logic
        driver = SubprocessDriver(self._subprocess_config)
        return driver.classify_result(return_code, stderr)
    
    def _select_passes(self, test_input: str) -> list[str]:
        """Select passes based on test content."""
        if self.config.random_passes:
            return self._random_passes()
        else:
            return self._dialect_based_passes(test_input)
    
    def _dialect_based_passes(self, test_input: str) -> list[str]:
        """Select passes based on dialects present in the test."""
        available = []
        
        for dialect, passes in self.config.dialect_passes.items():
            if dialect in test_input:
                available.extend(passes)
        
        if not available:
            return self._random_passes()
        
        # Remove duplicates while preserving some randomness
        available = list(set(available))
        self._rng.shuffle(available)
        
        return available[:self.config.max_passes]
    
    def _random_passes(self) -> list[str]:
        """Select random passes from all available."""
        all_passes = []
        for passes in self.config.dialect_passes.values():
            all_passes.extend(passes)
        
        all_passes = list(set(all_passes))
        
        if not all_passes:
            return []
        
        k = min(len(all_passes), self.config.max_passes)
        return self._rng.sample(all_passes, k)