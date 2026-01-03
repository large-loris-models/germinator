"""
Base class for test execution drivers.

Drivers are responsible for:
- Running generated test cases against a target
- Determining if a result is interesting (crash, bug, etc.)
- Filtering out expected/uninteresting failures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class ResultType(Enum):
    """Classification of test execution results."""
    PASS = auto()          # Test passed (exit code 0, no issues)
    FAIL_EXPECTED = auto() # Test failed but it's expected (invalid input, etc.)
    FAIL_INTERESTING = auto()  # Test failed in an interesting way (potential bug)
    CRASH = auto()         # Target crashed
    TIMEOUT = auto()       # Test timed out
    ERROR = auto()         # Driver error (couldn't run test)


@dataclass
class ExecutionResult:
    """Result of executing a test case."""
    result_type: ResultType
    return_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_interesting(self) -> bool:
        """Check if this result is worth investigating."""
        return self.result_type in (ResultType.FAIL_INTERESTING, ResultType.CRASH)
    
    @property
    def is_pass(self) -> bool:
        return self.result_type == ResultType.PASS


class Driver(ABC):
    """
    Abstract base for test execution drivers.
    
    Subclasses implement target-specific execution and result classification.
    """
    
    @abstractmethod
    def execute(self, test_input: str) -> ExecutionResult:
        """
        Execute a test case.
        
        Args:
            test_input: The test case content (as string)
        
        Returns:
            ExecutionResult with outcome details
        """
        ...
    
    def execute_file(self, path: Path | str) -> ExecutionResult:
        """
        Execute a test case from a file.
        
        Args:
            path: Path to test file
        
        Returns:
            ExecutionResult with outcome details
        """
        path = Path(path)
        content = path.read_text()
        return self.execute(content)
    
    @abstractmethod
    def classify_result(self, return_code: int, stderr: str) -> ResultType:
        """
        Classify a test result.
        
        Args:
            return_code: Process exit code
            stderr: Standard error output
        
        Returns:
            ResultType classification
        """
        ...
    
    def is_interesting(self, result: ExecutionResult) -> bool:
        """
        Check if a result is interesting (potential bug).
        
        Override for custom filtering logic.
        """
        return result.is_interesting