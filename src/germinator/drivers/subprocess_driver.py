"""
Generic subprocess-based driver.

Runs tests by invoking an external command.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from germinator.drivers.base import Driver, ExecutionResult, ResultType


@dataclass
class SubprocessDriverConfig:
    """Configuration for subprocess driver."""
    
    # Command to run (test input passed via stdin)
    command: list[str]
    
    # Timeout in seconds
    timeout: float = 30.0
    
    # Return codes to treat as "pass"
    pass_codes: set[int] = field(default_factory=lambda: {0})
    
    # Return codes to filter out (expected failures)
    filter_codes: set[int] = field(default_factory=set)
    
    # Regex patterns for stderr to filter out
    filter_patterns: list[str] = field(default_factory=list)
    
    # Environment variables
    env: dict[str, str] = field(default_factory=dict)


class SubprocessDriver(Driver):
    """
    Executes tests via subprocess.
    
    Runs a command with test input on stdin, captures output,
    and classifies the result.
    """
    
    def __init__(self, config: SubprocessDriverConfig):
        self.config = config
        self._filter_regex = None
        
        if config.filter_patterns:
            combined = "|".join(config.filter_patterns)
            self._filter_regex = re.compile(combined)
    
    @classmethod
    def from_command(cls, command: list[str], **kwargs) -> "SubprocessDriver":
        """Create driver from a simple command."""
        return cls(SubprocessDriverConfig(command=command, **kwargs))
    
    def execute(self, test_input: str) -> ExecutionResult:
        start_time = time.time()
        
        try:
            proc = subprocess.run(
                self.config.command,
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env={**dict(subprocess.os.environ), **self.config.env} if self.config.env else None,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            result_type = self.classify_result(proc.returncode, proc.stderr)
            
            return ExecutionResult(
                result_type=result_type,
                return_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_ms=duration_ms,
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                result_type=ResultType.TIMEOUT,
                return_code=-1,
                stderr="Timeout expired",
                duration_ms=self.config.timeout * 1000,
            )
            
        except Exception as e:
            return ExecutionResult(
                result_type=ResultType.ERROR,
                return_code=-1,
                stderr=str(e),
            )
    
    def classify_result(self, return_code: int, stderr: str) -> ResultType:
        # Check for pass
        if return_code in self.config.pass_codes:
            return ResultType.PASS
        
        # Check for filtered codes
        if return_code in self.config.filter_codes:
            return ResultType.FAIL_EXPECTED
        
        # Check for filtered patterns
        if self._filter_regex and self._filter_regex.search(stderr):
            return ResultType.FAIL_EXPECTED
        
        # Check for crash signals (negative return codes on Unix)
        if return_code < 0:
            return ResultType.CRASH
        
        # Otherwise it's interesting
        return ResultType.FAIL_INTERESTING