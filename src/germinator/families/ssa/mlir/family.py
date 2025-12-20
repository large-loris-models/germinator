"""
MLIR mutation family.

MLIR-specific configuration for SSA-based mutations.
"""

from typing import ClassVar

from germinator.families.ssa.base import SSAMutationFamily


class MLIRMutationFamily(SSAMutationFamily):
    """
    Mutation family for MLIR (all dialects).
    
    MLIR-specific considerations:
    - Values are prefixed with % (e.g., %0, %arg0)
    - Types follow values with : annotation
    - Operations have dialect.op_name format
    - Blocks are labeled with ^name
    - Regions are enclosed in {}
    """
    
    TARGET_NAME: ClassVar[str] = "mlir"
    GRAMMAR_FILE: ClassVar[str] = "mlir.g4"
    
    # MLIR-specific blacklist
    # These are structural tokens that shouldn't be parameterized
    DEFAULT_PARAMETER_BLACKLIST: ClassVar[dict] = {
        # Don't parameterize operation names - they're structural
        "op_name": "*",
        # Don't parameterize dialect names
        "dialect_namespace": "*",
        # Block labels are structural
        "caret_id": "*",
        # Don't parameterize attribute names
        "bare_id": ["attribute_entry"],
    }
    
    # Values that must be substituted for valid mutations
    DEFAULT_MUST_SUBSTITUTE: ClassVar[dict] = {
        # SSA value uses must be substituted to maintain def-use
        "value_use": "*",
    }
    
    # Values that cannot appear twice (SSA property)
    DEFAULT_NO_DUPLICATES: ClassVar[dict] = {
        # SSA value definitions cannot be duplicated
        "value_def": "*",
        # Block arguments are definitions too
        "block_arg": "*",
    }
    
    # Valid mutation targets in MLIR
    DEFAULT_VALID_TARGETS: ClassVar[list[str]] = [
        "operation",
        "block",
        "region",
    ]
    
    @classmethod
    def name(cls) -> str:
        return "ssa.mlir"