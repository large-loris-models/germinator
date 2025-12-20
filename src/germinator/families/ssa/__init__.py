"""
SSA-based IR mutation family.

Handles IRs with:
- Static single assignment (def-use relationships)
- Type consistency requirements  
- Hierarchical structure (modules, functions, blocks)

Targets: MLIR, LLVM IR, WebAssembly, SPIR-V
"""

from germinator.families.ssa.base import SSAMutationFamily

__all__ = ["SSAMutationFamily"]