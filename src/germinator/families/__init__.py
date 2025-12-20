"""
Mutation families define domain-specific mutation strategies.
"""

from germinator.families.base import MutationFamily, FamilyConfig

__all__ = ["MutationFamily", "FamilyConfig"]


def get_family(name: str) -> MutationFamily:
    """
    Get a mutation family by name.
    
    Args:
        name: Family name (e.g., "ssa.mlir", "ssa.llvm")
    
    Returns:
        Configured MutationFamily instance
    
    Raises:
        ValueError: If family not found
    """
    # Registry of known families
    families = {
        "ssa.mlir": "germinator.families.ssa.mlir.MLIRMutationFamily",
    }
    
    if name not in families:
        available = ", ".join(families.keys())
        raise ValueError(f"Unknown family: {name}. Available: {available}")
    
    # Dynamic import
    module_path, class_name = families[name].rsplit(".", 1)
    from importlib import import_module
    module = import_module(module_path)
    family_class = getattr(module, class_name)
    
    return family_class()