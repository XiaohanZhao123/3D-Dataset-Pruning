"""Baseline pruning methods.

This module contains baseline coreset selection methods:
- DRoP: Difficulty-based Redundancy-aware data Pruning
- NUCS: Non-Uniform Class-wise Coreset Selection
"""

from pruning.baseline.drop import (
    compute_drop_quota,
    compute_per_class_metrics,
    get_class_sizes,
    select_with_drop_quota,
)

from pruning.baseline.nucs import (
    # Main entry point
    NUCSSelector,
    nucs_select,
    # Budget allocation
    compute_class_difficulties,
    compute_nucs_budgets,
    NUCSBudgetAllocator,
    # Window selection
    select_window_indices,
    find_optimal_endpoint,
    NUCSWindowSelector,
)

__all__ = [
    # DRoP quota allocation
    "compute_drop_quota",
    "compute_per_class_metrics",
    "get_class_sizes",
    "select_with_drop_quota",
    # NUCS main entry point
    "NUCSSelector",
    "nucs_select",
    # NUCS budget allocation
    "compute_class_difficulties",
    "compute_nucs_budgets",
    "NUCSBudgetAllocator",
    # NUCS window selection
    "select_window_indices",
    "find_optimal_endpoint",
    "NUCSWindowSelector",
]
