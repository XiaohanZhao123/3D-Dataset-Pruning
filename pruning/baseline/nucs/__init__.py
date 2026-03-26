"""NUCS (Non-Uniform Class-wise Coreset Selection) implementation.

Based on: "Non-Uniform Class-Wise Coreset Selection: Characterizing
Category Difficulty for Data-Efficient Transfer Learning" (arXiv:2504.13234)

Components:
- budget.py: Class-wise budget allocation based on difficulty
- window.py: Window-based sample selection with KRR endpoint optimization
- selector.py: Integrated NUCS selector (main entry point)

Usage:
    # Full pipeline (recommended)
    from pruning.baseline.nucs import NUCSSelector
    selector = NUCSSelector(num_classes=40, total_samples=400)
    selected_indices, info = selector.select(
        scores=el2n_scores,
        labels=labels,
        features=embeddings,
        label_to_indices=label_to_indices,
    )

    # Or use convenience function
    from pruning.baseline.nucs import nucs_select
    selected_indices, info = nucs_select(
        scores=el2n_scores, labels=labels, features=embeddings,
        label_to_indices=label_to_indices,
        num_classes=40, total_samples=400,
    )
"""

from pruning.baseline.nucs.budget import (
    compute_class_difficulties,
    compute_nucs_budgets,
    NUCSBudgetAllocator,
)

from pruning.baseline.nucs.window import (
    select_window_indices,
    train_krr,
    evaluate_krr,
    find_optimal_endpoint,
    NUCSWindowSelector,
)

from pruning.baseline.nucs.selector import (
    NUCSSelector,
    nucs_select,
)

__all__ = [
    # Main entry point
    "NUCSSelector",
    "nucs_select",
    # Budget allocation
    "compute_class_difficulties",
    "compute_nucs_budgets",
    "NUCSBudgetAllocator",
    # Window selection
    "select_window_indices",
    "train_krr",
    "evaluate_krr",
    "find_optimal_endpoint",
    "NUCSWindowSelector",
]
