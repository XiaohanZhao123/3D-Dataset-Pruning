"""Integrated NUCS Selector.

Combines budget allocation and window selection into a single pipeline.
This is the main entry point for using NUCS coreset selection.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from pruning.baseline.nucs.budget import (
    compute_class_difficulties,
    compute_nucs_budgets,
    NUCSBudgetAllocator,
)
from pruning.baseline.nucs.window import (
    find_optimal_endpoint,
    select_window_indices,
    NUCSWindowSelector,
)

logger = logging.getLogger(__name__)


class NUCSSelector:
    """Integrated NUCS coreset selector.

    Combines:
    1. Budget allocation based on class difficulty
    2. Window-based sample selection with KRR endpoint optimization

    Example usage:
        >>> selector = NUCSSelector(num_classes=40, total_samples=400)
        >>> selected_indices = selector.select(
        ...     scores=el2n_scores,
        ...     labels=labels,
        ...     features=embeddings,
        ...     label_to_indices=label_to_indices,
        ... )
    """

    def __init__(
        self,
        num_classes: int,
        total_samples: int,
        # Budget allocation params
        aggregation: str = "mean",
        min_samples_per_class: int = 1,
        # Window selection params
        endpoint_candidates: Optional[List[float]] = None,
        krr_alpha: float = 1.0,
        krr_kernel: str = "rbf",
        krr_gamma: Optional[float] = None,
        # Fixed endpoint (skip KRR optimization)
        fixed_endpoint: Optional[float] = None,
    ):
        """Initialize NUCS selector.

        Args:
            num_classes: Total number of classes
            total_samples: Total samples to select across all classes
            aggregation: Difficulty aggregation ("mean", "median", "p75")
            min_samples_per_class: Minimum samples per class
            endpoint_candidates: Window endpoints to test
            krr_alpha: KRR regularization strength
            krr_kernel: KRR kernel type
            krr_gamma: KRR kernel coefficient
            fixed_endpoint: If provided, skip KRR and use this endpoint
        """
        self.num_classes = num_classes
        self.total_samples = total_samples

        # Budget allocator
        self.budget_allocator = NUCSBudgetAllocator(
            num_classes=num_classes,
            aggregation=aggregation,
            min_samples_per_class=min_samples_per_class,
        )

        # Window selector
        self.window_selector = NUCSWindowSelector(
            num_classes=num_classes,
            krr_alpha=krr_alpha,
            krr_kernel=krr_kernel,
            krr_gamma=krr_gamma,
            endpoint_candidates=endpoint_candidates or [0.25, 0.5, 0.75, 1.0],
        )

        self.fixed_endpoint = fixed_endpoint

    def compute_budgets(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        label_to_indices: Dict[int, List[int]],
    ) -> Dict[int, int]:
        """Compute per-class budgets based on difficulty.

        Args:
            scores: [N] difficulty scores
            labels: [N] class labels
            label_to_indices: Dict mapping class to dataset indices

        Returns:
            Dict mapping class to budget
        """
        return self.budget_allocator.allocate(
            scores=scores,
            labels=labels,
            label_to_indices=label_to_indices,
            total_samples=self.total_samples,
        )

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        label_to_indices: Dict[int, List[int]],
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Select coreset using full NUCS pipeline.

        Args:
            scores: [N] difficulty scores (e.g., EL2N)
            labels: [N] class labels
            features: [N, D] feature embeddings
            label_to_indices: Dict mapping class to dataset indices
            val_features: Validation features for KRR (optional)
            val_labels: Validation labels for KRR (optional)

        Returns:
            Tuple of (selected_indices, info_dict)
            info_dict contains:
                - budgets: per-class budgets
                - difficulties: per-class difficulties
                - endpoint: selected window endpoint
                - endpoint_results: KRR results per endpoint
        """
        # Step 1: Compute per-class budgets
        logger.info("Step 1: Computing per-class budgets based on difficulty")
        difficulties = self.budget_allocator.compute_difficulties(scores, labels)
        budgets = self.budget_allocator.allocate_from_difficulties(
            class_difficulties=difficulties,
            label_to_indices=label_to_indices,
            total_samples=self.total_samples,
        )

        # Log budget allocation
        logger.info("Per-class budgets:")
        for c in sorted(budgets.keys()):
            class_size = len(label_to_indices.get(c, []))
            logger.info(
                f"  Class {c}: difficulty={difficulties[c]:.4f}, "
                f"budget={budgets[c]}/{class_size}"
            )

        # Step 2: Find optimal endpoint (or use fixed)
        if self.fixed_endpoint is not None:
            logger.info(f"Step 2: Using fixed endpoint: {self.fixed_endpoint}")
            endpoint = self.fixed_endpoint
            endpoint_results = {endpoint: None}
        else:
            logger.info("Step 2: Finding optimal window endpoint using KRR")
            endpoint, endpoint_results = self.window_selector.find_optimal_endpoint(
                scores=scores,
                labels=labels,
                features=features,
                budgets=budgets,
                val_features=val_features,
                val_labels=val_labels,
            )

        # Step 3: Select samples using window
        logger.info(f"Step 3: Selecting samples with endpoint={endpoint:.2f}")
        selected_indices = self.window_selector.select(
            scores=scores,
            labels=labels,
            budgets=budgets,
            endpoint=endpoint,
        )

        # Compile info dict
        info = {
            "budgets": budgets,
            "difficulties": difficulties,
            "endpoint": endpoint,
            "endpoint_results": endpoint_results,
            "total_selected": len(selected_indices),
        }

        logger.info(f"Selected {len(selected_indices)} samples total")

        return selected_indices, info

    def select_without_krr(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        label_to_indices: Dict[int, List[int]],
        endpoint: float = 0.75,
    ) -> Tuple[torch.Tensor, Dict]:
        """Select coreset without KRR optimization.

        Faster alternative when you don't need endpoint optimization
        or don't have features available.

        Args:
            scores: [N] difficulty scores
            labels: [N] class labels
            label_to_indices: Dict mapping class to dataset indices
            endpoint: Window endpoint to use (default: 0.75)

        Returns:
            Tuple of (selected_indices, info_dict)
        """
        # Step 1: Compute budgets
        difficulties = self.budget_allocator.compute_difficulties(scores, labels)
        budgets = self.budget_allocator.allocate_from_difficulties(
            class_difficulties=difficulties,
            label_to_indices=label_to_indices,
            total_samples=self.total_samples,
        )

        # Step 2: Select with fixed endpoint
        selected_indices = self.window_selector.select(
            scores=scores,
            labels=labels,
            budgets=budgets,
            endpoint=endpoint,
        )

        info = {
            "budgets": budgets,
            "difficulties": difficulties,
            "endpoint": endpoint,
            "endpoint_results": None,
            "total_selected": len(selected_indices),
        }

        return selected_indices, info


def nucs_select(
    scores: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    label_to_indices: Dict[int, List[int]],
    num_classes: int,
    total_samples: int,
    aggregation: str = "mean",
    min_samples_per_class: int = 1,
    endpoint_candidates: Optional[List[float]] = None,
    krr_alpha: float = 1.0,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Convenience function for NUCS selection.

    Args:
        scores: [N] difficulty scores
        labels: [N] class labels
        features: [N, D] feature embeddings
        label_to_indices: Dict mapping class to dataset indices
        num_classes: Total number of classes
        total_samples: Total samples to select
        aggregation: Difficulty aggregation method
        min_samples_per_class: Minimum per class
        endpoint_candidates: Endpoints to test
        krr_alpha: KRR regularization
        val_features: Validation features
        val_labels: Validation labels

    Returns:
        Tuple of (selected_indices, info_dict)
    """
    selector = NUCSSelector(
        num_classes=num_classes,
        total_samples=total_samples,
        aggregation=aggregation,
        min_samples_per_class=min_samples_per_class,
        endpoint_candidates=endpoint_candidates,
        krr_alpha=krr_alpha,
    )

    return selector.select(
        scores=scores,
        labels=labels,
        features=features,
        label_to_indices=label_to_indices,
        val_features=val_features,
        val_labels=val_labels,
    )
