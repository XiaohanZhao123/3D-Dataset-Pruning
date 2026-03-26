"""NUCS Class-wise Budget Allocation.

Implements the budget allocation component of NUCS (Non-Uniform Class-wise
Coreset Selection). Allocates per-class sample budgets based on intrinsic
class difficulty.

Paper formula:
    r_i = ((1 - alpha) * S_i) / T

Where:
    - r_i: selection ratio for class i
    - alpha: overall pruning ratio (e.g., 0.9 means keep 10%)
    - S_i: average difficulty score for class i
    - T: normalization factor ensuring sum(r_i) = (1 - alpha)

Key insight: Harder classes get more samples, easier classes get fewer.
"""

import logging
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


def compute_class_difficulties(
    scores: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    aggregation: str = "mean",
) -> Dict[int, float]:
    """Compute per-class difficulty from sample-level scores.

    Args:
        scores: [N] tensor of difficulty scores (e.g., EL2N, loss)
                Higher score = harder sample
        labels: [N] tensor of class labels
        num_classes: Total number of classes
        aggregation: How to aggregate per-sample scores to class difficulty
                    - "mean": Average score (default, used in NUCS paper)
                    - "median": Median score (robust to outliers)
                    - "p75": 75th percentile (focuses on harder samples)

    Returns:
        Dict mapping class label to difficulty score

    Raises:
        ValueError: If scores and labels have different lengths or unknown aggregation

    Example:
        >>> scores = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.5])
        >>> labels = torch.tensor([0, 0, 1, 1, 2])
        >>> difficulties = compute_class_difficulties(scores, labels, 3)
        >>> # Class 0: mean(0.1, 0.2) = 0.15 (easy)
        >>> # Class 1: mean(0.8, 0.9) = 0.85 (hard)
        >>> # Class 2: mean(0.5) = 0.5 (medium)
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have same length, "
            f"got {len(scores)} and {len(labels)}"
        )

    if aggregation not in ("mean", "median", "p75"):
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            f"Supported: 'mean', 'median', 'p75'"
        )

    difficulties = {}

    for c in range(num_classes):
        class_mask = labels == c
        class_scores = scores[class_mask]

        if len(class_scores) == 0:
            # No samples for this class - assign zero difficulty
            # (will get zero budget, which is appropriate)
            difficulties[c] = 0.0
            logger.warning(f"Class {c} has no samples, assigning difficulty=0")
            continue

        if aggregation == "mean":
            difficulty = class_scores.mean().item()
        elif aggregation == "median":
            difficulty = class_scores.median().item()
        elif aggregation == "p75":
            difficulty = torch.quantile(class_scores.float(), 0.75).item()

        difficulties[c] = difficulty

    return difficulties


def compute_nucs_budgets(
    total_samples: int,
    class_difficulties: Dict[int, float],
    label_to_indices: Dict[int, List[int]],
    min_samples_per_class: int = 1,
) -> Dict[int, int]:
    """Compute per-class budgets based on NUCS difficulty-proportional allocation.

    NUCS Formula:
        r_i = ((1 - alpha) * S_i) / T

    Where:
        - alpha = 1 - (total_samples / dataset_size) is the pruning ratio
        - S_i = class difficulty
        - T = sum(S_i) normalizes so budgets sum to total_samples

    Simplified to:
        budget_i = total_samples * (S_i / sum(S_i))

    This allocates more samples to harder classes and fewer to easier classes.

    Args:
        total_samples: Total number of samples to select across all classes
        class_difficulties: Dict mapping class label to difficulty score
                           (from compute_class_difficulties)
        label_to_indices: Dict mapping class label to list of dataset indices
                         (used to cap budgets at class size)
        min_samples_per_class: Minimum samples per class (default: 1)
                              Set to 0 to allow classes to be completely pruned

    Returns:
        Dict mapping class label to integer budget

    Raises:
        ValueError: If total_samples exceeds dataset size or difficulties invalid

    Example:
        >>> difficulties = {0: 0.15, 1: 0.85, 2: 0.50}  # sum = 1.5
        >>> label_to_indices = {0: [0,1,2], 1: [3,4,5], 2: [6,7,8]}
        >>> budgets = compute_nucs_budgets(6, difficulties, label_to_indices)
        >>> # Class 0: 6 * (0.15/1.5) = 0.6 -> 1 (min)
        >>> # Class 1: 6 * (0.85/1.5) = 3.4 -> 3
        >>> # Class 2: 6 * (0.50/1.5) = 2.0 -> 2
    """
    num_classes = len(class_difficulties)
    dataset_size = sum(len(indices) for indices in label_to_indices.values())

    if total_samples > dataset_size:
        raise ValueError(
            f"total_samples ({total_samples}) exceeds dataset size ({dataset_size})"
        )

    # Validate difficulties first (before computing total)
    for c, difficulty in class_difficulties.items():
        if difficulty < 0:
            raise ValueError(f"Class {c} has negative difficulty {difficulty}")

    # Sum of all difficulties for normalization
    total_difficulty = sum(class_difficulties.values())

    if total_difficulty <= 0:
        # All classes have zero difficulty - fall back to uniform allocation
        logger.warning(
            "All classes have zero difficulty, falling back to uniform allocation"
        )
        num_classes = len(class_difficulties)
        base_per_class = total_samples // num_classes
        remainder = total_samples % num_classes

        budgets = {}
        for i, c in enumerate(sorted(class_difficulties.keys())):
            class_size = len(label_to_indices.get(c, []))
            budget = base_per_class + (1 if i < remainder else 0)
            budget = min(budget, class_size)
            budgets[c] = budget

        return budgets

    # Compute raw proportional budgets
    raw_budgets = {}
    for c, difficulty in class_difficulties.items():

        # Proportional allocation: budget_i = total * (S_i / sum(S_i))
        raw_budget = total_samples * (difficulty / total_difficulty)
        raw_budgets[c] = raw_budget

    # Convert to integers with constraints
    budgets = {}
    allocated = 0

    # First pass: floor allocation with minimum constraint
    for c in sorted(raw_budgets.keys()):
        class_size = len(label_to_indices.get(c, []))
        raw = raw_budgets[c]

        # Apply minimum and maximum constraints
        budget = max(min_samples_per_class, int(raw))
        budget = min(budget, class_size)  # Can't select more than available

        budgets[c] = budget
        allocated += budget

    # Second pass: distribute remainder to classes with capacity
    remainder = total_samples - allocated

    if remainder > 0:
        # Keep distributing until we've allocated all or no class has capacity
        while remainder > 0:
            # Find classes with remaining capacity, sorted by how much they "deserve"
            # (fractional part of raw allocation, or capacity if capped)
            classes_with_capacity = [
                c for c in budgets.keys()
                if budgets[c] < len(label_to_indices.get(c, []))
            ]

            if not classes_with_capacity:
                # No class has capacity left
                break

            # Sort by fractional part of raw allocation (fairest distribution)
            classes_with_capacity.sort(
                key=lambda c: raw_budgets[c] - int(raw_budgets[c]),
                reverse=True
            )

            # Distribute one sample to each class with capacity (round-robin)
            distributed_this_round = 0
            for c in classes_with_capacity:
                if remainder <= 0:
                    break

                class_size = len(label_to_indices.get(c, []))
                if budgets[c] < class_size:
                    budgets[c] += 1
                    remainder -= 1
                    distributed_this_round += 1

            if distributed_this_round == 0:
                # No progress made, avoid infinite loop
                break

    elif remainder < 0:
        # Over-allocated due to minimum constraint
        # Remove from classes that got more than their raw allocation
        overshoot = -remainder
        overallocated = [
            (c, budgets[c] - raw_budgets[c])
            for c in budgets.keys()
            if budgets[c] > raw_budgets[c] and budgets[c] > min_samples_per_class
        ]
        overallocated.sort(key=lambda x: x[1], reverse=True)

        for c, _ in overallocated:
            if overshoot <= 0:
                break

            if budgets[c] > min_samples_per_class:
                budgets[c] -= 1
                overshoot -= 1

    # Final validation
    final_total = sum(budgets.values())
    if final_total != total_samples:
        logger.warning(
            f"Budget allocation mismatch: requested {total_samples}, "
            f"allocated {final_total}. This can happen when minimum constraints "
            f"or class sizes prevent exact allocation."
        )

    return budgets


class NUCSBudgetAllocator:
    """High-level interface for NUCS budget allocation.

    Combines difficulty computation and budget allocation into a single class.
    Designed to integrate with the existing scorer infrastructure.

    Example:
        >>> allocator = NUCSBudgetAllocator(num_classes=40, aggregation="mean")
        >>> # Compute budgets from EL2N scores
        >>> budgets = allocator.allocate(
        ...     scores=el2n_scores,
        ...     labels=labels,
        ...     label_to_indices=label_to_indices,
        ...     total_samples=400,
        ... )
        >>> print(budgets)  # {0: 5, 1: 15, 2: 8, ...}
    """

    def __init__(
        self,
        num_classes: int,
        aggregation: str = "mean",
        min_samples_per_class: int = 1,
    ):
        """Initialize NUCS budget allocator.

        Args:
            num_classes: Total number of classes
            aggregation: Difficulty aggregation method ("mean", "median", "p75")
            min_samples_per_class: Minimum samples per class
        """
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.min_samples_per_class = min_samples_per_class

    def compute_difficulties(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[int, float]:
        """Compute per-class difficulties from scores.

        Args:
            scores: [N] difficulty scores
            labels: [N] class labels

        Returns:
            Dict of class difficulties
        """
        return compute_class_difficulties(
            scores=scores,
            labels=labels,
            num_classes=self.num_classes,
            aggregation=self.aggregation,
        )

    def allocate(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        label_to_indices: Dict[int, List[int]],
        total_samples: int,
    ) -> Dict[int, int]:
        """Compute difficulties and allocate budgets in one step.

        Args:
            scores: [N] difficulty scores (e.g., EL2N)
            labels: [N] class labels
            label_to_indices: Dict mapping class to dataset indices
            total_samples: Total samples to allocate

        Returns:
            Dict mapping class to integer budget
        """
        difficulties = self.compute_difficulties(scores, labels)

        logger.info("NUCS class difficulties:")
        for c in sorted(difficulties.keys()):
            logger.info(f"  Class {c}: {difficulties[c]:.4f}")

        budgets = compute_nucs_budgets(
            total_samples=total_samples,
            class_difficulties=difficulties,
            label_to_indices=label_to_indices,
            min_samples_per_class=self.min_samples_per_class,
        )

        logger.info("NUCS budget allocation:")
        for c in sorted(budgets.keys()):
            class_size = len(label_to_indices.get(c, []))
            logger.info(
                f"  Class {c}: {budgets[c]} / {class_size} "
                f"({100*budgets[c]/class_size:.1f}%)"
            )

        total_allocated = sum(budgets.values())
        logger.info(f"Total allocated: {total_allocated} / {total_samples} requested")

        return budgets

    def allocate_from_difficulties(
        self,
        class_difficulties: Dict[int, float],
        label_to_indices: Dict[int, List[int]],
        total_samples: int,
    ) -> Dict[int, int]:
        """Allocate budgets from pre-computed difficulties.

        Useful when difficulties are computed externally or cached.

        Args:
            class_difficulties: Dict mapping class to difficulty score
            label_to_indices: Dict mapping class to dataset indices
            total_samples: Total samples to allocate

        Returns:
            Dict mapping class to integer budget
        """
        return compute_nucs_budgets(
            total_samples=total_samples,
            class_difficulties=class_difficulties,
            label_to_indices=label_to_indices,
            min_samples_per_class=self.min_samples_per_class,
        )
