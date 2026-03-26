"""
Naive single level data pruners, does not consider the subsequent KL process
"""

import logging
from typing import Literal, Callable, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

from pruning.functional import (
    _random_selection,
    get_all_samples,
    select_by_score,
    get_label_samples_map,
    get_label_indices_map,
    compute_class_budgets,
)

logger = logging.getLogger(__name__)


class ScoreBasedDataPruner:
    """Simple wrapper class for data is_pruning using scoring and selection functions.

    This class composes scoring functions (from is_pruning.scorers) with selection functions
    (from is_pruning.selection) to provide a unified interface for data is_pruning.
    """

    def __init__(
        self, score_fn: Optional[Callable[[Dict[str, Any]], torch.Tensor]] = None
    ):
        """Initialize ScoreBasedDataPruner with an optional scoring function.

        Args:
            score_fn: Optional scoring function that maps samples to scores.
                     If None, random selection will be used.
        """
        self.score_fn = score_fn

    def select(
        self,
        dataset: Dataset,
        total_samples: int,
        mode: Literal["min", "max", "random"] = "min",
        per_class: bool = True,
    ) -> torch.Tensor:
        """Select samples from dataset using the configured scoring function.

        Args:
            dataset: Dataset to select samples from
            total_samples: Total number of samples to select. When per_class=True,
                          this is divided evenly among classes (must be divisible).
            mode: Selection mode - "min" for lowest scores, "max" for highest, "random" for random
            per_class: If True, divide total_samples evenly among classes; if False, select globally

        Returns:
            Selected indices as 1D torch.Tensor

        Raises:
            ValueError: If mode is invalid, or if per_class=True and total_samples
                       is not divisible by number of classes
        """
        # Handle random selection case
        if mode not in ["min", "max", "random"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'min', 'max', or 'random'."
            )

        if self.score_fn is None or mode == "random":
            return _random_selection(dataset, total_samples, per_class)

        if not per_class:
            # Global selection logic
            logger.info(f"Global selection of {total_samples} samples")
            all_samples = get_all_samples(dataset)
            all_scores = self.score_fn(all_samples)
            selected_indices = select_by_score(all_scores, total_samples, mode)

            return selected_indices

        # Per-class selection with budget redistribution
        label_to_data = get_label_samples_map(dataset)
        label_to_indices = {label: data["indices"] for label, data in label_to_data.items()}
        
        # Compute per-class budgets with redistribution
        budgets = compute_class_budgets(total_samples, label_to_indices, per_class=True)
        
        all_selected = []

        # Determine collate function based on data format
        first_sample = dataset[0]
        if isinstance(first_sample, dict):
            # PointNeXt: use PyTorch default collate
            from torch.utils.data.dataloader import default_collate
            collate_fn = default_collate
        else:
            # PyG: use PyG Collater
            collater = Collater(dataset=dataset, follow_batch=[], exclude_keys=[])
            collate_fn = collater

        logger.info(f"Selecting {total_samples} total samples across {len(label_to_data)} classes using mode '{mode}'")
        for class_label, data in tqdm(
            label_to_data.items(), desc="Processing classes", unit="class"
        ):
            class_samples = data["samples"]
            class_indices = data["indices"]
            class_budget = budgets[class_label]

            if class_budget == 0:
                continue

            batched_class_samples = collate_fn(class_samples)

            scores = self.score_fn(batched_class_samples)
            effective_num = min(class_budget, len(scores))
            selected_positions = select_by_score(scores, effective_num, mode)

            selected_global_indices = [
                class_indices[i.item()] for i in selected_positions
            ]
            all_selected.extend(selected_global_indices)

        return torch.tensor(all_selected, dtype=torch.long)

    def prune_dataset(
        self,
        dataset: Dataset,
        total_samples: int,
        mode: Literal["min", "max", "random"] = "min",
        per_class: bool = True,
    ) -> Subset:
        """Prune dataset and return a PyTorch Subset with selected samples.

        Args:
            dataset: Dataset to prune
            total_samples: Total number of samples to select. When per_class=True,
                          this is divided evenly among classes (must be divisible).
            mode: Selection mode - "min" for lowest scores, "max" for highest, "random" for random
            per_class: If True, divide total_samples evenly among classes; if False, select globally

        Returns:
            PyTorch Subset containing only the selected samples
        """
        selected_indices = self.select(dataset, total_samples, mode, per_class)
        return Subset(dataset, selected_indices.tolist())

    def get_selected_samples(
        self,
        dataset: Dataset,
        total_samples: int,
        mode: Literal["min", "max", "random"] = "min",
        per_class: bool = True,
    ) -> tuple:
        """Get both selected indices and sample data.

        Args:
            dataset: Dataset to select samples from
            total_samples: Total number of samples to select. When per_class=True,
                          this is divided evenly among classes (must be divisible).
            mode: Selection mode - "min" for lowest scores, "max" for highest, "random" for random
            per_class: If True, divide total_samples evenly among classes; if False, select globally

        Returns:
            Tuple of (selected_indices, selected_samples_batched)
            - selected_indices: 1D torch.Tensor of selected indices
            - selected_samples_batched: torch_geometric.Data object containing batched sample data
        """
        selected_indices = self.select(dataset, total_samples, mode, per_class)

        # Get the actual samples using the selected indices
        selected_samples_list = [dataset[idx.item()] for idx in selected_indices]

        # Use appropriate collate function based on format
        first_sample = selected_samples_list[0] if selected_samples_list else dataset[0]

        if isinstance(first_sample, dict):
            # PointNeXt: use PyTorch default collate
            from torch.utils.data.dataloader import default_collate
            selected_samples_batched = default_collate(selected_samples_list)
        else:
            # PyG: use PyG Collater
            collater = Collater(dataset=dataset, follow_batch=[], exclude_keys=[])
            selected_samples_batched = collater(selected_samples_list)

        return selected_indices, selected_samples_batched
