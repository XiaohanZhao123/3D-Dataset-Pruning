"""DRoP: Distributionally Robust Data Pruning - Per-class Quota Allocation.

This module wraps the DRoP quota allocation algorithm from:
"DRoP: Distributionally Robust Data Pruning" (Vysogorets et al., ICLR 2025)

The core idea is to allocate more samples to difficult classes (lower validation
performance) during pruning, which improves worst-case class accuracy.

Key formula:
    class_scores[k] = 1 - class_metrics[k]  (difficulty score)
    quota[k] = class_scores[k] / sum(class_scores)

Usage:
    from pruning.baseline.drop_quota import compute_drop_quota, compute_per_class_metrics

    # Compute per-class validation metrics (recall, accuracy, etc.)
    class_metrics = compute_per_class_metrics(model, val_loader, cfg, device, num_classes)

    # Compute DRoP quota allocation
    quotas = compute_drop_quota(
        class_metrics=class_metrics,
        class_sizes=class_sizes,  # Number of samples per class in training set
        total_samples=1000,
        num_classes=40,
        metric="recall",  # or "accuracy", "f1-score"
    )

    # quotas[k] is the fraction of total_samples to select from class k
    # sum(quotas) == 1.0
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_prep import prepare_batch

# Add third_party to path for importing drop_data_pruning.
_third_party_path = Path(__file__).parent.parent.parent.parent / "third_party"
if str(_third_party_path) not in sys.path:
    sys.path.insert(0, str(_third_party_path))

# Import DRoP's core utilities directly

logger = logging.getLogger(__name__)


def compute_per_class_metrics(
    model: nn.Module,
    dataloader,
    cfg,
    device: torch.device,
    num_classes: int,
) -> Dict[str, List[float]]:
    """Compute per-class validation metrics for DRoP quota allocation.

    Args:
        model: Model to evaluate
        dataloader: Validation/holdout dataloader
        cfg: OpenPoint config (cfg.openpoint from merged config)
        device: Device for computation
        num_classes: Total number of classes

    Returns:
        Dict with keys: 'recall', 'precision', 'f1-score', 'accuracy'
        Each value is a list of length num_classes with per-class metrics.
    """
    model.eval()
    npoints = cfg.num_points

    # Collect predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Computing per-class metrics")
        for data in pbar:
            # Use prepare_batch for proper data handling (handles channel mismatch)
            data, target = prepare_batch(
                data, cfg, device, npoints=npoints, resample=False, truncate=True
            )

            # Forward pass: Point-MAE has group_divider and expects tensor [B, N, 3]
            # PointNeXt/PointMLP has encoder and expects dict with 'pos' and 'x'
            actual_model = model.module if hasattr(model, "module") else model
            if hasattr(actual_model, "group_divider"):
                logits = model(data["pos"])
            else:
                logits = model(data)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute per-class metrics
    metrics = {
        "recall": [],
        "precision": [],
        "f1-score": [],
        "accuracy": [],
    }

    for k in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((all_preds == k) & (all_labels == k)).sum().item()
        fp = ((all_preds == k) & (all_labels != k)).sum().item()
        fn = ((all_preds != k) & (all_labels == k)).sum().item()
        tn = ((all_preds != k) & (all_labels != k)).sum().item()

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # F1 = 2 * P * R / (P + R)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        # Per-class accuracy (correct predictions for this class)
        accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall

        metrics["recall"].append(recall)
        metrics["precision"].append(precision)
        metrics["f1-score"].append(f1)
        metrics["accuracy"].append(accuracy)

    logger.info(
        f"Per-class metrics computed: mean_recall={np.mean(metrics['recall']):.4f}"
    )
    return metrics


def _drop_quota_algorithm(
    class_metrics: List[float],
    class_sizes: List[int],
    select_size: int,
    num_classes: int,
) -> List[float]:
    """Core DRoP quota algorithm - directly adapted from drop_data_pruning/quoters/drop.py.

    This is the exact algorithm from the DRoP paper (lines 16-42 of drop.py).

    Args:
        class_metrics: Per-class metric values (e.g., recall)
        class_sizes: Number of available samples per class
        select_size: Total number of samples to select
        num_classes: Total number of classes

    Returns:
        List of quotas (fractions) summing to ~1.0
    """
    # DRoP algorithm: lines 16-42 of third_party/drop_data_pruning/quoters/drop.py
    class_scores = [1 - c for c in class_metrics]
    Z = sum(class_scores)

    # Handle edge case
    if Z == 0:
        return [1.0 / num_classes] * num_classes

    select_sizes = [int(select_size * c / Z) for c in class_scores]

    error = 0
    active_classes = []
    for k in range(num_classes):
        if class_sizes[k] < select_sizes[k]:
            error += select_sizes[k] - class_sizes[k]
            select_sizes[k] = class_sizes[k]
        else:
            active_classes.append(k)

    is_ok = False
    while error > 0 and not is_ok:
        Z = sum([class_scores[k] for k in active_classes])
        if Z == 0:
            break
        to_add = {k: int(error * class_scores[k] / Z) for k in active_classes}
        is_ok = True
        for k in active_classes:
            select_sizes[k] += to_add[k]
            error -= to_add[k]
            if class_sizes[k] < select_sizes[k]:
                is_ok = False
                error += select_sizes[k] - class_sizes[k]
                select_sizes[k] = class_sizes[k]
                active_classes.remove(k)

    class_quota = [select_sizes[k] / select_size for k in range(num_classes)]
    return class_quota


def compute_drop_quota(
    class_metrics: Union[Dict[str, List[float]], List[float]],
    class_sizes: List[int],
    total_samples: int,
    num_classes: int,
    metric: str = "recall",
) -> List[float]:
    """Compute DRoP per-class quota allocation.

    Wrapper around the core DRoP algorithm implemented in the external
    drop_data_pruning package.

    Args:
        class_metrics: Either:
            - Dict with 'recall', 'accuracy', etc. keys (from compute_per_class_metrics)
            - List of floats (per-class metric values directly)
        class_sizes: Number of available samples per class in training set
        total_samples: Total number of samples to select
        num_classes: Total number of classes
        metric: Which metric to use for difficulty estimation ('recall', 'accuracy', 'f1-score')

    Returns:
        List of quotas, where quota[k] is the fraction of total_samples to select
        from class k. sum(quotas) ~= 1.0

    Raises:
        ValueError: If class_metrics format is invalid
    """
    # Extract metric values
    if isinstance(class_metrics, dict):
        if metric not in class_metrics:
            raise ValueError(
                f"Metric '{metric}' not in class_metrics. Available: {list(class_metrics.keys())}"
            )
        metric_values = class_metrics[metric]
    else:
        metric_values = class_metrics

    if len(metric_values) != num_classes:
        raise ValueError(
            f"Expected {num_classes} metric values, got {len(metric_values)}"
        )

    # Call core DRoP algorithm
    class_quota = _drop_quota_algorithm(
        class_metrics=list(metric_values),
        class_sizes=list(class_sizes),
        select_size=total_samples,
        num_classes=num_classes,
    )

    # Log quota distribution
    class_scores = [1 - m for m in metric_values]
    logger.info(f"DRoP quota allocation (metric={metric}):")
    for k in range(num_classes):
        actual_samples = int(class_quota[k] * total_samples)
        logger.info(
            f"  Class {k}: metric={metric_values[k]:.4f}, difficulty={class_scores[k]:.4f}, "
            f"quota={class_quota[k]:.4f}, samples={actual_samples}/{class_sizes[k]}"
        )

    return class_quota


def get_class_sizes(dataset, num_classes: int) -> List[int]:
    """Get number of samples per class in a dataset.

    Args:
        dataset: PyTorch dataset with 'y' label
        num_classes: Total number of classes

    Returns:
        List of class sizes
    """
    class_sizes = [0] * num_classes

    for i in range(len(dataset)):
        sample = dataset[i]
        # Handle different label formats
        if isinstance(sample, dict):
            label = sample["y"]
        elif isinstance(sample, (tuple, list)):
            label = sample[1]
        else:
            label = sample.y

        if isinstance(label, torch.Tensor):
            label = label.item()

        class_sizes[label] += 1

    return class_sizes


def select_with_drop_quota(
    scores: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    quotas: List[float],
    total_samples: int,
    num_classes: int,
    mode: str = "random",
) -> List[int]:
    """Select samples based on DRoP quota allocation.

    Args:
        scores: [N] tensor of scores (unused if mode='random')
        labels: [N] tensor of ground truth labels
        indices: [N] tensor of original dataset indices
        quotas: Per-class quota allocation from compute_drop_quota()
        total_samples: Total number of samples to select
        num_classes: Total number of classes
        mode: Selection mode within each class:
            - 'random': Random selection (DRoP default)
            - 'min': Select lowest score samples
            - 'max': Select highest score samples

    Returns:
        List of selected dataset indices
    """
    selected = []

    for k in range(num_classes):
        class_mask = labels == k
        class_positions = torch.where(class_mask)[0]

        if len(class_positions) == 0:
            continue

        # Compute number of samples for this class based on quota
        n_select = int(quotas[k] * total_samples)
        n_select = min(n_select, len(class_positions))

        if n_select == 0:
            continue

        if mode == "random":
            # Random selection within class
            perm = torch.randperm(len(class_positions))[:n_select]
            selected_positions = class_positions[perm]
        elif mode == "min":
            # Select lowest scores within class
            class_scores = scores[class_mask]
            sorted_local = torch.argsort(class_scores)[:n_select]
            selected_positions = class_positions[sorted_local]
        elif mode == "max":
            # Select highest scores within class
            class_scores = scores[class_mask]
            sorted_local = torch.argsort(class_scores, descending=True)[:n_select]
            selected_positions = class_positions[sorted_local]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        selected.extend(indices[selected_positions].cpu().numpy().tolist())

    logger.info(f"DRoP selection: {len(selected)} samples (mode={mode})")
    return selected
