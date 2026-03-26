"""Metrics utilities for head/medium/tail class analysis."""

from collections import Counter

import numpy as np

from utils.samplers import get_labels_from_dataset


def compute_class_counts(dataset, num_classes):
    """Count samples per class in dataset.

    Args:
        dataset: Dataset or Subset with labels
        num_classes: Total number of classes

    Returns:
        np.ndarray of shape (num_classes,) with sample counts
    """
    labels = get_labels_from_dataset(dataset)
    label_counts = Counter(labels)
    return np.array([label_counts[i] for i in range(num_classes)])


def define_hmt_classes(class_counts, head_ratio=0.33, tail_ratio=0.33):
    """Split classes into head/medium/tail by frequency.

    Classes are sorted by sample count (descending):
    - Head: top head_ratio (most frequent)
    - Tail: bottom tail_ratio (least frequent)
    - Medium: remaining classes

    Args:
        class_counts: Array of sample counts per class
        head_ratio: Fraction of classes to consider as head (default 0.33)
        tail_ratio: Fraction of classes to consider as tail (default 0.33)

    Returns:
        Dict with 'head', 'medium', 'tail' keys, each containing a set of class indices
    """
    sorted_indices = np.argsort(class_counts)[::-1]  # descending by count
    n = len(class_counts)
    n_head = int(n * head_ratio)
    n_tail = int(n * tail_ratio)

    head_set = set(sorted_indices[:n_head].tolist())
    tail_set = set(sorted_indices[-n_tail:].tolist()) if n_tail > 0 else set()
    medium_set = set(range(n)) - head_set - tail_set

    return {
        "head": head_set,
        "medium": medium_set,
        "tail": tail_set,
    }


def compute_group_accuracies(per_class_accs, hmt_classes):
    """Compute mean accuracy for each HMT group.

    Args:
        per_class_accs: Array/list of per-class accuracies
        hmt_classes: Dict from define_hmt_classes()

    Returns:
        Dict with 'head', 'medium', 'tail' mean accuracies
    """
    result = {}
    for group_name, class_indices in hmt_classes.items():
        if class_indices:
            group_accs = [per_class_accs[i] for i in class_indices]
            result[group_name] = float(np.mean(group_accs))
        else:
            result[group_name] = 0.0
    return result


def build_hmt_wandb_log(per_class_accs, original_hmt, pruned_hmt=None):
    """Build wandb log dict for HMT metrics.

    Args:
        per_class_accs: Array/list of per-class accuracies
        original_hmt: HMT grouping based on original dataset distribution
        pruned_hmt: HMT grouping based on pruned dataset (None if per_class=True)

    Returns:
        Dict ready for wandb.log()
    """
    log = {}

    # Per-class accuracies
    for i, acc in enumerate(per_class_accs):
        log[f"val/class_acc/{i}"] = float(acc)

    # Original distribution grouping (always logged)
    orig_group_accs = compute_group_accuracies(per_class_accs, original_hmt)
    log["val/head_acc_original"] = orig_group_accs["head"]
    log["val/medium_acc_original"] = orig_group_accs["medium"]
    log["val/tail_acc_original"] = orig_group_accs["tail"]

    # Pruned distribution grouping (only for global pruning)
    if pruned_hmt is not None:
        pruned_group_accs = compute_group_accuracies(per_class_accs, pruned_hmt)
        log["val/head_acc_pruned"] = pruned_group_accs["head"]
        log["val/medium_acc_pruned"] = pruned_group_accs["medium"]
        log["val/tail_acc_pruned"] = pruned_group_accs["tail"]

    return log
