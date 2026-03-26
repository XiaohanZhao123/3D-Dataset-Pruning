"""Pure selection functions for data is_pruning.

This module contains pure functions that convert scores/features to selected indices.
These functions have no side effects and can be easily composed and tested.
"""

from functools import lru_cache
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.loader.dataloader import Collater

# =============================================================================
# CCS (Coverage-centric Coreset Selection) Functions
# Adapted from Coverage-centric Coreset Selection (CCS).
# =============================================================================


def _ccs_bin_allocate(total_budget: int, bin_counts: torch.Tensor) -> torch.Tensor:
    """
    Allocate budget across bins proportionally, capped at bin population.

    Algorithm (from CCS paper):
    1. Sort bins by size (smallest first)
    2. For each bin: allocate min(bin_size, avg_remaining_budget)
    3. Redistribute unused budget to remaining bins

    Args:
        total_budget: Total number of samples to select
        bin_counts: Number of samples in each bin [num_bins]

    Returns:
        Budget allocation for each bin [num_bins]
    """
    sorted_indices = torch.argsort(bin_counts)
    sorted_counts = bin_counts[sorted_indices]

    num_bins = bin_counts.shape[0]
    remaining_budget = total_budget
    budgets = []

    for i in range(num_bins):
        remaining_bins = num_bins - i
        avg_budget = remaining_budget // remaining_bins
        # Allocate min(bin_size, average_budget)
        allocated = min(sorted_counts[i].item(), avg_budget)
        budgets.append(allocated)
        remaining_budget -= allocated

    # Map back to original bin order
    result = torch.zeros(num_bins, dtype=torch.int)
    result[sorted_indices] = torch.tensor(budgets, dtype=torch.int)

    return result


def _ccs_stratified_sampling(
    scores: torch.Tensor,
    total_budget: int,
    num_strata: int = 50,
) -> List[int]:
    """
    CCS stratified sampling: select samples across score distribution bins.

    Algorithm (from CCS paper):
    1. Divide score range into equal-width bins (default 50)
    2. Count samples in each bin
    3. Allocate budget proportionally (capped at bin size)
    4. Randomly sample within each bin

    Args:
        scores: Per-sample scores [num_samples]
        total_budget: Total number of samples to select
        num_strata: Number of bins (default 50, as in CCS paper)

    Returns:
        List of selected sample indices
    """
    min_score = torch.min(scores)
    max_score = torch.max(scores) * 1.0001  # Ensure max is included
    step = (max_score - min_score) / num_strata

    def bin_range(k):
        return min_score + k * step, min_score + (k + 1) * step

    # Count samples in each stratum
    strata_counts = []
    for i in range(num_strata):
        start, end = bin_range(i)
        count = torch.logical_and(scores >= start, scores < end).sum()
        strata_counts.append(count)

    strata_counts = torch.tensor(strata_counts)

    # Allocate budget across strata
    budgets = _ccs_bin_allocate(total_budget, strata_counts)

    # Sample from each stratum
    selected_indices = []
    sample_indices = torch.arange(len(scores))

    for i in range(num_strata):
        start, end = bin_range(i)
        mask = torch.logical_and(scores >= start, scores < end)
        pool = sample_indices[mask]

        if len(pool) > 0 and budgets[i] > 0:
            # Random sampling within bin
            rand_perm = torch.randperm(len(pool))
            selected = pool[rand_perm[: budgets[i]]]
            selected_indices.extend(selected.tolist())

    return selected_indices


def _ccs_mislabel_mask(
    scores: torch.Tensor,
    num_to_remove: int,
    remove_low: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove hard/mislabeled samples before coreset selection.

    Args:
        scores: Per-sample scores [num_samples]
        num_to_remove: Number of samples to remove
        remove_low: If True, remove low scores (hard samples); else remove high

    Returns:
        Tuple of (filtered_scores, kept_indices)
    """
    if num_to_remove <= 0:
        return scores, torch.arange(len(scores))

    sorted_indices = torch.argsort(scores, descending=not remove_low)

    # Remove first num_to_remove samples (the hard/mislabeled ones)
    kept_indices = sorted_indices[num_to_remove:]
    filtered_scores = scores[kept_indices]

    return filtered_scores, kept_indices


def select_by_ccs(
    scores: torch.Tensor,
    num_samples: int,
    mislabel_ratio: float = 0.0,
    num_strata: int = 50,
) -> torch.Tensor:
    """
    CCS selection: mislabel removal + stratified sampling.

    This implements the Coverage-centric Coreset Selection algorithm:
    1. Remove hard/mislabeled samples (low scores) based on mislabel_ratio
    2. Apply stratified sampling on remaining samples

    Args:
        scores: Per-sample scores [num_samples]. Higher scores = easier samples.
        num_samples: Total number of samples to select
        mislabel_ratio: Fraction of samples to remove as mislabeled (0-1)
        num_strata: Number of bins for stratified sampling (default 50)

    Returns:
        Selected indices as 1D torch.Tensor (in original indexing)
    """
    # Step 1: Remove hard/mislabeled samples (low scores)
    num_to_remove = int(mislabel_ratio * len(scores))
    filtered_scores, kept_indices = _ccs_mislabel_mask(
        scores, num_to_remove, remove_low=True
    )

    # Step 2: Stratified sampling on remaining samples
    local_selected = _ccs_stratified_sampling(filtered_scores, num_samples, num_strata)

    # Map back to original indices
    original_selected = [kept_indices[i].item() for i in local_selected]

    return torch.tensor(original_selected, dtype=torch.long)


def extract_label(sample: Any):
    """Return the numeric label stored in a dataset sample.

    Supports:
    - PointNeXt dict format: {'y': label, ...}
    - Dict with 'label' key: {'label': label, ...}
    - torch_geometric.Data objects: sample.y
    """
    # PointNeXt format: dict with 'y' key
    if isinstance(sample, dict):
        if "y" in sample:
            label = sample["y"]
        elif "label" in sample:
            label = sample["label"]
        else:
            raise ValueError(
                f"Dict sample missing label. Keys: {sample.keys()}. "
                f"Expected 'y' or 'label' key."
            )
    # PyG Data format
    elif hasattr(sample, "y"):
        label = sample.y
    else:
        raise ValueError(
            f"Unsupported sample format: {type(sample)}. "
            f"Expected dict with 'y'/'label' key or torch_geometric.Data with .y attribute."
        )

    # Handle tensor labels
    if isinstance(label, torch.Tensor):
        if label.numel() != 1:
            raise ValueError(
                f"Expected scalar label, got tensor with {label.numel()} elements"
            )
        return label.view(-1).item()

    # Handle numeric labels (including numpy types)
    if isinstance(label, (int, float, np.integer, np.floating)):
        return int(label)

    raise ValueError(
        f"Unsupported label type: {type(label)}. Expected scalar tensor or numeric value."
    )


@lru_cache(maxsize=128)
def get_label_indices_map(dataset):
    """Single pass to build label->indices mapping for torch_geometric.Data objects.

    This function efficiently builds a mapping from class labels to dataset indices
    in a single pass through the dataset. It primarily handles torch_geometric.Data
    objects with a .y attribute, but also supports dict format for testing.

    Args:
        dataset: Dataset to build mapping for

    Returns:
        Dict mapping class labels to lists of dataset indices
    """
    label_to_indices = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]

        label = extract_label(sample)

        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    return label_to_indices


@lru_cache(maxsize=128)
def get_label_samples_map(dataset):
    """Single pass to build label->(indices, samples) mapping for torch_geometric.Data objects.

    This function efficiently collects both indices and samples in a single pass through
    the dataset, eliminating the need for multiple dataset iterations during per-class selection.
    This achieves TRUE single-pass collection, avoiding the performance bottleneck of
    C additional dataset scans (one per class) that occurs with index-only collection.

    It primarily handles torch_geometric.Data objects with a .y attribute, but also supports
    dict format for testing.

    Args:
        dataset: Dataset to build mapping for

    Returns:
        Dict mapping class labels to dicts containing:
            - 'indices': List of dataset indices for this class
            - 'samples': List of sample objects for this class

    Raises:
        ValueError: If dataset is empty or samples have unsupported format
        TypeError: If dataset doesn't support length or indexing
    """
    if not hasattr(dataset, "__len__"):
        raise TypeError("Dataset must support len() operation")

    dataset_length = len(dataset)
    if dataset_length == 0:
        raise ValueError("Dataset is empty")

    label_to_data = {}

    for idx in range(dataset_length):
        try:
            sample = dataset[idx]
        except (IndexError, KeyError) as e:
            raise ValueError(f"Cannot access dataset at index {idx}: {e}")

        label = extract_label(sample)

        # Validate label is a valid class identifier
        if not isinstance(label, (int, float)):
            raise ValueError(
                f"Label at index {idx} must be numeric, got {type(label)}: {label}"
            )

        if label not in label_to_data:
            label_to_data[label] = {"indices": [], "samples": []}
        label_to_data[label]["indices"].append(idx)
        label_to_data[label]["samples"].append(sample)

    if not label_to_data:
        raise ValueError("No valid labeled samples found in dataset")

    return label_to_data


def compute_class_budgets(
    total_samples: int,
    label_to_indices: dict,
    per_class: bool = True,
) -> dict:
    """Compute per-class sample budgets with redistribution for insufficient classes.

    When per_class=True, divides total_samples evenly among classes. If some classes
    have fewer samples than their quota, the remainder is redistributed round-robin
    to other classes that have capacity.

    Args:
        total_samples: Total number of samples to select
        label_to_indices: Dict mapping class labels to lists of dataset indices
        per_class: If True, divide evenly among classes; if False, return total budget

    Returns:
        Dict mapping class labels to their sample budgets

    Raises:
        ValueError: If per_class=True and total_samples is not divisible by num_classes
    """
    num_classes = len(label_to_indices)

    if not per_class:
        # Global selection: no per-class budgets needed
        return {label: len(indices) for label, indices in label_to_indices.items()}

    # Validate divisibility
    if total_samples % num_classes != 0:
        raise ValueError(
            f"total_samples ({total_samples}) must be divisible by num_classes ({num_classes}) "
            f"when per_class=True. Consider using {total_samples - (total_samples % num_classes)} "
            f"or {total_samples + (num_classes - total_samples % num_classes)} instead."
        )

    base_quota = total_samples // num_classes

    # First pass: identify classes with insufficient samples
    budgets = {}
    remainder = 0
    classes_with_capacity = []

    for label, indices in label_to_indices.items():
        available = len(indices)
        if available < base_quota:
            # Take all available, track the shortfall
            budgets[label] = available
            remainder += base_quota - available
        else:
            # Has enough samples, may receive extras
            budgets[label] = base_quota
            classes_with_capacity.append(label)

    # Second pass: redistribute remainder round-robin to classes with capacity
    if remainder > 0 and classes_with_capacity:
        # Sort for deterministic ordering
        classes_with_capacity = sorted(classes_with_capacity)
        idx = 0
        while remainder > 0:
            label = classes_with_capacity[idx % len(classes_with_capacity)]
            available = len(label_to_indices[label])
            if budgets[label] < available:
                budgets[label] += 1
                remainder -= 1
            else:
                # This class is full, remove from rotation
                classes_with_capacity.remove(label)
                if not classes_with_capacity:
                    # All classes are full, can't redistribute more
                    break
                continue
            idx += 1

    return budgets


def compute_class_proportional_budgets(
    total_samples: int,
    label_to_indices: dict,
    min_samples_per_class: int = 1,
) -> dict:
    """Compute per-class budgets proportional to class size (CCS-CP algorithm).

    Implements Algorithm 1 from "Class-Proportional Coreset Selection for
    Difficulty-Separable Data" (Tsai et al., ICCV 2025 Workshop).

    Budget allocation:
        B'_c = max(floor(B * n_c / n), m)

    If total exceeds budget B, iteratively reduce from largest classes
    (those with B'_c > m) until sum <= B.

    Args:
        total_samples: Total budget B (number of samples to select)
        label_to_indices: Dict mapping class labels to lists of dataset indices
        min_samples_per_class: Minimum samples per class (m), default 1

    Returns:
        Dict mapping class labels to their sample budgets
    """
    n_total = sum(len(indices) for indices in label_to_indices.values())

    # Compute raw budget per class: B'_c = max(floor(B * n_c / n), m)
    budgets = {}
    for label, indices in label_to_indices.items():
        n_c = len(indices)
        raw_budget = max(int(total_samples * n_c / n_total), min_samples_per_class)
        # Also cap at actual class size
        budgets[label] = min(raw_budget, n_c)

    # If sum exceeds total budget, reduce from largest classes
    current_sum = sum(budgets.values())

    if current_sum > total_samples:
        # Sort classes by size in descending order
        sorted_classes = sorted(
            label_to_indices.keys(),
            key=lambda c: len(label_to_indices[c]),
            reverse=True,
        )

        while current_sum > total_samples:
            reduced = False
            for label in sorted_classes:
                if budgets[label] > min_samples_per_class:
                    budgets[label] -= 1
                    current_sum -= 1
                    reduced = True
                    if current_sum <= total_samples:
                        break
            if not reduced:
                # All classes are at minimum, cannot reduce further
                break

    # If sum is less than total budget (due to small classes), redistribute
    # to larger classes that have capacity
    if current_sum < total_samples:
        remainder = total_samples - current_sum
        # Sort by size descending, give extra to larger classes first
        sorted_classes = sorted(
            label_to_indices.keys(),
            key=lambda c: len(label_to_indices[c]),
            reverse=True,
        )

        idx = 0
        while remainder > 0 and sorted_classes:
            label = sorted_classes[idx % len(sorted_classes)]
            available = len(label_to_indices[label])
            if budgets[label] < available:
                budgets[label] += 1
                remainder -= 1
                idx += 1
            else:
                # This class is full, remove from rotation
                sorted_classes.remove(label)
                if sorted_classes:
                    idx = idx % len(sorted_classes)

    return budgets


def select_by_score(
    scores: torch.Tensor,
    num_samples: int,
    mode: Literal["min", "max", "random", "ccs"] = "min",
    mislabel_ratio: float = 0.0,
    num_strata: int = 50,
) -> torch.Tensor:
    """Select indices based on score ranking.

    Pure function that takes scores and returns selected indices.

    Args:
        scores: 1D tensor of scores for each sample
        num_samples: Number of samples to select
        mode: Selection mode:
            - "min": Select lowest scores (easy samples)
            - "max": Select highest scores (hard samples)
            - "random": Random selection
            - "ccs": CCS stratified sampling (coverage across score distribution)
        mislabel_ratio: (CCS only) Fraction of hard samples to remove before selection
        num_strata: (CCS only) Number of bins for stratified sampling (default 50)

    Returns:
        Selected indices as 1D torch.Tensor
    """
    if mode == "random":
        indices = torch.randperm(len(scores))[:num_samples]
        return indices

    if mode == "ccs":
        return select_by_ccs(scores, num_samples, mislabel_ratio, num_strata)

    _, sorted_indices = torch.sort(scores)

    if mode == "min":
        selected_indices = sorted_indices[:num_samples]
    elif mode == "max":
        selected_indices = sorted_indices[-num_samples:]
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported: 'min', 'max', 'random', 'ccs'"
        )

    return selected_indices


def _random_selection(
    dataset, total_samples: int, per_class: bool = True
) -> torch.Tensor:
    """Helper function for random baseline selection.

    Args:
        dataset: Dataset object
        total_samples: Total number of samples to select. When per_class=True,
                      this is divided evenly among classes (must be divisible).
        per_class: If True, divide total_samples evenly among classes; if False, select globally

    Returns:
        Randomly selected indices

    Raises:
        ValueError: If per_class=True and total_samples is not divisible by num_classes
    """
    if not per_class:
        # Global random selection
        dataset_size = len(dataset)
        effective_num = min(total_samples, dataset_size)
        selected_indices = torch.randperm(dataset_size)[:effective_num]
        return selected_indices

    # Per-class random selection with budget redistribution
    label_to_indices = get_label_indices_map(dataset)

    # Compute per-class budgets with redistribution
    budgets = compute_class_budgets(total_samples, label_to_indices, per_class=True)

    all_selected = []

    for class_label, class_indices in label_to_indices.items():
        class_budget = budgets[class_label]
        if class_budget == 0:
            continue

        effective_num = min(class_budget, len(class_indices))
        if effective_num > 0:
            selected_positions = torch.randperm(len(class_indices))[:effective_num]
            selected_global_indices = [
                class_indices[i.item()] for i in selected_positions
            ]
            all_selected.extend(selected_global_indices)

    return torch.tensor(all_selected, dtype=torch.long)


def get_all_samples(dataset: Dataset):
    """Get all samples from dataset as a batched dict or Data object.

    Uses PyTorch's default collate for dict format (PointNeXt),
    or PyG Collater for Data objects.

    Args:
        dataset: Dataset to get all samples from

    Returns:
        Batched dict (PointNeXt) or torch_geometric.Data object (PyG)
    """
    all_samples = []
    for sample in dataset:
        all_samples.append(sample)

    if not all_samples:
        return {}

    # Check format and use appropriate collate
    first_sample = all_samples[0]
    if isinstance(first_sample, dict):
        # PointNeXt format - use PyTorch default collate
        from torch.utils.data.dataloader import default_collate

        return default_collate(all_samples)
    else:
        # PyG format - use PyG Collater
        collater = Collater(dataset=dataset, follow_batch=[], exclude_keys=[])
        return collater(all_samples)


def kd_soft_ce(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float = 4.0,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Standard KD soft cross-entropy with temperature and T^2 scaling.

    Both logits are divided by tau, CE(q_t, p_s) is computed, and multiplied by tau**2
    to preserve gradient magnitudes (Hinton et al.). Computed in float32 for stability.
    """
    tau = float(tau)
    t_logits = (teacher_logits.detach() / tau).float()
    s_logits = (student_logits / tau).float()
    q = F.softmax(t_logits, dim=-1)
    log_p = F.log_softmax(s_logits, dim=-1)

    # Per-sample loss (sum over class dim only)
    loss = -(q * log_p).sum(dim=-1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "none":
        pass  # return per-sample loss (and any extra dims like augments)
    else:
        raise ValueError(
            f"Unsupported reduction: {reduction}. Supported: 'mean', 'none'"
        )

    return loss * (tau * tau)


def teacher_topk_ce_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    temperature: float = 2.0,
    gate_temperature: float = 1.0,
    gate_threshold: float = 0.7,
    confidence_power: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute teacher-centric Top-K CE divergence with confidence gating.

    This method focuses divergence computation on the teacher's Top-K predicted classes
    and collapses the remaining classes into a single "OTHER" bin. Only samples where
    the teacher is confident on the true label (p_T(y) >= gate_threshold) are scored.

    The key idea: measure student-teacher mismatch on bins where the teacher has
    meaningful signal (its Top-K), while ignoring noisy/uncertain predictions.

    IMPORTANT: Gating and divergence use SEPARATE temperatures:
    - gate_temperature (default=1.0): For assessing teacher's TRUE confidence
    - temperature (default=2.0): For softening distributions in divergence computation

    Formula:
        # Gating uses RAW probabilities (gate_temperature, typically 1.0)
        pT_gate = softmax(logits / gate_temperature)
        gate(x) = 1[pT_gate(y) >= α]

        # Divergence uses SOFTENED probabilities (temperature, typically 2.0)
        pT = softmax(logits / temperature)
        pS = softmax(logits / temperature)
        J_T = TopK(pT, K)  # teacher's top-K classes
        p_T^oth = 1 - Σ_{k∈J_T} p_T(k)  # tail mass
        p_S^oth = 1 - Σ_{k∈J_T} p_S(k)

        D_K(T||S) = -Σ_{k∈J_T} p_T(k) log(p_S(k) + ε) - p_T^oth log(p_S^oth + ε)
        score = gate(x) * [pT_gate(y)]^a * D_K(T||S)

    Args:
        teacher_logits: Teacher model logits of shape (N, C)
        student_logits: Student model logits of shape (N, C)
        labels: Ground truth labels of shape (N,)
        k: Number of top classes to focus on from teacher predictions
        temperature: Temperature for divergence computation (τ_div), typical range [1, 2]
            Higher values soften distributions, making divergence smoother
        gate_temperature: Temperature for confidence gating (τ_gate), typically 1.0
            Use 1.0 to assess teacher's TRUE confidence without softening
            Use >1.0 only if you have a calibrated "trustworthy" temperature
        gate_threshold: Minimum teacher confidence on true label (α ∈ [0.6, 0.8])
            Only score samples where teacher is confident on ground truth
            With gate_temperature=1.0, typical values: 0.6-0.8
            With gate_temperature=2.0, adjust down: 0.3-0.5
        confidence_power: Power to raise teacher confidence p_T(y) (a), typically 1.0
        eps: Small constant for numerical stability (1e-8)

    Returns:
        Per-sample divergence scores of shape (N,). Higher scores indicate greater
        teacher-student divergence on important bins where teacher is confident.
        Gated samples (where teacher is uncertain) have score=0.

    Example:
        >>> teacher_logits = torch.randn(32, 10)  # 32 samples, 10 classes
        >>> student_logits = torch.randn(32, 10)
        >>> labels = torch.randint(0, 10, (32,))

        # Standard usage: separate temperatures for gating and divergence
        >>> scores = teacher_topk_ce_divergence(
        ...     teacher_logits, student_logits, labels,
        ...     k=5,
        ...     temperature=2.0,        # Soft divergence
        ...     gate_temperature=1.0,   # True confidence gating
        ...     gate_threshold=0.7
        ... )
        >>> scores.shape
        torch.Size([32])

    Notes:
        - Defaults: K ∈ {3,5,8}, τ_div ∈ [1,2], τ_gate=1.0, α=0.7, a=1.0
        - Teacher-centric: no symmetric term (teacher is ground truth)
        - Asymmetric by design: penalizes student for not matching teacher's important bins
        - Separation of concerns: gating checks TRUE confidence, divergence uses SOFT distributions
    """
    N = teacher_logits.size(0)
    ar = torch.arange(N, device=teacher_logits.device)

    # Gating: use gate_temperature (typically 1.0) for TRUE confidence
    pT_gate = F.softmax(teacher_logits / gate_temperature, dim=-1)
    gate = (pT_gate[ar, labels] >= gate_threshold).float()

    # Divergence: use temperature (typically 2.0) for SOFT distributions
    pT = F.softmax(teacher_logits / temperature, dim=-1)
    pS = F.softmax(student_logits / temperature, dim=-1)

    # Teacher's Top-K classes (where teacher has meaningful signal)
    topk_idx = pT.topk(k, dim=-1).indices  # [N, K]

    # Gather probabilities for Top-K bins
    pT_K = pT.gather(1, topk_idx)  # [N, K]
    pS_K = pS.gather(1, topk_idx)  # [N, K]

    # Tail mass collapsed into OTHER bin
    pT_oth = (1 - pT_K.sum(-1)).clamp_min(0.0)  # [N]
    pS_oth = (1 - pS_K.sum(-1)).clamp_min(0.0)  # [N]

    # Cross-entropy: Top-K bins + OTHER bin
    ce_topk = -(pT_K * (pS_K.add(eps).log())).sum(-1)  # [N]
    ce_oth = -pT_oth * (pS_oth.add(eps).log())  # [N]
    ce = ce_topk + ce_oth

    # Final score: gate * [teacher_confidence]^a * divergence
    # Use gated confidence (from gate_temperature) for weighting
    teacher_confidence = pT_gate[ar, labels].clamp_min(eps)
    score = gate * (teacher_confidence**confidence_power) * ce

    return score


# =============================================================================
# Hybrid Selection Functions
# =============================================================================


def hybrid_select_score_based(
    scores: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    total_samples: int,
    hybrid_per_class_ratio: float,
    mode: str,
    num_classes: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Hybrid selection for score-based scorers.

    Phase 1: Per-class selection with floor division (balanced)
    Phase 2: Global selection from remaining samples (no overlap)

    Args:
        scores: [N] tensor of per-sample scores
        labels: [N] tensor of ground truth labels
        indices: [N] tensor of original dataset indices
        total_samples: Total number of samples to select
        hybrid_per_class_ratio: Fraction of total_samples for per-class phase (0-1)
        mode: Selection mode ("min" for easy, "max" for hard)
        num_classes: Number of classes

    Returns:
        Tuple of (phase1_indices, phase2_indices, all_selected_indices)
    """
    import logging
    import math

    logger = logging.getLogger(__name__)

    # Phase 1 budget: floor to ensure divisibility
    phase1_per_class = math.floor(hybrid_per_class_ratio * total_samples / num_classes)
    phase1_budget = phase1_per_class * num_classes
    phase2_budget = total_samples - phase1_budget

    logger.info(
        f"Hybrid Phase 1: {phase1_per_class} per class × {num_classes} = {phase1_budget}"
    )
    logger.info(f"Hybrid Phase 2: {phase2_budget} global")

    # Phase 1: Per-class selection
    phase1_indices = []
    for c in range(num_classes):
        class_mask = labels == c
        class_scores = scores[class_mask]
        class_indices = indices[class_mask]

        if len(class_indices) == 0:
            continue

        if len(class_indices) < phase1_per_class:
            logger.warning(
                f"Class {c} has only {len(class_indices)} samples, need {phase1_per_class}"
            )
            selected_positions = torch.arange(len(class_indices))
        else:
            # Sort by score
            sorted_positions = torch.argsort(class_scores)
            if mode == "min":
                selected_positions = sorted_positions[:phase1_per_class]
            elif mode == "max":
                selected_positions = sorted_positions[-phase1_per_class:]
            elif mode == "mid":
                # Middle selection
                n = len(sorted_positions)
                start = (n - phase1_per_class) // 2
                selected_positions = sorted_positions[start : start + phase1_per_class]
            else:
                raise ValueError(f"Unknown mode for hybrid selection: {mode}")

        selected_class_indices = class_indices[selected_positions].tolist()
        phase1_indices.extend(selected_class_indices)

    logger.info(f"Phase 1 selected: {len(phase1_indices)} indices")

    # Phase 2: Global selection from remaining
    phase1_set = set(phase1_indices)
    remaining_mask = torch.tensor(
        [idx.item() not in phase1_set for idx in indices], dtype=torch.bool
    )
    remaining_scores = scores[remaining_mask]
    remaining_indices = indices[remaining_mask]

    logger.info(f"Remaining pool: {len(remaining_indices)} samples")

    phase2_indices = []
    if phase2_budget > 0 and len(remaining_indices) > 0:
        actual_phase2 = min(phase2_budget, len(remaining_indices))
        sorted_positions = torch.argsort(remaining_scores)
        if mode == "min":
            selected_positions = sorted_positions[:actual_phase2]
        elif mode == "max":
            selected_positions = sorted_positions[-actual_phase2:]
        elif mode == "mid":
            n = len(sorted_positions)
            start = (n - actual_phase2) // 2
            selected_positions = sorted_positions[start : start + actual_phase2]
        else:
            raise ValueError(f"Unknown mode for hybrid selection: {mode}")

        phase2_indices = remaining_indices[selected_positions].tolist()

    logger.info(f"Phase 2 selected: {len(phase2_indices)} indices")

    all_selected = phase1_indices + phase2_indices
    return phase1_indices, phase2_indices, all_selected


def double_budget_hybrid_merge(
    phase1_indices: List[int],
    phase2_indices: List[int],
    total_samples: int,
) -> List[int]:
    """Merge Phase 1 and Phase 2 selections using the double-budget strategy.

    This implements a simplified hybrid selection merge that:
    1. Keeps ALL Phase 1 samples (ensures class balance)
    2. Fills remaining budget from Phase 2 in order (respects ranking)
    3. Skips Phase 2 samples that overlap with Phase 1

    The key insight is that Phase 2 should be run with DOUBLE budget (total_samples)
    to ensure enough non-overlapping samples exist to fill the gap, even when
    Phase 1 and Phase 2 have significant overlap.

    This is equivalent to the incremental hybrid approach (with initial_subset)
    because submodular greedy selection is deterministic and incremental:
    selecting 400 samples = selecting 200, then continuing to select 200 more.

    Args:
        phase1_indices: Per-class selected indices (unordered, all kept)
        phase2_indices: Global selected indices (ORDERED by selection rank)
        total_samples: Target total number of samples

    Returns:
        Final selected indices (exactly total_samples, or less if insufficient)

    Example:
        >>> phase1 = [0, 1, 2, 3, 4]  # 5 per-class samples
        >>> phase2 = [2, 5, 6, 0, 7, 8, 9, 10]  # 8 global samples (ranked)
        >>> result = double_budget_hybrid_merge(phase1, phase2, total_samples=8)
        >>> # Result: [0, 1, 2, 3, 4, 5, 6, 7] - all phase1 + fill from phase2
    """
    phase1_set = set(phase1_indices)

    # Start with all Phase 1 samples (guaranteed class balance)
    final_indices = list(phase1_indices)

    # Fill remaining budget from Phase 2 in order (respects ranking)
    for idx in phase2_indices:
        if len(final_indices) >= total_samples:
            break
        if idx not in phase1_set:
            final_indices.append(idx)

    return final_indices
