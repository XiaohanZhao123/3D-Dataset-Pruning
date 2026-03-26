"""Composition-based sample selection for dataset pruning experiments.

This module provides utilities for selecting training samples based on their
difficulty (easy/medium/hard) and testing different composition strategies.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from torch.utils.data import Subset


def generate_composition_configs() -> List[Tuple[float, float, float]]:
    """Generate ~15 composition configurations as percentages (MVP mode).

    Configurations follow the strategy:
    - 3 boundary cases: (100%, 0%, 0%), (0%, 100%, 0%), (0%, 0%, 100%)
    - Easy sweep: E=20%, 40%, 60%, 80% (4 configs)
    - Medium sweep: M=20%, 40%, 60%, 80% (4 configs, remove duplicates)
    - Hard sweep: H=20%, 40%, 60%, 80% (4 configs, remove duplicates)

    Returns:
        List of (easy_pct, medium_pct, hard_pct) tuples, each summing to 1.0
        Total: ~12-15 unique configurations
    """
    configs = []

    # === Boundary Cases ===
    configs.append((1.0, 0.0, 0.0))  # All easy
    configs.append((0.0, 1.0, 0.0))  # All medium
    configs.append((0.0, 0.0, 1.0))  # All hard

    # === Easy Sweep (E=20%, 40%, 60%, 80%) ===
    for e_pct in [0.2, 0.4, 0.6, 0.8]:
        remainder = 1.0 - e_pct
        m_pct, h_pct = _split_remainder(remainder, priority='hard')
        configs.append((e_pct, m_pct, h_pct))

    # === Medium Sweep (M=20%, 40%, 60%, 80%) ===
    for m_pct in [0.2, 0.4, 0.6, 0.8]:
        remainder = 1.0 - m_pct
        e_pct, h_pct = _split_remainder(remainder, priority='hard')
        config = (e_pct, m_pct, h_pct)
        if config not in configs:  # Avoid duplicates
            configs.append(config)

    # === Hard Sweep (H=20%, 40%, 60%, 80%) ===
    for h_pct in [0.2, 0.4, 0.6, 0.8]:
        remainder = 1.0 - h_pct
        e_pct, m_pct = _split_remainder(remainder, priority='medium')
        config = (e_pct, m_pct, h_pct)
        if config not in configs:  # Avoid duplicates
            configs.append(config)

    return configs


def _split_remainder(remainder: float, priority: str) -> Tuple[float, float]:
    """Split remainder percentage between two dimensions.

    Args:
        remainder: Percentage to split (0.0-1.0)
        priority: Which dimension gets extra if odd ('hard', 'medium', or 'easy')

    Returns:
        Tuple of (first_pct, second_pct) based on priority
    """
    half = remainder / 2.0

    # For splitting, priority determines which gets the larger share
    # if remainder can't be evenly divided
    if priority == 'hard':
        return (half, half)  # Equal split, second is "hard"
    elif priority == 'medium':
        return (half, half)  # Equal split, second is "medium"
    else:  # priority == 'easy'
        return (half, half)  # Equal split

    # Note: In percentage form, splits are always even (0.5, 0.5)
    # The "extra sample" logic applies during conversion to integer counts


def define_difficulty_pools(
    scores: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    num_classes: int,
    pool_size: float = 0.33
) -> Dict[int, Dict[str, np.ndarray]]:
    """Define easy/medium/hard pools for each class based on loss scores.

    Args:
        scores: [num_samples] loss scores for all samples
        labels: [num_samples] ground truth class labels
        indices: [num_samples] original dataset indices
        num_classes: Total number of classes
        pool_size: Fraction of samples in easy/hard pools (default 0.33 = 33%)

    Returns:
        Dict mapping class_id → {
            'easy': array of dataset indices (bottom pool_size by loss),
            'medium': array of dataset indices (middle),
            'hard': array of dataset indices (top pool_size by loss)
        }
    """
    pools = {}

    for c in range(num_classes):
        # Get all samples for this class
        class_mask = labels == c
        class_scores = scores[class_mask]
        class_indices = indices[class_mask]

        # Sort by loss (ascending: easy → hard)
        sorted_positions = torch.argsort(class_scores)
        sorted_indices = class_indices[sorted_positions]

        n_samples = len(sorted_indices)
        n_pool = int(n_samples * pool_size)

        # Define pool boundaries
        easy_end = n_pool
        hard_start = n_samples - n_pool

        pools[c] = {
            'easy': sorted_indices[:easy_end].cpu().numpy(),
            'medium': sorted_indices[easy_end:hard_start].cpu().numpy(),
            'hard': sorted_indices[hard_start:].cpu().numpy()
        }

    return pools


def sample_by_composition(
    pools: Dict[int, Dict[str, np.ndarray]],
    easy_pct: float,
    medium_pct: float,
    hard_pct: float,
    samples_per_class: int,
    seed: int
) -> List[int]:
    """Sample from difficulty pools according to composition percentages.

    Args:
        pools: Output from define_difficulty_pools
        easy_pct: Percentage of samples from easy pool (0.0-1.0)
        medium_pct: Percentage of samples from medium pool (0.0-1.0)
        hard_pct: Percentage of samples from hard pool (0.0-1.0)
        samples_per_class: Total number of samples to select per class
        seed: Random seed for reproducibility

    Returns:
        List of selected dataset indices across all classes
    """
    rng = np.random.RandomState(seed)
    selected_indices = []

    # Convert percentages to counts with rounding rules
    n_easy_target = easy_pct * samples_per_class
    n_medium_target = medium_pct * samples_per_class
    n_hard_target = hard_pct * samples_per_class

    # Round with priority: Hard > Medium > Easy
    # This ensures we always get exactly samples_per_class
    n_easy, n_medium, n_hard = _round_with_priority(
        n_easy_target, n_medium_target, n_hard_target, samples_per_class
    )

    for class_id, class_pools in pools.items():
        class_selected = []

        # Sample from easy pool
        if n_easy > 0:
            easy_pool = class_pools['easy']
            n_sample = min(n_easy, len(easy_pool))
            sampled = rng.choice(easy_pool, size=n_sample, replace=False)
            class_selected.extend(sampled.tolist())

        # Sample from medium pool
        if n_medium > 0:
            medium_pool = class_pools['medium']
            n_sample = min(n_medium, len(medium_pool))
            sampled = rng.choice(medium_pool, size=n_sample, replace=False)
            class_selected.extend(sampled.tolist())

        # Sample from hard pool
        if n_hard > 0:
            hard_pool = class_pools['hard']
            n_sample = min(n_hard, len(hard_pool))
            sampled = rng.choice(hard_pool, size=n_sample, replace=False)
            class_selected.extend(sampled.tolist())

        selected_indices.extend(class_selected)

    return selected_indices


def _round_with_priority(
    easy: float,
    medium: float,
    hard: float,
    target_total: int
) -> Tuple[int, int, int]:
    """Round three floats to integers with priority Hard > Medium > Easy.

    Ensures the sum equals target_total exactly.

    Args:
        easy, medium, hard: Float values to round
        target_total: Required sum of rounded values

    Returns:
        Tuple of (easy_rounded, medium_rounded, hard_rounded)
    """
    # Start with floor values
    e = int(np.floor(easy))
    m = int(np.floor(medium))
    h = int(np.floor(hard))

    # Calculate remaining budget
    remaining = target_total - (e + m + h)

    # Distribute remaining based on fractional parts and priority
    fracs = [
        (easy - e, 'easy'),
        (medium - m, 'medium'),
        (hard - h, 'hard')
    ]

    # Sort by priority: hard > medium > easy
    priority_order = {'hard': 0, 'medium': 1, 'easy': 2}
    fracs.sort(key=lambda x: (priority_order[x[1]], -x[0]))

    # Assign remaining samples by priority
    for i in range(remaining):
        _, dim = fracs[i]
        if dim == 'easy':
            e += 1
        elif dim == 'medium':
            m += 1
        else:  # hard
            h += 1

    return e, m, h


def create_composition_subset(
    dataset,
    pools: Dict[int, Dict[str, np.ndarray]],
    easy_pct: float,
    medium_pct: float,
    hard_pct: float,
    samples_per_class: int,
    seed: int
) -> Subset:
    """Create a dataset subset based on composition.

    Convenience function combining sample_by_composition and Subset creation.

    Args:
        dataset: Original dataset
        pools: Difficulty pools from define_difficulty_pools
        easy_pct, medium_pct, hard_pct: Composition percentages
        samples_per_class: Number of samples per class
        seed: Random seed

    Returns:
        Subset of dataset with selected samples
    """
    selected_indices = sample_by_composition(
        pools, easy_pct, medium_pct, hard_pct, samples_per_class, seed
    )
    return Subset(dataset, selected_indices)
