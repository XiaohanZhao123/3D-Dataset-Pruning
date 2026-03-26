"""Custom samplers for class-balanced and distribution-aware training."""

import random
from collections import defaultdict
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler, Subset


def get_labels_from_dataset(dataset):
    """Extract labels from dataset, handling Subset wrapper.

    Args:
        dataset: Dataset or Subset with labels

    Returns:
        List of labels for each sample in the dataset
    """
    # Handle Subset: extract labels for the subset indices only
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices

        # Get labels from base dataset
        if hasattr(base_dataset, "targets"):
            base_labels = base_dataset.targets
        elif hasattr(base_dataset, "labels"):
            base_labels = base_dataset.labels
        elif hasattr(base_dataset, "label"):
            base_labels = base_dataset.label
        else:
            raise ValueError(
                "Base dataset must have 'targets', 'labels', or 'label' attribute"
            )

        # Convert to list if tensor
        if isinstance(base_labels, torch.Tensor):
            base_labels = base_labels.tolist()

        # Extract labels for subset indices
        return [base_labels[i] for i in indices]

    # Handle regular dataset
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "label"):
        labels = dataset.label
    else:
        raise ValueError("Dataset must have 'targets', 'labels', or 'label' attribute")

    if isinstance(labels, torch.Tensor):
        return labels.tolist()
    return list(labels)


class UniformClassSampler(Sampler):
    """
    Uniform over-classes sampler for class-balanced training.

    For each batch:
    1. Sample B class labels uniformly (each class has equal probability)
    2. For each sampled label, randomly select one data point from that class

    This ensures each class has equal representation probability regardless of
    class size, unlike standard sampling which is proportional to class size.

    Args:
        dataset: Dataset with .targets or .labels attribute
        batch_size: Number of samples per batch
        num_classes: Total number of classes
        num_samples: Total samples per epoch (default: len(dataset))

    Example:
        With batch_size=32, num_classes=40:
        - Sample 32 labels: [5, 12, 5, 3, 40, 5, ...] (uniform random)
        - For label 5: pick random sample from all class-5 samples
        - For label 12: pick random sample from all class-12 samples
        - etc.

        A class can appear multiple times in the same batch.
    """

    def __init__(
        self, dataset, batch_size: int, num_classes: int, num_samples: int = None
    ):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_samples = num_samples if num_samples is not None else len(dataset)

        # Build mapping: class_id -> list of sample indices
        self.class_to_indices = defaultdict(list)

        # Get labels from dataset (handles Subset)
        labels = get_labels_from_dataset(dataset)

        # Populate class_to_indices mapping
        for idx, label in enumerate(labels):
            self.class_to_indices[int(label)].append(idx)

        # Only validate classes that exist in the dataset (pruned datasets may not have all classes)
        self.active_classes = [
            c for c in range(num_classes) if len(self.class_to_indices[c]) > 0
        ]
        if len(self.active_classes) == 0:
            raise ValueError("No classes found in dataset")

        print("UniformClassSampler initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Active classes: {len(self.active_classes)}/{num_classes}")
        print(f"  Samples per epoch: {self.num_samples}")

    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch."""
        num_batches = self.num_samples // self.batch_size

        for _ in range(num_batches):
            batch_indices = []

            # Sample batch_size class labels uniformly from ACTIVE classes only
            sampled_classes = random.choices(self.active_classes, k=self.batch_size)

            # For each sampled class, pick one random data point
            for class_id in sampled_classes:
                # Randomly select one sample from this class
                sample_idx = random.choice(self.class_to_indices[class_id])
                batch_indices.append(sample_idx)

            # Yield all indices in this batch
            yield from batch_indices

    def __len__(self) -> int:
        """Total number of samples per epoch."""
        return (self.num_samples // self.batch_size) * self.batch_size


class BalancedClassSampler(Sampler):
    """
    Class-balanced sampler with configurable alpha exponent.

    Implements the sampling strategy from:
    Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition," ICLR 2020.

    Sampling probability for class c: p_c ∝ n_c^alpha
    - alpha = 1.0: original (imbalanced) distribution
    - alpha = 0.5: square-root balanced (common compromise)
    - alpha = 0.0: fully class-balanced (uniform over classes)

    Args:
        dataset: Dataset with .targets, .labels, or .label attribute
        num_classes: Total number of classes
        alpha: Sampling exponent (default: 0.5 for sqrt-balanced)
        num_samples: Total samples per epoch (default: len(dataset))

    Example:
        With class counts [1000, 100, 10] and alpha=0.5:
        - Original probs: [0.90, 0.09, 0.01]
        - Sqrt probs: [31.6, 10, 3.16] / 44.76 = [0.71, 0.22, 0.07]

        Tail classes get higher sampling probability with lower alpha.
    """

    def __init__(
        self,
        dataset,
        num_classes: int,
        alpha: float = 0.5,
        num_samples: int = None,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.num_samples = num_samples if num_samples is not None else len(dataset)

        # Get labels from dataset (handles Subset)
        labels = get_labels_from_dataset(dataset)
        labels = np.array(labels)

        # Build class_to_indices mapping
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[int(label)].append(idx)

        # Compute class counts
        class_counts = np.array(
            [len(self.class_to_indices[c]) for c in range(num_classes)],
            dtype=np.float64,
        )

        # Compute sampling probabilities: p_c ∝ n_c^alpha
        class_probs = np.power(class_counts + 1e-8, alpha)  # Add epsilon to avoid zero
        class_probs = class_probs / class_probs.sum()
        self.class_probs = class_probs

        # Compute per-sample weights for WeightedRandomSampler-style iteration
        # weight[i] = 1 / p_{label[i]} (inverse of class probability)
        sample_weights = 1.0 / class_probs[labels]
        sample_weights = sample_weights / sample_weights.sum()
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.float64)

        # Track active classes
        self.active_classes = [c for c in range(num_classes) if class_counts[c] > 0]

        alpha_name = {1.0: "original", 0.5: "sqrt", 0.0: "class-balanced"}.get(
            alpha, "custom"
        )
        print("BalancedClassSampler initialized:")
        print(f"  Alpha: {alpha} ({alpha_name})")
        print(f"  Active classes: {len(self.active_classes)}/{num_classes}")
        print(f"  Samples per epoch: {self.num_samples}")

    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch using weighted random sampling."""
        # Sample indices with replacement based on sample weights
        indices = torch.multinomial(
            self.sample_weights, self.num_samples, replacement=True
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        """Total number of samples per epoch."""
        return self.num_samples


class PrunedDistributionSampler(Sampler):
    """
    Sample from FULL dataset with class weights derived from PRUNED dataset distribution.

    This sampler enables training on the full dataset while following the class
    distribution of a pruned subset. Useful for retraining a teacher model's
    classifier to match the pruned data distribution.

    Sampling process:
    1. Compute class distribution from pruned indices
    2. Sample class c with probability p(c) = pruned_count[c] / total_pruned
    3. Sample a random data point from class c in the FULL dataset

    Args:
        full_dataset: Original full training dataset
        pruned_indices: List of indices selected by pruning
        num_classes: Total number of classes
        batch_size: Number of samples per batch
        num_samples: Total samples per epoch (default: len(full_dataset))

    Example:
        If pruning selected: class 0: 50, class 1: 10, class 2: 40
        Then p(class 0) = 50/100 = 0.5, p(class 1) = 0.1, p(class 2) = 0.4

        During sampling:
        - 50% chance to sample from class 0 (using FULL dataset's class 0 samples)
        - 10% chance to sample from class 1 (using FULL dataset's class 1 samples)
        - 40% chance to sample from class 2 (using FULL dataset's class 2 samples)
    """

    def __init__(
        self,
        full_dataset,
        pruned_indices: List[int],
        num_classes: int,
        batch_size: int,
        num_samples: int = None,
    ):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_samples = num_samples if num_samples is not None else len(full_dataset)

        # Get labels from full dataset
        if hasattr(full_dataset, "targets"):
            all_labels = full_dataset.targets
        elif hasattr(full_dataset, "labels"):
            all_labels = full_dataset.labels
        elif hasattr(full_dataset, "label"):
            all_labels = full_dataset.label
        else:
            raise ValueError(
                "Dataset must have 'targets', 'labels', or 'label' attribute"
            )

        # Convert to list if tensor
        if isinstance(all_labels, torch.Tensor):
            all_labels = all_labels.tolist()

        # Build class_to_indices mapping for FULL dataset
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(all_labels):
            self.class_to_indices[int(label)].append(idx)

        # Compute class distribution from PRUNED indices
        pruned_labels = [int(all_labels[i]) for i in pruned_indices]
        pruned_class_counts = np.bincount(pruned_labels, minlength=num_classes)

        # Compute class sampling probabilities
        total_pruned = len(pruned_indices)
        self.class_probs = pruned_class_counts / total_pruned

        # Handle classes with zero probability (not in pruned set)
        # These classes won't be sampled
        self.active_classes = np.where(self.class_probs > 0)[0]

        print("PrunedDistributionSampler initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Num classes: {num_classes}")
        print(f"  Active classes (in pruned set): {len(self.active_classes)}")
        print(f"  Samples per epoch: {self.num_samples}")
        print(f"  Total pruned samples: {total_pruned}")
        print("  Pruned class distribution (top 10):")
        sorted_classes = np.argsort(pruned_class_counts)[::-1]
        for i, class_id in enumerate(sorted_classes[:10]):
            count = pruned_class_counts[class_id]
            prob = self.class_probs[class_id]
            full_count = len(self.class_to_indices[class_id])
            print(
                f"    Class {class_id}: {count} pruned ({prob:.3f}) | {full_count} in full"
            )
        if num_classes > 10:
            print(f"    ... ({num_classes - 10} more classes)")

    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch."""
        num_batches = self.num_samples // self.batch_size

        for _ in range(num_batches):
            batch_indices = []

            # Sample batch_size class labels according to pruned distribution
            sampled_classes = np.random.choice(
                self.num_classes, size=self.batch_size, p=self.class_probs
            )

            # For each sampled class, pick one random data point from FULL dataset
            for class_id in sampled_classes:
                class_id = int(class_id)
                # Randomly select one sample from this class in FULL dataset
                sample_idx = random.choice(self.class_to_indices[class_id])
                batch_indices.append(sample_idx)

            yield from batch_indices

    def __len__(self) -> int:
        """Total number of samples per epoch."""
        return (self.num_samples // self.batch_size) * self.batch_size

    def get_class_distribution(self) -> np.ndarray:
        """Return the class sampling probabilities."""
        return self.class_probs.copy()
