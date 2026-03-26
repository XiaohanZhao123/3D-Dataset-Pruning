"""
Score function factories for data is_pruning.

This module provides factory functions that create pure scoring functions via closures.
Each factory captures necessary dependencies (models, devices, etc.) and returns a function
that maps samples to scores for is_pruning purposes.
"""

from typing import Any, Callable, Dict, Union, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def make_loss_scorer(
    feature_pipe: Callable[[Dict[str, Any]], torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
) -> Callable[[Dict[str, Any]], torch.Tensor]:
    """
    Create a loss-based scorer function.

    Args:
        feature_pipe: Callable that computes loss for samples, maps batch -> loss_values
        loss_fn: Callable that use features and labels to compute loss

    Returns:
        Scorer function that takes samples and returns loss scores
    """

    def score_samples(
        samples: Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute loss scores for samples."""
        if "labels" in samples:
            labels = samples["labels"]
        elif "y" in samples:
            labels = samples["y"]
        else:
            labels = samples[1]  # the second element
        features = feature_pipe(samples)
        labels = labels.to(features.device)
        loss = loss_fn(features, labels, reduction="none")
        return loss

    return score_samples


def make_prototype_scorer(
    feature_pipe: Callable[[Dict[str, Any]], torch.Tensor],
) -> Callable[[Dict[str, Any]], torch.Tensor]:
    """
    Create a prototype distance-based scorer function.

    Args:
        feature_pipe: Callable that extracts features from samples

    Returns:
        Scorer function that takes samples and returns prototype distance scores
    """

    def score_samples(samples: Dict[str, Any]) -> torch.Tensor:
        """Compute prototype distance scores for samples."""
        features = feature_pipe(samples).detach()
        prototype_feature = torch.mean(features, dim=0, keepdim=True)
        distances = torch.norm(features - prototype_feature, dim=-1)

        return distances

    return score_samples


def make_cluster_scorer(
    feature_pipe: Callable[[Dict[str, Any]], torch.Tensor],
    n_clusters: int,
    random_state: int = 42,
) -> Callable[[Dict[str, Any]], torch.Tensor]:
    """
    Create a cluster distance-based scorer function.

    Args:
        feature_pipe: Callable that extracts features from samples
        n_clusters: Number of clusters for k-means
        random_state: Random seed for reproducibility

    Returns:
        Scorer function that takes samples and returns minimum cluster distance scores
    """

    def score_samples(samples: Dict[str, Any]) -> torch.Tensor:
        """Compute minimum cluster distance scores for samples."""
        features = feature_pipe(samples)
        features_np = features.detach().cpu().numpy()

        # Normalize features
        features_np = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)

        # Handle edge case: if n_clusters > number of samples, use all samples as clusters
        num_samples = features_np.shape[0]
        effective_clusters = min(n_clusters, num_samples)

        # K-means clustering on CPU
        kmeans = KMeans(
            n_clusters=effective_clusters, random_state=random_state, n_init="auto"
        )
        kmeans.fit(features_np)

        # Compute distances from each sample to all cluster centers
        distances = cdist(kmeans.cluster_centers_, features_np, metric="euclidean")

        # Return minimum distance for each sample (closest cluster distance)
        min_distances = np.min(distances, axis=0)

        return torch.from_numpy(min_distances).to(features.device)

    return score_samples


def make_herding_scorer(
    feature_pipe: Callable[[Dict[str, Any]], torch.Tensor],
    num_samples: int,
) -> Callable[[Dict[str, Any]], torch.Tensor]:
    """
    Create a herding-based scorer function.

    Args:
        feature_pipe: Callable that extracts features from samples
        num_samples: Number of samples to select via herding

    Returns:
        Scorer function that takes samples and returns herding scores
    """

    def score_samples(samples: Dict[str, Any]) -> torch.Tensor:
        """Compute herding scores for samples."""
        features = feature_pipe(samples).detach()
        
        # Compute mean of all features
        mu = features.mean(0)
        
        # Initialize scores to -100 for all samples (ensures unselected samples have very low scores)
        scores = torch.full((len(features),), -100.0, device=features.device)
        
        # Iteratively select samples using herding algorithm
        res = mu.clone()
        sum_sel = torch.zeros_like(mu)
        
        for t in range(1, min(num_samples + 1, len(features) + 1)):
            scores_iter = features @ res
            j = scores_iter.argmax().item()
            
            # Select j-th sample (assign positive score for selected samples)
            scores[j] = 1.0
            
            # Update running sum and residual
            sum_sel += features[j]
            res = mu - sum_sel / t
        
        return scores

    return score_samples


# Alias for backward compatibility
make_clustering_score_pipeline = make_cluster_scorer


# Alias for backward compatibility
make_prototype_score_pipeline = make_prototype_scorer
