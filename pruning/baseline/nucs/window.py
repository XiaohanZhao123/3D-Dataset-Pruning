"""NUCS Window-based Sample Selection with KRR Endpoint Optimization.

Implements the window selection component of NUCS (Non-Uniform Class-wise
Coreset Selection). Uses Kernel Ridge Regression to find the optimal
difficulty window endpoint.

Paper Algorithm:
1. Sort each class by difficulty scores (ascending)
2. For each candidate endpoint k in {0, t, 2t, ..., 1}:
   - Select samples from interval [k-r, k]% per class
   - Train KRR on selected subset features
   - Evaluate KRR on validation set
3. Return endpoint k* with best validation performance
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge

logger = logging.getLogger(__name__)


def select_window_indices(
    scores: torch.Tensor,
    labels: torch.Tensor,
    budgets: Dict[int, int],
    endpoint: float,
    num_classes: int,
) -> torch.Tensor:
    """Select sample indices using window-based selection.

    For each class, samples are sorted by difficulty (ascending) and
    a window [endpoint - ratio, endpoint] is selected.

    Args:
        scores: [N] tensor of difficulty scores (higher = harder)
        labels: [N] tensor of class labels
        budgets: Dict mapping class label to number of samples to select
        endpoint: Window endpoint as percentile (0.0 to 1.0)
                 e.g., 0.5 means select up to 50th percentile
        num_classes: Total number of classes

    Returns:
        1D tensor of selected global indices

    Example:
        If endpoint=0.75 and budget=10 for a class with 100 samples:
        - Sort samples by difficulty ascending
        - Select 10 samples from percentile range [0.65, 0.75]
        - This selects "medium-hard" samples
    """
    selected_indices = []

    for c in range(num_classes):
        budget = budgets.get(c, 0)
        if budget == 0:
            continue

        # Get indices for this class
        class_mask = labels == c
        class_global_indices = torch.where(class_mask)[0]
        class_scores = scores[class_mask]
        n_class = len(class_scores)

        if n_class == 0:
            continue

        # Sort by difficulty (ascending)
        sorted_local_indices = torch.argsort(class_scores)

        # Compute window range as percentile positions
        # ratio = budget / n_class (selection ratio for this class)
        ratio = budget / n_class

        # Window: [endpoint - ratio, endpoint]
        # Clamp to valid range [0, 1]
        window_start = max(0.0, endpoint - ratio)
        window_end = min(1.0, endpoint)

        # Convert percentiles to actual indices
        start_idx = int(window_start * n_class)
        end_idx = int(window_end * n_class)

        # Ensure we select exactly 'budget' samples
        # If window is too small, expand it
        if end_idx - start_idx < budget:
            # Expand window to include enough samples
            # Try expanding towards lower difficulty first
            shortage = budget - (end_idx - start_idx)
            start_idx = max(0, start_idx - shortage)

            # If still not enough, expand towards higher difficulty
            if end_idx - start_idx < budget:
                end_idx = min(n_class, start_idx + budget)

        # Select samples from window
        window_local_indices = sorted_local_indices[start_idx:end_idx]

        # If we have more than budget, take the ones closest to endpoint
        if len(window_local_indices) > budget:
            # Take the last 'budget' samples (closest to endpoint)
            window_local_indices = window_local_indices[-budget:]

        # Convert to global indices
        selected_global = class_global_indices[window_local_indices]
        selected_indices.append(selected_global)

    if selected_indices:
        return torch.cat(selected_indices)
    else:
        return torch.tensor([], dtype=torch.long)


def train_krr(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> KernelRidge:
    """Train Kernel Ridge Regression model.

    Args:
        features: [N, D] feature matrix
        labels: [N] label array (converted to one-hot internally)
        alpha: Regularization strength (lambda in paper)
        kernel: Kernel type ("rbf", "linear", "poly")
        gamma: Kernel coefficient for RBF. If None, uses 1/n_features

    Returns:
        Trained KernelRidge model
    """
    if gamma is None:
        gamma = 1.0 / features.shape[1]

    # Convert labels to one-hot for multi-class regression
    num_classes = int(labels.max()) + 1
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1

    krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
    krr.fit(features, one_hot)

    return krr


def evaluate_krr(
    krr: KernelRidge,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Evaluate KRR model accuracy.

    Args:
        krr: Trained KernelRidge model
        features: [N, D] feature matrix
        labels: [N] ground truth labels

    Returns:
        Classification accuracy (0.0 to 1.0)
    """
    predictions = krr.predict(features)
    predicted_labels = predictions.argmax(axis=1)
    accuracy = (predicted_labels == labels).mean()
    return float(accuracy)


def find_optimal_endpoint(
    scores: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    budgets: Dict[int, int],
    num_classes: int,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    endpoint_candidates: Optional[List[float]] = None,
    krr_alpha: float = 1.0,
    krr_kernel: str = "rbf",
    krr_gamma: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """Find optimal window endpoint using KRR validation.

    Tests multiple endpoint candidates and returns the one with
    best KRR validation accuracy.

    Args:
        scores: [N] difficulty scores for training samples
        labels: [N] class labels for training samples
        features: [N, D] feature embeddings for training samples
        budgets: Per-class sample budgets from NUCS budget allocator
        num_classes: Total number of classes
        val_features: [M, D] validation features (if None, uses train features)
        val_labels: [M] validation labels (if None, uses train labels)
        endpoint_candidates: List of endpoints to test (default: [0.25, 0.5, 0.75, 1.0])
        krr_alpha: KRR regularization strength
        krr_kernel: KRR kernel type
        krr_gamma: KRR kernel coefficient

    Returns:
        Tuple of (optimal_endpoint, results_dict)
        results_dict maps endpoint -> validation accuracy
    """
    if endpoint_candidates is None:
        # Default: test 4 endpoints (25%, 50%, 75%, 100%)
        endpoint_candidates = [0.25, 0.5, 0.75, 1.0]

    # Use training data for validation if not provided
    if val_features is None:
        val_features = features
    if val_labels is None:
        val_labels = labels

    # Convert to numpy for sklearn
    features_np = features.cpu().numpy() if torch.is_tensor(features) else features
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    val_features_np = val_features.cpu().numpy() if torch.is_tensor(val_features) else val_features
    val_labels_np = val_labels.cpu().numpy() if torch.is_tensor(val_labels) else val_labels

    results = {}
    best_endpoint = endpoint_candidates[0]
    best_accuracy = -1.0

    for endpoint in endpoint_candidates:
        # Select samples using this endpoint
        selected_indices = select_window_indices(
            scores=scores,
            labels=labels,
            budgets=budgets,
            endpoint=endpoint,
            num_classes=num_classes,
        )

        if len(selected_indices) == 0:
            logger.warning(f"Endpoint {endpoint}: No samples selected, skipping")
            results[endpoint] = 0.0
            continue

        # Get features and labels for selected samples
        selected_indices_np = selected_indices.cpu().numpy()
        train_features = features_np[selected_indices_np]
        train_labels = labels_np[selected_indices_np]

        # Train KRR
        try:
            krr = train_krr(
                features=train_features,
                labels=train_labels,
                alpha=krr_alpha,
                kernel=krr_kernel,
                gamma=krr_gamma,
            )

            # Evaluate on validation set
            accuracy = evaluate_krr(krr, val_features_np, val_labels_np)
            results[endpoint] = accuracy

            logger.info(
                f"Endpoint {endpoint:.2f}: selected {len(selected_indices)} samples, "
                f"val accuracy = {accuracy:.4f}"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_endpoint = endpoint

        except Exception as e:
            logger.warning(f"Endpoint {endpoint}: KRR failed with error: {e}")
            results[endpoint] = 0.0

    logger.info(f"Optimal endpoint: {best_endpoint:.2f} (accuracy: {best_accuracy:.4f})")

    return best_endpoint, results


class NUCSWindowSelector:
    """High-level interface for NUCS window-based selection.

    Combines window selection with KRR endpoint optimization.

    Example:
        >>> selector = NUCSWindowSelector(num_classes=40)
        >>> # Find optimal endpoint
        >>> endpoint, results = selector.find_optimal_endpoint(
        ...     scores=difficulty_scores,
        ...     labels=labels,
        ...     features=embeddings,
        ...     budgets=per_class_budgets,
        ... )
        >>> # Select samples with optimal endpoint
        >>> selected_indices = selector.select(
        ...     scores=difficulty_scores,
        ...     labels=labels,
        ...     budgets=per_class_budgets,
        ...     endpoint=endpoint,
        ... )
    """

    def __init__(
        self,
        num_classes: int,
        krr_alpha: float = 1.0,
        krr_kernel: str = "rbf",
        krr_gamma: Optional[float] = None,
        endpoint_candidates: Optional[List[float]] = None,
    ):
        """Initialize NUCS window selector.

        Args:
            num_classes: Total number of classes
            krr_alpha: KRR regularization strength
            krr_kernel: KRR kernel type ("rbf", "linear", "poly")
            krr_gamma: KRR kernel coefficient (None = auto)
            endpoint_candidates: Endpoints to test (default: [0.25, 0.5, 0.75, 1.0])
        """
        self.num_classes = num_classes
        self.krr_alpha = krr_alpha
        self.krr_kernel = krr_kernel
        self.krr_gamma = krr_gamma
        self.endpoint_candidates = endpoint_candidates or [0.25, 0.5, 0.75, 1.0]

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        budgets: Dict[int, int],
        endpoint: float,
    ) -> torch.Tensor:
        """Select samples using window-based selection.

        Args:
            scores: [N] difficulty scores
            labels: [N] class labels
            budgets: Per-class budgets
            endpoint: Window endpoint (0.0 to 1.0)

        Returns:
            1D tensor of selected indices
        """
        return select_window_indices(
            scores=scores,
            labels=labels,
            budgets=budgets,
            endpoint=endpoint,
            num_classes=self.num_classes,
        )

    def find_optimal_endpoint(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        budgets: Dict[int, int],
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal endpoint using KRR validation.

        Args:
            scores: [N] difficulty scores
            labels: [N] class labels
            features: [N, D] feature embeddings
            budgets: Per-class budgets
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)

        Returns:
            Tuple of (optimal_endpoint, results_dict)
        """
        return find_optimal_endpoint(
            scores=scores,
            labels=labels,
            features=features,
            budgets=budgets,
            num_classes=self.num_classes,
            val_features=val_features,
            val_labels=val_labels,
            endpoint_candidates=self.endpoint_candidates,
            krr_alpha=self.krr_alpha,
            krr_kernel=self.krr_kernel,
            krr_gamma=self.krr_gamma,
        )

    def select_with_optimization(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        budgets: Dict[int, int],
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        """Find optimal endpoint and select samples in one call.

        Args:
            scores: [N] difficulty scores
            labels: [N] class labels
            features: [N, D] feature embeddings
            budgets: Per-class budgets
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)

        Returns:
            Tuple of (selected_indices, optimal_endpoint, results_dict)
        """
        endpoint, results = self.find_optimal_endpoint(
            scores=scores,
            labels=labels,
            features=features,
            budgets=budgets,
            val_features=val_features,
            val_labels=val_labels,
        )

        selected_indices = self.select(
            scores=scores,
            labels=labels,
            budgets=budgets,
            endpoint=endpoint,
        )

        return selected_indices, endpoint, results
