"""Balanced model scorers with OOP design and registry pattern.

Design decisions:
- Registry pattern: Easy extension, no if-elif chains in main()
- Mixin for shared logic: Feature extraction shared across herding/kcenter/submodular

Usage:
    from pruning.balanced_scorers import get_scorer, SCORER_REGISTRY

    scorer = get_scorer("loss", model, cfg, device="cuda")
    scores, labels, indices = scorer.compute(dataloader, **kwargs)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Optional apricot import for fast submodular selection
try:
    from apricot import FacilityLocationSelection

    APRICOT_AVAILABLE = True
except ImportError:
    APRICOT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Base Class
# ============================================================================


class BaseScorer(ABC):
    """Base class for all scorers.

    Simple interface - no over-abstraction. Each scorer implements compute().

    Class attributes:
        name: Primary name for registry lookup
        aliases: Alternative names (e.g., "gradnorm" -> "grad_norm")
        mode_override: Force selection mode ("max" for herding/kcenter, None for loss-based)
        requires_grad: Whether scorer needs model gradients (affects model loading)
        is_score_based: True if scorer returns continuous scores (can filter at select time),
                        False if scorer does selection in compute() (returns binary 1.0/-100.0)
        supports_hybrid: True if scorer supports hybrid selection mode.
                         Custom-logic scorers (NUCS, DRoP) should set this to False.
    """

    name: str = "base"
    aliases: List[str] = []
    mode_override: Optional[str] = None
    requires_grad: bool = False
    is_score_based: bool = True  # Default: continuous scores, filtered at select time
    supports_hybrid: bool = True  # Default: supports hybrid selection

    def __init__(
        self,
        model: nn.Module,
        cfg,
        device: str = "cuda",
        mislabel_ratio: float = 0.0,
        num_strata: int = 50,
        ccscp_min_samples: int = 1,
    ):
        """Initialize scorer.

        Args:
            model: Teacher model
            cfg: OpenPoint config (cfg.openpoint from merged config)
            device: Device for computation
            mislabel_ratio: (CCS only) Fraction of hard samples to remove before selection
            num_strata: (CCS only) Number of bins for stratified sampling (default 50)
            ccscp_min_samples: (CCS-CP only) Minimum samples per class (default 1)
        """
        self.model = model
        self.cfg = cfg
        self.device = device

        # CCS-specific parameters
        self.mislabel_ratio = mislabel_ratio
        self.num_strata = num_strata
        self.ccscp_min_samples = ccscp_min_samples

        # Get actual model (handle DataParallel)
        self.actual_model = model.module if hasattr(model, "module") else model

    @abstractmethod
    def compute(
        self, dataloader, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scores for all samples.

        Args:
            dataloader: Non-shuffled dataloader for scoring
            **kwargs: Scorer-specific arguments

        Returns:
            scores: [N] tensor of scores
            labels: [N] tensor of ground truth labels
            indices: [N] tensor of original dataset indices
        """
        pass

    def prepare_batch(self, data: Dict) -> Dict:
        """Prepare batch for model forward pass (no point truncation during scoring).

        Args:
            data: Raw batch dict from dataloader

        Returns:
            Prepared data dict with 'pos' and 'x' keys
        """
        for key in data.keys():
            data[key] = data[key].to(self.device, non_blocking=True)

        # Ensure labels are long type for cross-entropy loss
        if "y" in data:
            data["y"] = data["y"].long()

        points = data["x"]
        data["pos"] = points[:, :, :3].contiguous()
        in_channels = getattr(self.cfg.model.encoder_args, "in_channels", 3)
        data["x"] = points[:, :, :in_channels].transpose(1, 2).contiguous()
        return data

    def get_batch_indices(
        self, batch_idx: int, declared_batch_size: int, actual_size: int
    ) -> torch.Tensor:
        """Compute original dataset indices for a batch.

        Args:
            batch_idx: Current batch index
            declared_batch_size: Dataloader's declared batch size (for offset calculation)
            actual_size: Actual batch size (may be smaller for last batch)

        Returns:
            Tensor of dataset indices [actual_size]
        """
        start_idx = batch_idx * declared_batch_size
        return torch.arange(start_idx, start_idx + actual_size)

    def get_single_head_logits(self, data: Dict) -> torch.Tensor:
        """Forward pass using head 0 only (standard for all scorers).

        Args:
            data: Prepared data dict

        Returns:
            Logits [B, num_classes]
        """
        # Point-MAE has group_divider and expects tensor [B, N, 3]
        # PointNeXt/PointMLP has encoder and expects dict with 'pos' and 'x'
        actual_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if hasattr(actual_model, "group_divider"):
            return self.model(data["pos"])
        return self.model(data)

    def get_encoder_features(self, data: Dict) -> torch.Tensor:
        """Extract encoder features (pre-classifier).

        Works with both Point-MAE (via get_embeddings) and PointNeXt (via encoder).

        Args:
            data: Prepared data dict

        Returns:
            Features [B, D]
        """
        # Point-MAE has group_divider and expects tensor [B, N, 3]
        # PointNeXt/PointMLP has encoder and expects dict with 'pos' and 'x'
        actual_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if hasattr(actual_model, "group_divider"):
            return actual_model.get_embeddings(data["pos"])
        # PointNeXt encoder method
        return actual_model.encoder.forward_cls_feat(data)

    # ========================================================================
    # Selection methods
    # ========================================================================

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        **kwargs,
    ) -> List[int]:
        """Convert scores to selected dataset indices.

        Default implementation: threshold-based selection for continuous scores.
        Binary scorers (herding, kcenter, submodular) override this to return
        indices where score == 1.0.

        Args:
            scores: [N] tensor of scores from compute()
            labels: [N] tensor of ground truth labels
            indices: [N] tensor of original dataset indices
            total_samples: Total number of samples to select. When per_class=True,
                          this is divided evenly among classes.
            per_class: If True, divide total_samples evenly among classes
            mode: Selection mode ['min', 'max', 'mid', 'random']
            num_classes: Total number of classes
            **kwargs: Scorer-specific parameters (e.g., NUCS parameters)

        Returns:
            List of selected dataset indices

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """
        if mode == "random":
            return self._random_select(
                labels, indices, total_samples, per_class, num_classes
            )
        return self._score_based_select(
            scores, labels, indices, total_samples, per_class, mode, num_classes
        )

    def _random_select(
        self,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        num_classes: int,
    ) -> List[int]:
        """Random selection ignoring scores.

        Args:
            labels: [N] ground truth labels
            indices: [N] dataset indices
            total_samples: Total samples to select. When per_class=True, divided among classes.
            per_class: If True, divide total_samples evenly among classes
            num_classes: Total classes

        Returns:
            List of selected dataset indices

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """
        selected = []

        if per_class:
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True"
                )
            samples_per_class = total_samples // num_classes

            for c in range(num_classes):
                class_mask = labels == c
                class_positions = torch.where(class_mask)[0]
                n_select = min(samples_per_class, len(class_positions))
                perm = torch.randperm(len(class_positions))[:n_select]
                selected_positions = class_positions[perm]
                selected.extend(indices[selected_positions].cpu().numpy().tolist())
        else:
            perm = torch.randperm(len(indices))[:total_samples]
            selected = indices[perm].cpu().numpy().tolist()
            self._log_global_class_counts(labels, perm, num_classes, "Random selection")

        logger.info(f"Random selection: {len(selected)} samples")
        return selected

    def _score_based_select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
    ) -> List[int]:
        """Score-based selection (min/max/mid/ccs).

        Args:
            scores: [N] scores
            labels: [N] labels
            indices: [N] dataset indices
            total_samples: Total samples to select. When per_class=True, divided among classes.
            per_class: If True, use class-proportional budgets (CCS-CP algorithm)
            mode: 'min', 'max', 'mid', or 'ccs'
            num_classes: Total classes

        Returns:
            List of selected dataset indices
        """
        # Import CCS selection function and proportional budget allocation
        from pruning.functional import compute_class_proportional_budgets, select_by_ccs

        selected = []

        # CCS-CP: Always use class-proportional budgets (ignores per_class flag)
        # This is a dynamic quota method from Tsai et al., ICCV 2025 Workshop
        if mode == "ccs":
            # Build label_to_indices map for budget computation
            label_to_indices = {}
            for i, label in enumerate(labels.cpu().numpy()):
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(i)

            # Compute class-proportional budgets (Algorithm 1 from CCS-CP paper)
            class_budgets = compute_class_proportional_budgets(
                total_samples,
                label_to_indices,
                min_samples_per_class=self.ccscp_min_samples,
            )

            logger.info(
                f"CCS-CP selection: total={total_samples}, "
                f"min_per_class={self.ccscp_min_samples}"
            )
            logger.info(
                f"  CCS params: mislabel_ratio={self.mislabel_ratio}, "
                f"num_strata={self.num_strata}"
            )
            logger.info(
                f"  Class budgets (proportional): {dict(sorted(class_budgets.items()))}"
            )

            for c in range(num_classes):
                class_mask = labels == c
                class_scores = scores[class_mask]
                class_positions = torch.where(class_mask)[0]

                if len(class_positions) == 0:
                    continue

                n_select = class_budgets.get(c, 0)
                if n_select == 0:
                    continue

                # Apply CCS stratified sampling within this class
                local_selected = select_by_ccs(
                    class_scores,
                    n_select,
                    mislabel_ratio=self.mislabel_ratio,
                    num_strata=self.num_strata,
                )

                # Map back to global positions
                selected_positions = class_positions[local_selected]
                selected.extend(indices[selected_positions].cpu().numpy().tolist())

                selected_class_scores = class_scores[local_selected]
                logger.info(
                    f"  Class {c}: budget={n_select}, selected={len(local_selected)}, "
                    f"score range [{selected_class_scores.min():.4f}, "
                    f"{selected_class_scores.max():.4f}]"
                )

        elif per_class:
            # Standard min/max/mid selection with uniform budgets
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True with mode={mode}"
                )
            samples_per_class = total_samples // num_classes

            logger.info(
                f"Per-class selection: {samples_per_class} samples per class (mode={mode}), "
                f"total={total_samples}"
            )

            for c in range(num_classes):
                class_mask = labels == c
                class_scores = scores[class_mask]
                class_positions = torch.where(class_mask)[0]

                if len(class_positions) == 0:
                    continue

                # Sort by score (ascending)
                sorted_local = torch.argsort(class_scores)
                n_select = min(samples_per_class, len(class_positions))

                if mode == "min":
                    selected_sorted = sorted_local[:n_select]
                elif mode == "max":
                    selected_sorted = sorted_local[-n_select:]
                elif mode == "mid":
                    median_idx = len(class_positions) // 2
                    start_idx = max(0, median_idx - n_select // 2)
                    end_idx = start_idx + n_select
                    if end_idx > len(class_positions):
                        end_idx = len(class_positions)
                        start_idx = max(0, end_idx - n_select)
                    selected_sorted = sorted_local[start_idx:end_idx]
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                selected_positions = class_positions[selected_sorted]
                selected.extend(indices[selected_positions].cpu().numpy().tolist())

                logger.info(
                    f"  Class {c}: selected {n_select}, "
                    f"score range [{class_scores[selected_sorted[0]]:.4f}, "
                    f"{class_scores[selected_sorted[-1]]:.4f}]"
                )
        else:
            # Global selection (min/max/mid/random)
            logger.info(
                f"Global selection: {total_samples} samples total (mode={mode})"
            )
            sorted_positions = torch.argsort(scores)

            if mode == "min":
                selected_sorted = sorted_positions[:total_samples]
            elif mode == "max":
                selected_sorted = sorted_positions[-total_samples:]
            elif mode == "mid":
                median_idx = len(scores) // 2
                start_idx = max(0, median_idx - total_samples // 2)
                end_idx = start_idx + total_samples
                if end_idx > len(scores):
                    end_idx = len(scores)
                    start_idx = max(0, end_idx - total_samples)
                selected_sorted = sorted_positions[start_idx:end_idx]
            elif mode == "random":
                random_indices = torch.randperm(len(scores))[:total_samples]
                selected_sorted = random_indices
            else:
                raise ValueError(f"Unknown mode: {mode}")

            selected = indices[selected_sorted].cpu().numpy().tolist()
            self._log_global_class_counts(
                labels, selected_sorted, num_classes, f"Global {mode}"
            )

        logger.info(f"Total selected: {len(selected)} samples")
        return selected

    def _log_global_class_counts(
        self,
        labels: torch.Tensor,
        selected_positions: torch.Tensor,
        num_classes: int,
        header: str,
    ):
        """Log per-class counts for global selection."""
        selected_labels = labels[selected_positions]
        counts = torch.bincount(selected_labels.cpu(), minlength=num_classes).tolist()

        logger.info(f"{header} per-class counts:")
        for c, count in enumerate(counts):
            logger.info(f"  Class {c}: {count} samples")


# ============================================================================
# Registry
# ============================================================================

SCORER_REGISTRY: Dict[str, type] = {}


def register_scorer(cls):
    """Decorator to register a scorer class."""
    SCORER_REGISTRY[cls.name] = cls
    for alias in cls.aliases:
        SCORER_REGISTRY[alias] = cls
    return cls


def get_scorer(name: str, model: nn.Module, cfg, **kwargs) -> BaseScorer:
    """Factory function to create scorer by name.

    Args:
        name: Scorer name (or alias)
        model: Teacher model
        cfg: OpenPoint config
        **kwargs: Additional args passed to scorer __init__

    Returns:
        Initialized scorer instance
    """
    name_lower = name.lower()
    if name_lower not in SCORER_REGISTRY:
        available = sorted(set(SCORER_REGISTRY.keys()))
        raise ValueError(f"Unknown scorer '{name}'. Available: {available}")
    return SCORER_REGISTRY[name_lower](model, cfg, **kwargs)


def list_scorers() -> List[str]:
    """List all registered scorer names (excluding aliases)."""
    seen = set()
    names = []
    for name, cls in SCORER_REGISTRY.items():
        if cls.name not in seen:
            names.append(cls.name)
            seen.add(cls.name)
    return sorted(names)


# ============================================================================
# Mixins for shared functionality
# ============================================================================


class FeatureExtractorMixin:
    """Mixin for scorers that need encoder features (herding, kcenter, submodular)."""

    def extract_features(
        self, dataloader, desc: str = "Extracting features"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract encoder features for all samples.

        Args:
            dataloader: Non-shuffled dataloader
            desc: Progress bar description

        Returns:
            features: [N, D] tensor
            labels: [N] tensor
            indices: [N] tensor
        """
        self.model.eval()

        all_features = []
        all_labels = []
        all_indices = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            with torch.no_grad():
                features = self.get_encoder_features(data)

            all_features.append(features.cpu())
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)

        return torch.cat(all_features), torch.cat(all_labels), torch.cat(all_indices)


class ClassNamesMixin:
    """Mixin for getting dataset class names."""

    def get_class_names(self, dataset, num_classes: int) -> List[str]:
        """Get class names from dataset or fallback to generic names."""
        from torch.utils.data import Subset

        from utils.constants import get_class_names

        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

        candidates = [
            getattr(base_dataset, "classes", None),
            getattr(base_dataset, "class_names", None),
        ]

        for names in candidates:
            if names is None:
                continue
            if callable(names):
                try:
                    names = names()
                except TypeError:
                    continue
            if isinstance(names, dict):
                names = [names.get(i, f"class_{i}") for i in range(num_classes)]
            if isinstance(names, (list, tuple)) and len(names) >= num_classes:
                return [str(names[i]) for i in range(num_classes)]

        return get_class_names(num_classes)


# ============================================================================
# Concrete Scorers
# ============================================================================


@register_scorer
class LossScorer(BaseScorer):
    """Loss-based scoring using head 0 only.

    Supports multiple loss types:
    - "ce": Cross-entropy (default)
    - "focal": Focal loss (down-weights easy samples)
    - "cb": Class-balanced loss (weights by effective sample count)

    Lower loss = easier sample, higher loss = harder sample.
    Use mode='min' for easy samples, mode='max' for hard samples.
    """

    name: str = "loss"
    aliases: List[str] = ["ce", "cross_entropy", "focal", "cb"]
    mode_override: Optional[str] = None  # User chooses min/max/mid
    requires_grad: bool = False

    def _get_class_counts(self, dataloader) -> torch.Tensor:
        """Count samples per class for CB loss."""
        from torch.utils.data import Subset

        dataset = dataloader.dataset
        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

        # Try to get labels
        labels = None
        for attr in ["targets", "labels", "label"]:
            if hasattr(base_dataset, attr):
                labels = getattr(base_dataset, attr)
                break

        if labels is None:
            # Fallback: iterate through dataset
            logger.info("Computing class counts by iterating dataset...")
            labels = []
            for i in range(len(dataset)):
                sample = dataset[i]
                labels.append(
                    sample["y"].item()
                    if isinstance(sample["y"], torch.Tensor)
                    else sample["y"]
                )
            labels = torch.tensor(labels)
        else:
            labels = (
                torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
            )

        # Handle Subset indices
        if isinstance(dataset, Subset):
            labels = labels[dataset.indices]

        counts = torch.bincount(labels, minlength=self.cfg.num_classes)
        return counts.float()

    def _build_criterion(
        self, loss_type: str, dataloader, focal_gamma: float, cb_beta: float
    ):
        """Build criterion based on loss_type."""
        from openpoints.loss.build import CBCrossEntropyLoss, FocalLoss

        if loss_type == "focal":
            logger.info(f"Using FocalLoss (gamma={focal_gamma})")
            return FocalLoss(gamma=focal_gamma)
        elif loss_type == "cb":
            logger.info(f"Using CBCrossEntropyLoss (beta={cb_beta})")
            criterion = CBCrossEntropyLoss(beta=cb_beta, reduction="none")
            class_counts = self._get_class_counts(dataloader)
            criterion.set_class_counts(class_counts)
            logger.info(f"  Class counts: {class_counts.tolist()}")
            return criterion
        else:  # "ce" or default
            logger.info("Using CrossEntropyLoss")
            return None  # Use F.cross_entropy directly

    @torch.no_grad()
    def compute(
        self,
        dataloader,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        cb_beta: float = 0.9999,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample loss scores.

        Args:
            dataloader: Non-shuffled dataloader
            loss_type: "ce" (default), "focal", or "cb"
            focal_gamma: Focal loss gamma parameter (default: 2.0)
            cb_beta: CB loss beta parameter (default: 0.9999)
            **kwargs: Unused (for interface compatibility)

        Returns:
            scores: [N] loss per sample
            labels: [N] ground truth labels
            indices: [N] dataset indices
        """
        self.model.eval()

        # Build criterion
        criterion = self._build_criterion(loss_type, dataloader, focal_gamma, cb_beta)

        all_scores = []
        all_labels = []
        all_indices = []

        logger.info(f"Computing {loss_type} loss scores...")
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Scoring samples ({loss_type})",
        )

        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            # Get logits from head 0 only
            logits = self.get_single_head_logits(data)

            # Per-sample loss
            if criterion is None:
                loss = F.cross_entropy(logits, target, reduction="none")
            else:
                loss = criterion(logits, target)
                # Ensure per-sample output
                if loss.dim() == 0:
                    loss = loss.unsqueeze(0)

            all_scores.append(loss.cpu())
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        indices = torch.cat(all_indices)

        logger.info(
            f"Loss statistics: mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"min={scores.min():.4f}, max={scores.max():.4f}"
        )

        return scores, labels, indices


@register_scorer
class HerdingScorer(FeatureExtractorMixin, ClassNamesMixin, BaseScorer):
    """Feature-based herding selection.

    Iteratively selects samples to match class mean in feature space.
    Selected samples get score 1.0, others get -100.0.
    Always use mode='max' for selection.
    """

    name: str = "herding"
    aliases: List[str] = []
    mode_override: Optional[str] = "max"  # Selected samples have score 1.0
    requires_grad: bool = False
    is_score_based: bool = False  # Selection done in compute(), returns binary scores

    @torch.no_grad()
    def compute(
        self,
        dataloader,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute herding-based selection scores.

        Args:
            dataloader: Non-shuffled dataloader
            total_samples: Total number of samples to select. When per_class=True,
                          divided evenly among classes.
            per_class: If True, divide total_samples evenly among classes
            num_classes: Total number of classes

        Returns:
            scores: [N] tensor (1.0 for selected, -100.0 for not)
            labels: [N] ground truth labels
            indices: [N] dataset indices

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """
        # Extract features using mixin
        features, labels, indices = self.extract_features(
            dataloader, desc="Extracting features for herding"
        )
        logger.info(f"Extracted features: {features.shape}")

        class_names = self.get_class_names(dataloader.dataset, num_classes)
        scores = torch.full((len(features),), -100.0)

        if per_class:
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True"
                )
            samples_per_class = total_samples // num_classes

            logger.info(
                f"Applying herding per class: {samples_per_class} samples per class, total={total_samples}"
            )
            for c in range(num_classes):
                class_name = class_names[c] if c < len(class_names) else f"class_{c}"
                class_mask = labels == c
                class_features = features[class_mask]
                class_positions = torch.where(class_mask)[0]

                if len(class_features) == 0:
                    logger.warning(f"  Class {c} ({class_name}): no samples found")
                    continue

                # Herding selection
                n_select = min(samples_per_class, len(class_features))
                selected_in_class = self._herding_select(class_features, n_select)

                # Mark selected samples
                for j in selected_in_class:
                    global_idx = class_positions[j].item()
                    scores[global_idx] = 1.0

                logger.info(
                    f"  Class {c} ({class_name}): selected {len(selected_in_class)} samples"
                )
        else:
            logger.info(f"Applying global herding: {total_samples} samples total")
            n_select = min(total_samples, len(features))
            selected = self._herding_select(features, n_select)

            for j in selected:
                scores[j] = 1.0

            self._log_global_class_counts(
                labels, scores == 1.0, num_classes, class_names
            )

        num_selected = (scores == 1.0).sum().item()
        logger.info(f"Herding selected {num_selected} samples total")

        return scores, labels, indices

    def _herding_select(self, features: torch.Tensor, n_select: int) -> List[int]:
        """Core herding algorithm."""
        mu = features.mean(0)
        res = mu.clone()
        sum_sel = torch.zeros_like(mu)

        selected = []
        for t in range(1, n_select + 1):
            scores_iter = features @ res
            j = scores_iter.argmax().item()
            selected.append(j)

            sum_sel += features[j]
            res = mu - sum_sel / t

        return selected

    def _log_global_class_counts(self, labels, mask, num_classes, class_names):
        """Log per-class counts for global selection."""
        selected_labels = labels[mask]
        counts = torch.bincount(selected_labels, minlength=num_classes).tolist()

        logger.info("Global herding per-class counts:")
        for c, count in enumerate(counts):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            logger.info(f"  Class {c} ({name}): {count} samples")

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        **kwargs,
    ) -> List[int]:
        """Binary selection: return indices where score == 1.0.

        Selection already done in compute(), this just extracts the indices.
        The total_samples parameter is not used here since selection was done in compute().
        """
        selected_mask = scores == 1.0
        return indices[selected_mask].cpu().numpy().tolist()


@register_scorer
class SubmodularCosineScorer(FeatureExtractorMixin, ClassNamesMixin, BaseScorer):
    """Facility-location (submodular) selection with cosine similarity.

    Maximizes coverage diversity using cosine similarity.
    Selected samples get score 1.0, others get -100.0.
    """

    name: str = "submodular_cosine"
    aliases: List[str] = ["cosine_submodular", "cosine_facility"]
    mode_override: Optional[str] = "max"
    requires_grad: bool = False
    is_score_based: bool = False  # Selection done in compute(), returns binary scores

    @torch.no_grad()
    def compute(
        self,
        dataloader,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute submodular cosine selection scores.

        Args:
            dataloader: Non-shuffled dataloader
            total_samples: Total number of samples to select. When per_class=True,
                          divided evenly among classes.
            per_class: If True, divide total_samples evenly among classes
            num_classes: Total number of classes

        Returns:
            scores, labels, indices

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """
        # Extract features
        features, labels, indices = self.extract_features(
            dataloader, desc="Extracting features for submodular cosine"
        )

        # Move to GPU and normalize
        features = features.to(self.device)
        labels = labels.to(self.device)
        features = F.normalize(features, dim=1)

        logger.info("Building cosine similarity matrix...")
        sim_matrix = features @ features.T  # [N, N]
        sim_matrix = torch.clamp(sim_matrix, min=0.0)  # Clip negative similarities

        num_total = features.shape[0]
        scores = torch.full((num_total,), -100.0, device=self.device)
        covered = torch.zeros(num_total, device=self.device)
        class_names = self.get_class_names(dataloader.dataset, num_classes)

        if per_class:
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True"
                )
            samples_per_class = total_samples // num_classes

            logger.info(
                f"Per-class submodular cosine: {samples_per_class} per class, total={total_samples}"
            )
            class_indices = {
                c: torch.nonzero(labels == c, as_tuple=False).squeeze(1)
                for c in range(num_classes)
            }

            for c in range(num_classes):
                cand_idx = class_indices[c]
                if cand_idx.numel() == 0:
                    continue

                n_pick = min(samples_per_class, cand_idx.numel())
                picked, covered = self._greedy_select(
                    cand_idx, n_pick, covered, sim_matrix
                )

                if picked:
                    picked_tensor = torch.tensor(
                        picked, device=self.device, dtype=torch.long
                    )
                    scores[picked_tensor] = 1.0

                class_name = class_names[c] if c < len(class_names) else f"class_{c}"
                logger.info(f"  Class {c} ({class_name}): selected {len(picked)}")
        else:
            logger.info(f"Global submodular cosine: {total_samples} total")
            all_candidates = torch.arange(num_total, device=self.device)
            picked, covered = self._greedy_select(
                all_candidates, total_samples, covered, sim_matrix
            )

            if picked:
                picked_tensor = torch.tensor(
                    picked, device=self.device, dtype=torch.long
                )
                scores[picked_tensor] = 1.0

            self._log_global_class_counts(
                labels, scores == 1.0, num_classes, class_names
            )

        num_selected = (scores == 1.0).sum().item()
        logger.info(f"Submodular cosine selected {num_selected} samples total")

        return scores.cpu(), labels.cpu(), indices

    def _greedy_select(self, candidate_indices, n_select, covered, sim_matrix):
        """Greedy facility-location selection."""
        selected = []
        if len(candidate_indices) == 0 or n_select == 0:
            return selected, covered

        local_mask = torch.zeros(
            len(candidate_indices), dtype=torch.bool, device=self.device
        )

        for _ in range(n_select):
            sim_cols = sim_matrix[:, candidate_indices]
            gains = torch.sum(
                torch.clamp(sim_cols - covered.unsqueeze(1), min=0.0), dim=0
            )
            gains = gains.masked_fill(local_mask, -float("inf"))

            best_local = torch.argmax(gains).item()
            best_gain = gains[best_local].item()

            if best_gain <= 0:
                break

            global_idx = candidate_indices[best_local].item()
            selected.append(global_idx)
            local_mask[best_local] = True

            covered = torch.maximum(covered, sim_matrix[:, global_idx])

        return selected, covered

    def _log_global_class_counts(self, labels, mask, num_classes, class_names):
        """Log per-class counts."""
        selected_labels = labels[mask]
        counts = torch.bincount(selected_labels, minlength=num_classes).tolist()

        logger.info("Global submodular cosine per-class counts:")
        for c, count in enumerate(counts):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            logger.info(f"  Class {c} ({name}): {count} samples")

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        **kwargs,
    ) -> List[int]:
        """Binary selection: return indices where score == 1.0."""
        selected_mask = scores == 1.0
        return indices[selected_mask].cpu().numpy().tolist()


@register_scorer
class KMeansPrototypeScorer(FeatureExtractorMixin, ClassNamesMixin, BaseScorer):
    """K-means clustering + closest sample to each center.

    This is NOT the K-Center Greedy algorithm from Sener & Savarese.
    It clusters features with K-Means, then selects samples closest to centroids.

    Selected samples get score 1.0, others get -100.0.
    Always use mode='max' for selection.
    """

    name: str = "kmeans_prototype"
    aliases: List[str] = ["kmeans", "kmeans-prototype"]
    mode_override: Optional[str] = "max"
    requires_grad: bool = False
    is_score_based: bool = False  # Selection done in compute(), returns binary scores

    @torch.no_grad()
    def compute(
        self,
        dataloader,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute k-means prototype selection scores.

        Args:
            dataloader: Non-shuffled dataloader
            total_samples: Total number of clusters (= samples to select). When per_class=True,
                          divided evenly among classes.
            per_class: If True, divide total_samples evenly among classes
            num_classes: Total number of classes

        Returns:
            scores: [N] tensor (1.0 for selected, -100.0 for not)
            labels: [N] ground truth labels
            indices: [N] dataset indices

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans

        # Extract features using mixin
        features, labels, indices = self.extract_features(
            dataloader, desc="Extracting features for k-means prototype"
        )
        logger.info(f"Extracted features: {features.shape}")

        class_names = self.get_class_names(dataloader.dataset, num_classes)
        scores = torch.full((len(features),), -100.0)

        if per_class:
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True"
                )
            samples_per_class = total_samples // num_classes

            logger.info(
                f"Applying k-means prototype per class: {samples_per_class} clusters per class, "
                f"total={total_samples}"
            )
            for c in range(num_classes):
                class_name = class_names[c] if c < len(class_names) else f"class_{c}"
                class_mask = labels == c
                class_features = features[class_mask]
                class_positions = torch.where(class_mask)[0]

                if len(class_features) == 0:
                    logger.warning(f"  Class {c} ({class_name}): no samples found")
                    continue

                # K-means clustering
                n_clusters = min(samples_per_class, len(class_features))
                features_np = class_features.numpy()

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                kmeans.fit(features_np)

                # Find closest sample to each center (greedy assignment to avoid duplicates)
                distances = cdist(
                    kmeans.cluster_centers_, features_np, metric="euclidean"
                )
                selected_in_class = self._greedy_assign_centers(distances, n_clusters)

                for j in selected_in_class:
                    global_idx = class_positions[j].item()
                    scores[global_idx] = 1.0

                logger.info(
                    f"  Class {c} ({class_name}): selected {len(selected_in_class)} from {n_clusters} centers"
                )
        else:
            logger.info(
                f"Applying global k-means prototype: {total_samples} clusters total"
            )
            n_clusters = min(total_samples, len(features))
            features_np = features.numpy()

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            kmeans.fit(features_np)

            # Greedy assignment to avoid duplicates
            distances = cdist(kmeans.cluster_centers_, features_np, metric="euclidean")
            selected_indices = self._greedy_assign_centers(distances, n_clusters)

            for j in selected_indices:
                scores[j] = 1.0

            logger.info(
                f"Selected {len(selected_indices)} unique samples from {n_clusters} centers"
            )
            self._log_global_class_counts(
                labels, scores == 1.0, num_classes, class_names
            )

        num_selected = (scores == 1.0).sum().item()
        logger.info(f"K-means prototype selected {num_selected} samples total")

        return scores, labels, indices

    def _log_global_class_counts(self, labels, mask, num_classes, class_names):
        """Log per-class counts for global selection."""
        selected_labels = labels[mask]
        counts = torch.bincount(selected_labels, minlength=num_classes).tolist()

        logger.info("Global k-means prototype per-class counts:")
        for c, count in enumerate(counts):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            logger.info(f"  Class {c} ({name}): {count} samples")

    def _greedy_assign_centers(self, distances: np.ndarray, n_select: int) -> List[int]:
        """Greedy assignment of centers to samples, avoiding duplicates.

        Each center is assigned to its closest sample that hasn't been
        selected yet. This guarantees exactly n_select unique samples
        (assuming n_samples >= n_select).

        Args:
            distances: [n_centers, n_samples] distance matrix
            n_select: Number of samples to select (usually == n_centers)

        Returns:
            List of selected sample indices (length == n_select)
        """
        n_centers, n_samples = distances.shape

        # Sort sample indices by distance for each center
        sorted_indices = np.argsort(distances, axis=1)  # [n_centers, n_samples]

        selected = []
        selected_set = set()

        for center_idx in range(min(n_select, n_centers)):
            # Find closest sample not already selected
            for sample_idx in sorted_indices[center_idx]:
                if sample_idx not in selected_set:
                    selected.append(sample_idx)
                    selected_set.add(sample_idx)
                    break

        return selected

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        **kwargs,
    ) -> List[int]:
        """Binary selection: return indices where score == 1.0.

        Selection already done in compute(), this just extracts the indices.
        The total_samples parameter is not used here since selection was done in compute().
        """
        selected_mask = scores == 1.0
        return indices[selected_mask].cpu().numpy().tolist()


@register_scorer
class KCenterGreedyScorer(FeatureExtractorMixin, ClassNamesMixin, BaseScorer):
    """K-Center Greedy (Coreset) selection from Sener & Savarese, ICLR 2018.

    Core idea: Minimize the maximum distance from any point to its nearest selected point.
    Algorithm: Greedily select the point farthest from the current selected set.

    Properties:
    - Finds "outliers" and "uncovered regions" in feature space
    - In imbalanced settings, tends to select more tail-class samples
      (since they are often far from head-class clusters)
    - Sensitive to noise/outliers (may select noisy samples)

    Reference: "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
    """

    name: str = "kcenter_greedy"
    aliases: List[str] = ["kcenter", "k-center", "coreset"]
    mode_override: Optional[str] = "kcenter"  # Special mode, selection done in compute
    requires_grad: bool = False
    is_score_based: bool = False  # Selection done in compute(), returns binary scores

    @torch.no_grad()
    def compute(
        self,
        dataloader,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features and perform K-Center Greedy selection.

        Args:
            dataloader: Non-shuffled dataloader
            total_samples: Total number of samples to select. When per_class=True,
                          divided evenly among classes.
            per_class: If True, divide total_samples evenly among classes
            num_classes: Total number of classes

        Returns:
            scores: [N] tensor (1.0 for selected, -100.0 for not)
            labels: [N] ground truth labels
            indices: [N] dataset indices

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """

        # Extract features
        features, labels, indices = self.extract_features(
            dataloader, desc="Extracting features for K-Center Greedy"
        )
        logger.info(f"Extracted features: {features.shape}")

        class_names = self.get_class_names(dataloader.dataset, num_classes)
        scores = torch.full((len(features),), -100.0)

        if per_class:
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True"
                )
            samples_per_class = total_samples // num_classes

            logger.info(
                f"Applying K-Center Greedy per class: {samples_per_class} samples per class, "
                f"total={total_samples}"
            )
            for c in range(num_classes):
                class_name = class_names[c] if c < len(class_names) else f"class_{c}"
                class_mask = labels == c
                class_features = features[class_mask]
                class_positions = torch.where(class_mask)[0]

                if len(class_features) == 0:
                    logger.warning(f"  Class {c} ({class_name}): no samples found")
                    continue

                n_select = min(samples_per_class, len(class_features))
                selected_idx = self._kcenter_greedy(class_features, n_select)

                for idx in selected_idx:
                    global_idx = class_positions[idx].item()
                    scores[global_idx] = 1.0

                logger.info(
                    f"  Class {c} ({class_name}): selected {n_select}/{len(class_features)} samples"
                )
        else:
            logger.info(
                f"Applying global K-Center Greedy: {total_samples} samples total"
            )
            n_select = min(total_samples, len(features))
            selected_idx = self._kcenter_greedy(features, n_select)

            for idx in selected_idx:
                scores[idx] = 1.0

            logger.info(f"Selected {n_select} samples globally")
            self._log_global_class_counts(
                labels, scores == 1.0, num_classes, class_names
            )

        num_selected = (scores == 1.0).sum().item()
        logger.info(f"K-Center Greedy selected {num_selected} samples total")

        return scores, labels, indices

    def _kcenter_greedy(self, features: torch.Tensor, n_select: int) -> List[int]:
        """K-Center Greedy algorithm (memory-efficient version).

        Each iteration selects the point with maximum distance to the nearest
        already-selected point. This ensures coverage of the feature space.

        Memory-efficient: O(N) instead of O(N^2) by computing distances incrementally.
        At 55k samples, uses ~0.1 GB instead of ~11 GB.

        Args:
            features: [N, D] feature vectors
            n_select: Number of samples to select

        Returns:
            List of selected indices
        """
        n_samples = len(features)
        device = features.device

        # Normalize features for cosine-based distance
        features = F.normalize(features, dim=1)

        selected_indices = []

        # Track minimum distance to any selected point for each sample
        # Memory: O(N) instead of O(N^2)
        min_distances = torch.full((n_samples,), float("inf"), device=device)

        # Initialize: select first point
        first_idx = 0
        selected_indices.append(first_idx)

        # Compute distances from first point to all others
        # For normalized vectors: ||a-b||^2 = 2 - 2*cos(a,b)
        first_feat = features[first_idx : first_idx + 1]  # [1, D]
        dists = 2 - 2 * (features @ first_feat.T).squeeze()  # [N]
        min_distances = torch.minimum(min_distances, dists)
        min_distances[first_idx] = float("-inf")  # Mark as selected

        for _ in range(1, n_select):
            # Select point with maximum min-distance to selected set
            best_idx = min_distances.argmax().item()
            selected_indices.append(best_idx)
            min_distances[best_idx] = float("-inf")  # Mark as selected

            # Update min_distances with distances to new center
            new_feat = features[best_idx : best_idx + 1]  # [1, D]
            dists = 2 - 2 * (features @ new_feat.T).squeeze()  # [N]
            min_distances = torch.minimum(min_distances, dists)

        return selected_indices

    def _log_global_class_counts(self, labels, mask, num_classes, class_names):
        """Log per-class counts for global selection."""
        selected_labels = labels[mask]
        counts = torch.bincount(selected_labels, minlength=num_classes).tolist()

        logger.info("Global K-Center Greedy per-class counts:")
        for c, count in enumerate(counts):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            logger.info(f"  Class {c} ({name}): {count} samples")

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        **kwargs,
    ) -> List[int]:
        """Binary selection: return indices where score == 1.0.

        Selection already done in compute(), this just extracts the indices.
        The total_samples parameter is not used here since selection was done in compute().
        """
        selected_mask = scores == 1.0
        return indices[selected_mask].cpu().numpy().tolist()


@register_scorer
class EntropyScorer(BaseScorer):
    """Entropy scoring using head 0 only.

    Higher entropy = more uncertain prediction.
    Use mode='max' for uncertain samples, mode='min' for confident samples.
    """

    name: str = "entropy"
    aliases: List[str] = []
    mode_override: Optional[str] = None
    requires_grad: bool = False

    @torch.no_grad()
    def compute(
        self, dataloader, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample entropy scores."""
        self.model.eval()

        all_scores = []
        all_labels = []
        all_indices = []

        logger.info("Computing entropy scores (head 0 only)...")
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Scoring samples (entropy)",
        )

        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            logits = self.get_single_head_logits(data)
            probs = F.softmax(logits, dim=1)
            safe_probs = torch.clamp(probs, min=1e-12)
            entropy = -(safe_probs * safe_probs.log()).sum(dim=1)

            all_scores.append(entropy.cpu())
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        indices = torch.cat(all_indices)

        logger.info(
            f"Entropy statistics: mean={scores.mean():.4f}, std={scores.std():.4f}"
        )

        return scores, labels, indices


@register_scorer
class EL2NScorer(BaseScorer):
    """EL2N scoring (||one_hot - softmax||_2) using head 0 only.

    Higher EL2N = harder sample (further from correct prediction).
    """

    name: str = "el2n"
    aliases: List[str] = []
    mode_override: Optional[str] = None
    requires_grad: bool = False

    @torch.no_grad()
    def compute(
        self, dataloader, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute EL2N scores."""
        self.model.eval()

        all_scores = []
        all_labels = []
        all_indices = []

        logger.info("Computing EL2N scores (head 0 only)...")
        pbar = tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Scoring samples (el2n)"
        )

        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            logits = self.get_single_head_logits(data)
            probs = F.softmax(logits, dim=1)
            one_hot = F.one_hot(target, num_classes=self.cfg.num_classes).float()
            el2n = torch.norm(one_hot - probs, p=2, dim=1)

            all_scores.append(el2n.cpu())
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        indices = torch.cat(all_indices)

        logger.info(
            f"EL2N statistics: mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"min={scores.min():.4f}, max={scores.max():.4f}"
        )

        return scores, labels, indices


@register_scorer
class GradNormScorer(BaseScorer):
    """Gradient norm scoring using head 0 only.

    Higher gradient norm = sample has more influence on model.
    """

    name: str = "grad_norm"
    aliases: List[str] = ["gradnorm", "gradnd", "grad"]
    mode_override: Optional[str] = None
    requires_grad: bool = True  # Needs model gradients

    def compute(
        self, dataloader, grad_scope: str = "head", **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample gradient norm scores.

        Args:
            dataloader: Non-shuffled dataloader
            grad_scope: 'head' for classifier params only, 'all' for full model

        Returns:
            scores, labels, indices
        """
        self.model.eval()

        grad_params = self._get_grad_params(grad_scope)
        if not grad_params:
            raise ValueError("No trainable parameters found for grad-norm scoring")

        logger.info(f"Computing grad-norm scores (scope={grad_scope}, head 0 only)...")

        all_scores = []
        all_labels = []
        all_indices = []

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Scoring samples (grad-norm)",
        )

        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            # Per-sample gradient norm (process one at a time)
            for i in range(target.shape[0]):
                single_data = {k: v[i : i + 1] for k, v in data.items()}

                logits = self.get_single_head_logits(single_data)
                loss = F.cross_entropy(logits, target[i : i + 1], reduction="mean")

                grads = torch.autograd.grad(
                    loss, grad_params, retain_graph=False, allow_unused=True
                )
                grad_sq_sum = torch.tensor(0.0, device=logits.device)
                for g in grads:
                    if g is not None:
                        grad_sq_sum = grad_sq_sum + g.pow(2).sum()

                grad_norm = torch.sqrt(grad_sq_sum)

                all_scores.append(grad_norm.detach().cpu())
                all_labels.append(target[i].detach().cpu())
                all_indices.append(batch_indices[i].detach().cpu())

        scores = torch.stack(all_scores)
        labels = torch.stack(all_labels)
        indices = torch.stack(all_indices)

        logger.info(
            f"Grad-norm statistics: mean={scores.mean():.4f}, std={scores.std():.4f}"
        )

        return scores, labels, indices

    def _get_grad_params(self, scope: str) -> List[torch.Tensor]:
        """Get parameters for gradient computation."""
        if scope != "head":
            return [p for p in self.actual_model.parameters() if p.requires_grad]

        # Head parameters only
        if hasattr(self.actual_model, "prediction") and hasattr(
            self.actual_model.prediction, "head"
        ):
            head_attr = self.actual_model.prediction.head
            if isinstance(head_attr, nn.ModuleList):
                return [p for p in head_attr[0].parameters() if p.requires_grad]
            elif isinstance(head_attr, nn.Module):
                return [p for p in head_attr.parameters() if p.requires_grad]

        logger.warning("Head scope not found, falling back to all parameters")
        return [p for p in self.actual_model.parameters() if p.requires_grad]


@register_scorer
class GradHerdingScorer(ClassNamesMixin, BaseScorer):
    """Gradient-based herding: select samples matching class-wise mean gradients.

    Memory-efficient: processes one class at a time.
    Selected samples get score 1.0, others get -100.0.
    """

    name: str = "grad_herding"
    aliases: List[str] = ["gradherding", "grad_mean"]
    mode_override: Optional[str] = "max"
    requires_grad: bool = True

    def compute(
        self,
        dataloader,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        grad_scope: str = "head",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradient-herding selection scores.

        Args:
            dataloader: Non-shuffled dataloader
            total_samples: Total samples to select. Divided evenly among classes.
            per_class: Must be True for grad-herding (global selection not supported)
            num_classes: Total number of classes
            grad_scope: 'head' or 'all'

        Returns:
            scores, labels, indices

        Raises:
            ValueError: If per_class=False (not supported)
            ValueError: If total_samples is not divisible by num_classes
        """
        self.model.eval()

        if not per_class:
            raise ValueError(
                "GradHerdingScorer requires per_class=True. "
                "Global gradient herding (per_class=False) is not supported."
            )

        if total_samples % num_classes != 0:
            raise ValueError(
                f"total_samples ({total_samples}) must be divisible by "
                f"num_classes ({num_classes}) when per_class=True"
            )

        samples_per_class = total_samples // num_classes

        grad_params = self._get_grad_params(grad_scope)
        if not grad_params:
            raise ValueError("No trainable parameters found for grad-herding")

        flat_grad_len = sum(p.numel() for p in grad_params)
        logger.info(f"Grad-herding: scope={grad_scope}, flat_grad_len={flat_grad_len}")

        class_names = self.get_class_names(dataloader.dataset, num_classes)

        # Pass 1: Build class index mapping
        logger.info("Pass 1: Building class index mapping...")
        class_samples = {c: [] for c in range(num_classes)}
        all_labels = []
        all_indices = []

        for batch_idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Indexing samples"
        ):
            for key in data.keys():
                data[key] = data[key].to(self.device, non_blocking=True)

            target = data["y"]
            actual_batch_size = target.shape[0]
            start_idx = batch_idx * dataloader.batch_size
            batch_indices = torch.arange(start_idx, start_idx + actual_batch_size)

            for i in range(actual_batch_size):
                label = target[i].item()
                global_idx = batch_indices[i].item()
                class_samples[label].append((batch_idx, i, global_idx))
                all_labels.append(label)
                all_indices.append(global_idx)

        all_labels = torch.tensor(all_labels)
        all_indices = torch.tensor(all_indices)
        scores = torch.full((len(all_labels),), -100.0)

        # Pass 2: Process each class
        dataset = dataloader.dataset

        for c in range(num_classes):
            if len(class_samples[c]) == 0:
                logger.warning(f"  Class {c}: no samples")
                continue

            class_name = class_names[c] if c < len(class_names) else f"class_{c}"
            logger.info(
                f"  Class {c} ({class_name}): computing gradients for {len(class_samples[c])} samples..."
            )

            # Collect gradients for this class
            class_grads = []
            class_global_indices = []

            for batch_idx, sample_idx_in_batch, global_idx in tqdm(
                class_samples[c], desc=f"Class {c}", leave=False
            ):
                sample = dataset[global_idx]

                # Prepare single sample
                data = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.unsqueeze(0).to(self.device)
                    else:
                        data[key] = torch.tensor([value]).to(self.device)

                points = data["x"]
                data["pos"] = points[:, :, :3].contiguous()
                in_channels = getattr(self.cfg.model.encoder_args, "in_channels", 3)
                data["x"] = points[:, :, :in_channels].transpose(1, 2).contiguous()

                # Compute gradient
                logits = self.get_single_head_logits(data)
                loss = F.cross_entropy(logits, data["y"], reduction="mean")

                grads = torch.autograd.grad(
                    loss, grad_params, retain_graph=False, allow_unused=True
                )
                flat_chunks = []
                for g, p in zip(grads, grad_params):
                    if g is None:
                        flat_chunks.append(torch.zeros_like(p).flatten())
                    else:
                        flat_chunks.append(g.flatten())
                grad_vec = torch.cat(flat_chunks).detach().cpu()

                class_grads.append(grad_vec)
                class_global_indices.append(global_idx)

            # Herding selection
            class_grads = torch.stack(class_grads)
            n_select = min(samples_per_class, len(class_grads))

            mu = class_grads.mean(0)
            res = mu.clone()
            sum_sel = torch.zeros_like(mu)

            selected_in_class = []
            for t in range(1, n_select + 1):
                scores_iter = class_grads @ res
                j = scores_iter.argmax().item()
                selected_in_class.append(j)

                sum_sel += class_grads[j]
                res = mu - sum_sel / t

            # Mark selected
            for j in selected_in_class:
                global_idx = class_global_indices[j]
                pos = (all_indices == global_idx).nonzero(as_tuple=True)[0].item()
                scores[pos] = 1.0

            logger.info(
                f"  Class {c} ({class_name}): selected {len(selected_in_class)}/{len(class_grads)}"
            )

            # Free memory
            del class_grads
            torch.cuda.empty_cache()

        num_selected = (scores == 1.0).sum().item()
        logger.info(f"Grad-herding selected {num_selected} samples total")

        return scores, all_labels, all_indices

    def _get_grad_params(self, scope: str) -> List[torch.Tensor]:
        """Get parameters for gradient computation."""
        if scope != "head":
            return [p for p in self.actual_model.parameters() if p.requires_grad]

        if hasattr(self.actual_model, "prediction") and hasattr(
            self.actual_model.prediction, "head"
        ):
            head_attr = self.actual_model.prediction.head
            if isinstance(head_attr, nn.ModuleList):
                return [p for p in head_attr[0].parameters() if p.requires_grad]
            elif isinstance(head_attr, nn.Module):
                return [p for p in head_attr.parameters() if p.requires_grad]

        logger.warning("Head scope not found, falling back to all parameters")
        return [p for p in self.actual_model.parameters() if p.requires_grad]


@register_scorer
class SubmodularRBFScorer(FeatureExtractorMixin, ClassNamesMixin, BaseScorer):
    """Facility-location (submodular) selection with RBF kernel.

    Maximizes coverage diversity using RBF similarity.
    Selected samples get score 1.0, others get -100.0.

    Supports multiple representation spaces via `space` parameter:
    - "embedding": encoder features (default)
    - "logits": raw classifier outputs
    - "softmax": probability distribution
    """

    name: str = "submodular_rbf"
    aliases: List[str] = ["submodular", "facility", "facility_location"]
    mode_override: Optional[str] = "max"
    requires_grad: bool = False
    is_score_based: bool = False  # Selection done in compute(), returns binary scores

    @torch.no_grad()
    def compute(
        self,
        dataloader,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        sigma: float = None,
        space: str = "embedding",
        rbf_algorithm: str = "apricot",
        initial_subset: List[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute submodular RBF selection scores.

        Args:
            dataloader: Non-shuffled dataloader
            total_samples: Total number of samples to select. When per_class=True,
                          divided evenly among classes.
            per_class: If True, divide total_samples evenly among classes
            num_classes: Total number of classes
            sigma: RBF kernel bandwidth (required)
            space: Representation space - "embedding", "logits", or "softmax"
            rbf_algorithm: "apricot" (fast, recommended) or "original" (naive greedy)
            initial_subset: Pre-selected indices for warm-start (global selection only).
                           When provided, marginal gains are computed relative to this set.

        Returns:
            scores, labels, indices

        Raises:
            ValueError: If sigma is invalid, or if per_class=True and total_samples
                       is not divisible by num_classes
        """
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be a positive float for submodular_rbf scorer")

        # Validate algorithm choice
        if rbf_algorithm == "apricot" and not APRICOT_AVAILABLE:
            logger.warning(
                "apricot not installed, falling back to original algorithm. "
                "Install with: pip install apricot-select"
            )
            rbf_algorithm = "original"

        # Extract representations based on space
        if space == "embedding":
            features, labels, indices = self.extract_features(
                dataloader, desc="Extracting embeddings for submodular"
            )
        elif space in ("logits", "softmax"):
            features, labels, indices = self._extract_logits(
                dataloader, desc=f"Extracting {space} for submodular"
            )
            if space == "softmax":
                features = F.softmax(features, dim=1)
        else:
            raise ValueError(
                f"Unknown space '{space}'. Use 'embedding', 'logits', or 'softmax'"
            )

        logger.info(
            f"Submodular RBF: space='{space}', algorithm='{rbf_algorithm}', "
            f"features shape: {features.shape}"
        )

        # Normalize features (required for both algorithms)
        features = F.normalize(features, dim=1)
        num_total = features.shape[0]

        # Route to appropriate algorithm
        if rbf_algorithm == "apricot":
            return self._compute_apricot(
                features,
                labels,
                indices,
                total_samples,
                per_class,
                num_classes,
                sigma,
                initial_subset=initial_subset,
            )
        else:
            return self._compute_original(
                features, labels, indices, total_samples, per_class, num_classes, sigma
            )

    def _compute_apricot(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        sigma: float,
        initial_subset: List[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fast submodular selection using apricot library's lazy greedy.

        Args:
            initial_subset: Pre-selected indices for warm-start (global selection only).
                           Marginal gains are computed relative to this set.

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """
        num_total = features.shape[0]
        scores = torch.full((num_total,), -100.0)

        # Convert to numpy for apricot (it uses sklearn internally)
        features_np = features.cpu().numpy().astype(np.float64)
        labels_np = labels.cpu().numpy()

        # Compute RBF similarity matrix
        # sklearn rbf_kernel uses gamma = 1/(2*sigma^2)
        gamma = 1.0 / (2.0 * sigma * sigma)
        logger.info(
            f"Computing RBF similarity matrix (sigma={sigma}, gamma={gamma:.4f})..."
        )

        from sklearn.metrics.pairwise import rbf_kernel

        sim_matrix = rbf_kernel(features_np, gamma=gamma)

        class_names = self.get_class_names(None, num_classes)

        if per_class:
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True"
                )
            samples_per_class = total_samples // num_classes

            logger.info(
                f"Per-class submodular (apricot): {samples_per_class} per class, total={total_samples}"
            )

            for c in range(num_classes):
                class_mask = labels_np == c
                class_indices_local = np.where(class_mask)[0]

                if len(class_indices_local) == 0:
                    continue

                n_pick = min(samples_per_class, len(class_indices_local))

                # Extract class-specific similarity submatrix
                class_sim = sim_matrix[np.ix_(class_indices_local, class_indices_local)]

                # Run apricot on this class
                selector = FacilityLocationSelection(
                    n_samples=n_pick,
                    metric="precomputed",
                    optimizer="lazy",
                    verbose=False,
                )
                selector.fit(class_sim)

                # Map back to global indices
                picked_local = selector.ranking
                picked_global = class_indices_local[picked_local]

                scores[picked_global] = 1.0

                class_name = class_names[c] if c < len(class_names) else f"class_{c}"
                logger.info(
                    f"  Class {c} ({class_name}): selected {len(picked_global)}"
                )
        else:
            if initial_subset is not None:
                logger.info(
                    f"Global submodular (apricot) with warm-start: "
                    f"{total_samples} new + {len(initial_subset)} initial"
                )
            else:
                logger.info(f"Global submodular (apricot): {total_samples} total")

            selector = FacilityLocationSelection(
                n_samples=total_samples,
                metric="precomputed",
                optimizer="lazy",
                initial_subset=initial_subset,
                verbose=True,
            )
            selector.fit(sim_matrix)

            picked = list(selector.ranking)
            scores[picked] = 1.0

            # Also mark initial_subset as selected (score=1.0) if provided
            if initial_subset is not None:
                for idx in initial_subset:
                    scores[idx] = 1.0

            self._log_global_class_counts(
                labels, scores == 1.0, num_classes, class_names
            )

        num_selected = (scores == 1.0).sum().item()
        logger.info(f"Submodular RBF (apricot) selected {num_selected} samples total")

        return scores, labels.cpu(), indices

    def _compute_original(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        num_classes: int,
        sigma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original naive greedy implementation (slower, for debugging/comparison).

        Raises:
            ValueError: If per_class=True and total_samples is not divisible by num_classes
        """
        # Move to GPU
        features = features.to(self.device)
        labels = labels.to(self.device)

        logger.info(f"Building RBF similarity matrix (sigma={sigma})...")
        dist_sq = torch.cdist(features, features, p=2) ** 2
        sim_matrix = torch.exp(-dist_sq / (2.0 * sigma * sigma))

        num_total = features.shape[0]
        scores = torch.full((num_total,), -100.0, device=self.device)
        covered = torch.zeros(num_total, device=self.device)
        class_names = self.get_class_names(None, num_classes)

        if per_class:
            if total_samples % num_classes != 0:
                raise ValueError(
                    f"total_samples ({total_samples}) must be divisible by "
                    f"num_classes ({num_classes}) when per_class=True"
                )
            samples_per_class = total_samples // num_classes

            logger.info(
                f"Per-class submodular (original): {samples_per_class} per class, total={total_samples}"
            )
            class_indices = {
                c: torch.nonzero(labels == c, as_tuple=False).squeeze(1)
                for c in range(num_classes)
            }

            for c in range(num_classes):
                cand_idx = class_indices[c]
                if cand_idx.numel() == 0:
                    continue

                n_pick = min(samples_per_class, cand_idx.numel())
                picked, covered = self._greedy_select(
                    cand_idx, n_pick, covered, sim_matrix
                )

                if picked:
                    picked_tensor = torch.tensor(
                        picked, device=self.device, dtype=torch.long
                    )
                    scores[picked_tensor] = 1.0

                class_name = class_names[c] if c < len(class_names) else f"class_{c}"
                logger.info(f"  Class {c} ({class_name}): selected {len(picked)}")
        else:
            logger.info(f"Global submodular (original): {total_samples} total")
            all_candidates = torch.arange(num_total, device=self.device)
            picked, covered = self._greedy_select(
                all_candidates, total_samples, covered, sim_matrix
            )

            if picked:
                picked_tensor = torch.tensor(
                    picked, device=self.device, dtype=torch.long
                )
                scores[picked_tensor] = 1.0

            self._log_global_class_counts(
                labels, scores == 1.0, num_classes, class_names
            )

        num_selected = (scores == 1.0).sum().item()
        logger.info(f"Submodular RBF (original) selected {num_selected} samples total")

        return scores.cpu(), labels.cpu(), indices

    def _extract_logits(
        self, dataloader, desc: str = "Extracting logits"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract logits for all samples.

        Args:
            dataloader: Non-shuffled dataloader
            desc: Progress bar description

        Returns:
            logits: [N, num_classes] tensor
            labels: [N] tensor
            indices: [N] tensor
        """
        self.model.eval()

        all_logits = []
        all_labels = []
        all_indices = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            logits = self.get_single_head_logits(data)

            all_logits.append(logits.cpu())
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)

        return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_indices)

    def _greedy_select(self, candidate_indices, n_select, covered, sim_matrix):
        """Greedy facility-location selection."""
        selected = []
        if len(candidate_indices) == 0 or n_select == 0:
            return selected, covered

        local_mask = torch.zeros(
            len(candidate_indices), dtype=torch.bool, device=self.device
        )

        for _ in range(n_select):
            sim_cols = sim_matrix[:, candidate_indices]
            gains = torch.sum(
                torch.clamp(sim_cols - covered.unsqueeze(1), min=0.0), dim=0
            )
            gains = gains.masked_fill(local_mask, -float("inf"))

            best_local = torch.argmax(gains).item()
            best_gain = gains[best_local].item()

            if best_gain <= 0:
                break

            global_idx = candidate_indices[best_local].item()
            selected.append(global_idx)
            local_mask[best_local] = True

            covered = torch.maximum(covered, sim_matrix[:, global_idx])

        return selected, covered

    def _log_global_class_counts(self, labels, mask, num_classes, class_names):
        """Log per-class counts."""
        selected_labels = labels[mask]
        counts = torch.bincount(selected_labels, minlength=num_classes).tolist()

        logger.info("Global submodular per-class counts:")
        for c, count in enumerate(counts):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            logger.info(f"  Class {c} ({name}): {count} samples")

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        **kwargs,
    ) -> List[int]:
        """Binary selection: return indices where score == 1.0.

        Selection already done in compute(), this just extracts the indices.
        The total_samples parameter is not used here since selection was done in compute().
        """
        selected_mask = scores == 1.0
        return indices[selected_mask].cpu().numpy().tolist()


@register_scorer
class NUCSScorer(FeatureExtractorMixin, BaseScorer):
    """NUCS: Non-Uniform Class-wise Coreset Selection.

    Based on: "Non-Uniform Class-Wise Coreset Selection: Characterizing
    Category Difficulty for Data-Efficient Transfer Learning" (arXiv:2504.13234)

    Key differences from other scorers:
    - Non-uniform budget allocation (harder classes get more samples)
    - Window-based selection (not top-k/bottom-k)
    - Optional KRR endpoint optimization

    Uses EL2N scores as difficulty metric, then applies NUCS selection logic.
    """

    name: str = "nucs"
    aliases: List[str] = ["nucs_el2n", "non_uniform"]
    mode_override: Optional[str] = "nucs"  # Special marker for NUCS mode
    requires_grad: bool = False
    supports_hybrid: bool = False  # NUCS has its own budget allocation logic

    def __init__(self, model: nn.Module, cfg, device: str = "cuda"):
        super().__init__(model, cfg, device)
        self._features = None  # Cache features for KRR optimization
        self._label_to_indices = None  # Cache for NUCS selector

    @torch.no_grad()
    def compute(
        self, dataloader, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute EL2N scores (difficulty metric for NUCS).

        Args:
            dataloader: Non-shuffled dataloader
            **kwargs: Unused

        Returns:
            scores: [N] EL2N scores (higher = harder)
            labels: [N] ground truth labels
            indices: [N] dataset indices
        """
        self.model.eval()

        all_scores = []
        all_labels = []
        all_indices = []

        logger.info("NUCS: Computing EL2N scores...")
        pbar = tqdm(
            enumerate(dataloader), total=len(dataloader), desc="NUCS: Computing EL2N"
        )

        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            logits = self.get_single_head_logits(data)
            probs = F.softmax(logits, dim=1)
            one_hot = F.one_hot(target, num_classes=self.cfg.num_classes).float()
            el2n = torch.norm(one_hot - probs, p=2, dim=1)

            all_scores.append(el2n.cpu())
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        indices = torch.cat(all_indices)

        # Build label_to_indices mapping for NUCS
        self._label_to_indices = {}
        for i, label in enumerate(labels.tolist()):
            if label not in self._label_to_indices:
                self._label_to_indices[label] = []
            self._label_to_indices[label].append(i)

        logger.info(
            f"NUCS EL2N statistics: mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"min={scores.min():.4f}, max={scores.max():.4f}"
        )

        return scores, labels, indices

    def compute_with_features(
        self, dataloader, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute EL2N scores AND extract features for KRR optimization.

        Args:
            dataloader: Non-shuffled dataloader

        Returns:
            scores: [N] EL2N scores
            labels: [N] ground truth labels
            indices: [N] dataset indices
            features: [N, D] encoder features
        """
        self.model.eval()

        all_scores = []
        all_labels = []
        all_indices = []
        all_features = []

        logger.info("NUCS: Computing EL2N scores + extracting features...")
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="NUCS: EL2N + features",
        )

        for batch_idx, data in pbar:
            data = self.prepare_batch(data)
            target = data["y"]

            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )

            # Get logits and features
            logits = self.get_single_head_logits(data)
            features = self.get_encoder_features(data)

            # Compute EL2N
            probs = F.softmax(logits, dim=1)
            one_hot = F.one_hot(target, num_classes=self.cfg.num_classes).float()
            el2n = torch.norm(one_hot - probs, p=2, dim=1)

            all_scores.append(el2n.cpu())
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)
            all_features.append(features.cpu())

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        indices = torch.cat(all_indices)
        features = torch.cat(all_features)

        # Cache for select()
        self._features = features
        self._label_to_indices = {}
        for i, label in enumerate(labels.tolist()):
            if label not in self._label_to_indices:
                self._label_to_indices[label] = []
            self._label_to_indices[label].append(i)

        logger.info(
            f"NUCS: Extracted {features.shape[0]} samples with {features.shape[1]} features"
        )

        return scores, labels, indices, features

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        # NUCS-specific parameters
        nucs_aggregation: str = "mean",
        nucs_min_samples: int = 1,
        nucs_endpoint: Optional[float] = 0.75,
        nucs_use_krr: bool = False,
        nucs_endpoint_candidates: Optional[List[float]] = None,
        **kwargs,
    ) -> List[int]:
        """Select samples using NUCS non-uniform budget allocation + window selection.

        Ignores `per_class` and `mode` - NUCS always uses:
        - Non-uniform per-class allocation based on difficulty
        - Window-based selection (controlled by endpoint)

        Args:
            scores: [N] EL2N scores from compute()
            labels: [N] ground truth labels
            indices: [N] dataset indices
            total_samples: Total samples to select
            per_class: Ignored (NUCS always per-class)
            mode: Ignored (NUCS uses window selection)
            num_classes: Number of classes
            nucs_aggregation: How to aggregate scores to difficulty ("mean", "median", "p75")
            nucs_min_samples: Minimum samples per class
            nucs_endpoint: Window endpoint (0-1), higher = harder samples
            nucs_use_krr: If True, use KRR to find optimal endpoint (requires features)
            nucs_endpoint_candidates: Endpoints to test with KRR

        Returns:
            List of selected dataset indices
        """
        from pruning.baseline.nucs import NUCSSelector

        if self._label_to_indices is None:
            raise RuntimeError(
                "NUCS: label_to_indices not computed. Call compute() first."
            )

        # Log ignored parameters
        if mode != "nucs":
            logger.info(f"NUCS: Ignoring mode='{mode}', using window-based selection")
        if not per_class:
            logger.info(
                "NUCS: Ignoring per_class=False, using non-uniform per-class allocation"
            )

        # Create NUCS selector
        selector = NUCSSelector(
            num_classes=num_classes,
            total_samples=total_samples,
            aggregation=nucs_aggregation,
            min_samples_per_class=nucs_min_samples,
            endpoint_candidates=nucs_endpoint_candidates or [0.25, 0.5, 0.75, 1.0],
            fixed_endpoint=None if nucs_use_krr else nucs_endpoint,
        )

        # Run selection
        if nucs_use_krr and self._features is not None:
            logger.info("NUCS: Using KRR for endpoint optimization")
            selected_indices, info = selector.select(
                scores=scores,
                labels=labels,
                features=self._features,
                label_to_indices=self._label_to_indices,
            )
        else:
            if nucs_use_krr and self._features is None:
                logger.warning(
                    "NUCS: nucs_use_krr=True but features not available. "
                    "Use compute_with_features() to enable KRR. Falling back to fixed endpoint."
                )
            logger.info(f"NUCS: Using fixed endpoint={nucs_endpoint}")
            selected_indices, info = selector.select_without_krr(
                scores=scores,
                labels=labels,
                label_to_indices=self._label_to_indices,
                endpoint=nucs_endpoint,
            )

        # Log selection info
        logger.info(f"NUCS: Selected {info['total_selected']} samples")
        logger.info(f"NUCS: Endpoint used: {info['endpoint']}")
        logger.info("NUCS: Per-class budgets:")
        for c in sorted(info["budgets"].keys()):
            budget = info["budgets"][c]
            difficulty = info["difficulties"][c]
            class_size = len(self._label_to_indices.get(c, []))
            logger.info(
                f"  Class {c}: difficulty={difficulty:.4f}, budget={budget}/{class_size}"
            )

        # Convert from local indices to dataset indices
        selected_dataset_indices = indices[selected_indices].cpu().numpy().tolist()

        return selected_dataset_indices


@register_scorer
class DRoPScorer(BaseScorer):
    """DRoP: Distributionally Robust Data Pruning (Vysogorets et al., ICLR 2025).

    This is NOT a score-based method. DRoP:
    1. Computes per-class recall on validation set (for difficulty estimation)
    2. Allocates quotas: harder classes (lower recall) get more samples
    3. Random selection within each class based on quotas

    Config behavior:
    - Ignores `mode` (always random selection within class)
    - Ignores `per_class` (always stratified per-class)
    - Uses fixed metric: recall
    - Requires `val_loader` kwarg for computing per-class metrics
    """

    name: str = "drop"
    aliases: List[str] = ["drop_quota", "distributionally_robust"]
    mode_override: Optional[str] = "drop"  # Special marker for DRoP mode
    requires_grad: bool = False
    supports_hybrid: bool = False  # DRoP has its own quota-based allocation

    def __init__(self, model: nn.Module, cfg, device: str = "cuda", **kwargs):
        super().__init__(model, cfg, device)
        self._quotas = None  # Computed in compute(), used in select()
        self._class_sizes = None
        # Ignore extra kwargs (e.g., mislabel_ratio, num_strata)

    @torch.no_grad()
    def compute(
        self, dataloader, total_samples: int, num_classes: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DRoP quotas based on per-class recall on validation set.

        Note: This doesn't compute per-sample scores. Instead, it:
        1. Evaluates model on validation set to get per-class recall
        2. Computes DRoP quotas (harder classes get more samples)
        3. Returns dummy scores (selection done in select() via random)

        Args:
            dataloader: Training dataloader (for collecting labels/indices)
            total_samples: Total samples to select
            num_classes: Number of classes
            **kwargs: Must include `val_loader` for computing per-class metrics

        Returns:
            scores: Dummy tensor of zeros (not used for selection)
            labels: [N] ground truth labels
            indices: [N] dataset indices
        """
        from pruning.baseline.drop import (
            compute_drop_quota,
            compute_per_class_metrics,
            get_class_sizes,
        )

        # Get validation loader from kwargs (required for DRoP)
        val_loader = kwargs.get("val_loader")
        if val_loader is None:
            logger.warning(
                "DRoP: No val_loader provided, falling back to training set for metrics. "
                "This may result in overly optimistic difficulty estimates."
            )
            metrics_loader = dataloader
            eval_set_name = "training set"
        else:
            metrics_loader = val_loader
            eval_set_name = "validation set"

        logger.info("=" * 40)
        logger.info(f"DRoP: Computing per-class metrics on {eval_set_name}...")

        # Step 1: Compute per-class metrics on validation set
        class_metrics = compute_per_class_metrics(
            model=self.model,
            dataloader=metrics_loader,
            cfg=self.cfg,
            device=torch.device(self.device),
            num_classes=num_classes,
        )

        # Step 2: Get class sizes from dataset
        self._class_sizes = get_class_sizes(dataloader.dataset, num_classes)
        logger.info(f"Class sizes: {self._class_sizes}")

        # Step 3: Compute DRoP quotas (fixed metric: recall)
        self._quotas = compute_drop_quota(
            class_metrics=class_metrics,
            class_sizes=self._class_sizes,
            total_samples=total_samples,
            num_classes=num_classes,
            metric="recall",  # Fixed as per DRoP paper
        )

        logger.info("DRoP quotas computed. Selection will use random sampling.")
        logger.info("=" * 40)

        # Collect labels and indices (scores are dummy)
        all_labels = []
        all_indices = []

        for batch_idx, data in enumerate(dataloader):
            for key in data.keys():
                data[key] = data[key].to(self.device, non_blocking=True)
            target = data["y"]
            batch_indices = self.get_batch_indices(
                batch_idx, dataloader.batch_size, target.shape[0]
            )
            all_labels.append(target.cpu())
            all_indices.append(batch_indices)

        labels = torch.cat(all_labels)
        indices = torch.cat(all_indices)

        # Return dummy scores (zeros) - not used for selection
        scores = torch.zeros(len(labels))

        return scores, labels, indices

    def select(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        total_samples: int,
        per_class: bool,
        mode: str,
        num_classes: int,
        **kwargs,
    ) -> List[int]:
        """Select samples using DRoP quotas with random selection.

        Ignores `scores`, `per_class`, `mode`, and extra kwargs - DRoP always uses:
        - Quota-based per-class allocation
        - Random selection within each class
        """
        from pruning.baseline.drop import select_with_drop_quota

        if self._quotas is None:
            raise RuntimeError("DRoP quotas not computed. Call compute() first.")

        # Log that we're ignoring certain parameters
        if mode != "drop":
            logger.info(f"DRoP: Ignoring mode='{mode}', using random selection")
        if not per_class:
            logger.info("DRoP: Ignoring per_class=False, using stratified per-class")

        selected = select_with_drop_quota(
            scores=scores,  # Ignored internally
            labels=labels,
            indices=indices,
            quotas=self._quotas,
            total_samples=total_samples,
            num_classes=num_classes,
            mode="random",  # DRoP always uses random
        )

        return selected
