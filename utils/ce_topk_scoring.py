"""CE Top-K scoring utilities for dual-model sample selection.

This module implements teacher-student divergence scoring using the CE top-k
divergence metric from pruning/functional.py. It includes:
- Dual-model logits extraction
- CE top-k divergence computation with gating statistics
- Sample selection with random fallback for under-gated classes
- Comprehensive logging of gating and selection statistics
"""

import logging
import torch
from typing import Tuple, Dict
from tqdm import tqdm
from torch.utils.data import Subset

from pruning.functional import teacher_topk_ce_divergence
from utils.data_prep import prepare_batch
from utils.model_loading import get_head0_logits

logger = logging.getLogger(__name__)


def compute_ce_topk_scores_with_gating(
    teacher_model: torch.nn.Module,
    teacher_num_heads: int,
    student_model: torch.nn.Module,
    student_num_heads: int,
    dataloader: torch.utils.data.DataLoader,
    cfg,
    k: int = 5,
    temperature: float = 2.0,
    gate_temperature: float = 1.0,
    gate_threshold: float = 0.7,
    confidence_power: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Compute CE top-k divergence scores for all samples with gating statistics.

    This function scores each sample based on the divergence between teacher and
    student predictions on the teacher's top-k predicted classes. Samples where
    the teacher is uncertain (p_T(y) < gate_threshold) receive score=0.

    IMPORTANT: Uses separate temperatures for gating and divergence:
    - gate_temperature: For assessing teacher's TRUE confidence (default=1.0)
    - temperature: For softening distributions in divergence (default=2.0)

    Args:
        teacher_model: Teacher model (head 0 only)
        teacher_num_heads: Deprecated (head 0 only; kept for compatibility)
        student_model: Student model (head 0 only)
        student_num_heads: Deprecated (head 0 only; kept for compatibility)
        dataloader: Non-shuffled dataloader for index tracking
        cfg: OpenPoint config (for num_points, num_classes, etc.)
        k: Top-k parameter for divergence computation
        temperature: Temperature for divergence computation (typically 2.0)
        gate_temperature: Temperature for confidence gating (typically 1.0)
        gate_threshold: Minimum teacher confidence on true label
            With gate_temperature=1.0, use 0.6-0.8
            With gate_temperature=2.0, use 0.3-0.5
        confidence_power: Power to raise teacher confidence
        device: Torch device (default: auto-detect)

    Returns:
        scores: [N] tensor of divergence scores (0 for gated-out samples)
        labels: [N] tensor of ground truth labels
        indices: [N] tensor of original dataset indices
        gating_stats: Dict with per-class gating statistics
            {class_id: {'total': int, 'gated_in': int, 'gated_out': int, 'gate_rate': float}}
    """
    teacher_model.eval()
    student_model.eval()
    num_classes = cfg.model.get('num_classes', 40)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if teacher_num_heads > 1 or student_num_heads > 1:
        logger.warning("Multi-head models deprecated; using head 0 only.")

    all_scores = []
    all_labels = []
    all_indices = []

    # Track per-class gating
    class_total = [0] * num_classes
    class_gated_in = [0] * num_classes

    logger.info(f"Computing CE top-k scores (k={k}, temp={temperature}, gate={gate_threshold})...")
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Scoring samples")

    for batch_idx, data in pbar:
        data, target = prepare_batch(
            data,
            cfg,
            device,
            resample=False,
            truncate=False,
        )

        # Track original dataset indices
        actual_batch_size = target.shape[0]
        start_idx = batch_idx * dataloader.batch_size
        batch_indices = torch.arange(start_idx, start_idx + actual_batch_size)

        # Get head-0 logits (multi-head deprecated)
        with torch.inference_mode():
            teacher_logits = get_head0_logits(teacher_model, data)
            student_logits = get_head0_logits(student_model, data)

        # Compute CE top-k divergence scores
        scores = teacher_topk_ce_divergence(
            teacher_logits, student_logits, target,
            k=k, temperature=temperature,
            gate_threshold=gate_threshold,
            confidence_power=confidence_power
        )

        # Track per-class gating (score > 0 means passed gate)
        for i, label in enumerate(target):
            label_idx = label.item()
            class_total[label_idx] += 1
            if scores[i].item() > 0:
                class_gated_in[label_idx] += 1

        all_scores.append(scores.cpu())
        all_labels.append(target.cpu())
        all_indices.append(batch_indices.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    indices = torch.cat(all_indices)

    # Compute gating statistics
    gating_stats = {}
    for c in range(num_classes):
        gating_stats[c] = {
            'total': class_total[c],
            'gated_in': class_gated_in[c],
            'gated_out': class_total[c] - class_gated_in[c],
            'gate_rate': class_gated_in[c] / class_total[c] if class_total[c] > 0 else 0.0
        }

    logger.info(f"Scoring complete: {len(scores)} samples processed")
    logger.info(f"  Non-zero scores: {(scores > 0).sum()} / {len(scores)}")
    logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    return scores, labels, indices, gating_stats


def select_samples_with_fallback(
    dataset,
    scores: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    num_samples: int,
    per_class: bool,
    mode: str,
    num_classes: int
) -> Tuple[Subset, Dict]:
    """Select samples with random fallback for under-gated classes.

    For CE top-k scoring, mode should always be 'max' to select samples with
    highest teacher-student divergence. When a class has fewer gated samples
    than num_samples, all gated samples are taken and the rest are filled
    randomly from non-gated samples.

    Args:
        dataset: Original training dataset
        scores: [N] tensor of CE top-k divergence scores (0 for gated-out)
        labels: [N] tensor of ground truth labels
        indices: [N] tensor of original dataset indices
        num_samples: Number of samples to select (per class if per_class=True)
        per_class: If True, select num_samples per class; else num_samples total
        mode: Selection mode ('max' for highest divergence)
        num_classes: Total number of classes

    Returns:
        pruned_dataset: Subset of selected samples
        selection_stats: Dict with per-class selection statistics
            {class_id: {
                'n_gated': int,
                'n_from_gated': int,
                'n_random': int,
                'n_total': int,
                'score_min': float,
                'score_max': float
            }}
    """
    # CE top-k should always use max (highest divergence)
    if mode != 'max':
        logger.warning(
            f"CE top-k should use mode='max' (highest divergence), got '{mode}'. "
            f"Overriding to 'max'."
        )
        mode = 'max'

    if not per_class:
        # Global selection
        valid_mask = scores > 0
        valid_scores = scores[valid_mask]
        valid_indices = indices[valid_mask]

        n_gated = len(valid_indices)

        if n_gated >= num_samples:
            # Enough gated samples - select top-k by score
            sorted_positions = torch.argsort(valid_scores, descending=True)
            selected = valid_indices[sorted_positions[:num_samples]]
            n_random = 0
        else:
            # Not enough gated - take all + random fill
            invalid_indices = indices[~valid_mask]
            n_random = min(num_samples - n_gated, len(invalid_indices))
            random_fill = invalid_indices[torch.randperm(len(invalid_indices))[:n_random]]
            selected = torch.cat([valid_indices, random_fill])
            logger.warning(
                f"Global: Only {n_gated}/{num_samples} samples passed gate. "
                f"Filled {n_random} samples randomly."
            )

        selection_stats = {
            'global': {
                'n_gated': n_gated,
                'n_from_gated': min(n_gated, num_samples),
                'n_random': n_random,
                'n_total': len(selected)
            }
        }

        return Subset(dataset, selected.tolist()), selection_stats

    # Per-class selection with fallback
    selected_dataset_indices = []
    selection_stats = {}

    logger.info(f"Per-class selection: {num_samples} samples per class (mode={mode})")

    for c in range(num_classes):
        class_mask = labels == c
        class_scores = scores[class_mask]
        class_indices = indices[class_mask]

        # Separate gated vs non-gated
        gated_mask = class_scores > 0
        gated_scores = class_scores[gated_mask]
        gated_indices = class_indices[gated_mask]
        non_gated_indices = class_indices[~gated_mask]

        n_gated = len(gated_indices)
        n_needed = num_samples

        if n_gated >= n_needed:
            # Enough gated samples - select top-k by score (descending for max)
            sorted_positions = torch.argsort(gated_scores, descending=True)
            selected = gated_indices[sorted_positions[:n_needed]]
            n_from_gated = n_needed
            n_random = 0

            # Log score range for selected samples
            selected_scores = gated_scores[sorted_positions[:n_needed]]
            score_min = selected_scores.min().item()
            score_max = selected_scores.max().item()
        else:
            # Not enough gated - take all gated + random fill
            n_from_gated = n_gated
            n_random = min(n_needed - n_gated, len(non_gated_indices))

            if n_gated > 0:
                random_fill = non_gated_indices[torch.randperm(len(non_gated_indices))[:n_random]]
                selected = torch.cat([gated_indices, random_fill])
                score_min = gated_scores.min().item()
                score_max = gated_scores.max().item()
            else:
                # No gated samples at all - pure random
                random_fill = non_gated_indices[torch.randperm(len(non_gated_indices))[:n_random]]
                selected = random_fill
                score_min = score_max = 0.0

            logger.warning(
                f"Class {c}: Only {n_gated}/{n_needed} samples passed gate. "
                f"Filled {n_random} samples randomly."
            )

        selected_dataset_indices.extend(selected.cpu().numpy().tolist())

        # Store selection statistics
        selection_stats[c] = {
            'n_gated': n_gated,
            'n_from_gated': n_from_gated,
            'n_random': n_random,
            'n_total': len(selected),
            'score_min': score_min,
            'score_max': score_max
        }

        if n_gated >= n_needed:
            logger.info(
                f"  Class {c}: selected {n_from_gated} samples (all gated), "
                f"score range [{score_min:.4f}, {score_max:.4f}]"
            )

    logger.info(f"Total selected samples: {len(selected_dataset_indices)}")
    return Subset(dataset, selected_dataset_indices), selection_stats


def log_gating_and_selection_stats(
    gating_stats: Dict,
    selection_stats: Dict,
    num_classes: int
):
    """Pretty-print comprehensive gating and selection statistics.

    Displays a table showing per-class statistics including:
    - Total samples per class
    - Number of gated samples (passed confidence threshold)
    - Gating rate (percentage passed)
    - Number of selected samples
    - Number of random fills (for under-gated classes)

    Args:
        gating_stats: Per-class gating statistics from compute_ce_topk_scores_with_gating
        selection_stats: Per-class selection statistics from select_samples_with_fallback
        num_classes: Total number of classes
    """
    logger.info("=" * 80)
    logger.info("Gating and Selection Statistics")
    logger.info("=" * 80)

    # Header
    logger.info(
        f"{'Class':<6} {'Total':<8} {'Gated':<8} {'Gate%':<8} {'Selected':<10} {'Random':<8}"
    )
    logger.info("-" * 80)

    total_samples = 0
    total_gated = 0
    total_selected = 0
    total_random = 0

    # Per-class breakdown
    for c in range(num_classes):
        g = gating_stats[c]
        s = selection_stats[c]

        total_samples += g['total']
        total_gated += g['gated_in']
        total_selected += s['n_total']
        total_random += s['n_random']

        logger.info(
            f"{c:<6} {g['total']:<8} {g['gated_in']:<8} "
            f"{g['gate_rate']*100:<7.1f}% {s['n_total']:<10} {s['n_random']:<8}"
        )

    # Summary row
    logger.info("-" * 80)
    logger.info(
        f"{'TOTAL':<6} {total_samples:<8} {total_gated:<8} "
        f"{total_gated/total_samples*100:<7.1f}% {total_selected:<10} {total_random:<8}"
    )
    logger.info("=" * 80)

    # Additional insights
    classes_with_fallback = sum(1 for c in range(num_classes) if selection_stats[c]['n_random'] > 0)
    if classes_with_fallback > 0:
        logger.warning(
            f"⚠️  {classes_with_fallback}/{num_classes} classes required random fallback"
        )
    else:
        logger.info(f"✓ All classes had sufficient gated samples")

    avg_gate_rate = total_gated / total_samples if total_samples > 0 else 0.0
    logger.info(f"Average gating rate: {avg_gate_rate*100:.1f}%")
    logger.info(f"Random fill rate: {total_random/total_selected*100:.1f}%")
