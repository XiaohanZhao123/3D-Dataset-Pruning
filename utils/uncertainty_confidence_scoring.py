"""Uncertainty-Confidence scoring for dataset pruning.

This module computes sample informativeness using:
    score = (1 - student_prob[y])^b * (teacher_prob[y])^a

Where:
    - student_prob[y]: Student's predicted probability for true class y
    - teacher_prob[y]: Teacher's predicted probability for true class y
    - a, b: Power hyperparameters to control emphasis

High score indicates: teacher is confident but student is uncertain (informative samples)
"""

import logging
import torch
from tqdm import tqdm

from .data_prep import prepare_batch
from .model_loading import get_head0_logits

logger = logging.getLogger(__name__)


def compute_uncertainty_confidence_scores(
    teacher_model, teacher_heads,
    student_model, student_heads,
    dataloader, cfg,
    teacher_power=1.0,
    student_power=1.0,
    device: torch.device | None = None,
):
    """Compute uncertainty-confidence scores for all samples.

    Args:
        teacher_model: Trained teacher model
        teacher_heads: Deprecated (head 0 only; kept for compatibility)
        student_model: Trained student model
        student_heads: Deprecated (head 0 only; kept for compatibility)
        dataloader: Non-shuffled dataloader (preserves index order)
        cfg: OpenPoint config
        teacher_power: Power a for teacher confidence (default: 1.0)
        student_power: Power b for student uncertainty (default: 1.0)
        device: Torch device (default: auto-detect)

    Returns:
        scores: (N,) tensor of uncertainty-confidence scores
        labels: (N,) tensor of ground truth labels
        indices: (N,) tensor of sample indices
    """
    teacher_model.eval()
    student_model.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if teacher_heads > 1 or student_heads > 1:
        logger.warning("Multi-head models deprecated; using head 0 only.")

    all_scores = []
    all_labels = []
    all_indices = []

    with torch.inference_mode():
        for data in tqdm(dataloader, desc="Computing Unc×Conf scores"):
            data, labels = prepare_batch(
                data,
                cfg,
                device,
                resample=True,
                truncate=False,
            )

            # Get teacher and student predictions (ensemble-averaged if multi-head)
            teacher_logits = get_head0_logits(teacher_model, data)
            student_logits = get_head0_logits(student_model, data)

            # Compute probabilities
            teacher_probs = torch.softmax(teacher_logits, dim=1)
            student_probs = torch.softmax(student_logits, dim=1)

            # Extract probabilities for true class y
            batch_indices = torch.arange(labels.size(0), device=labels.device)
            teacher_conf = teacher_probs[batch_indices, labels]  # P_teacher(y)
            student_conf = student_probs[batch_indices, labels]  # P_student(y)

            # Compute score: (1 - P_student(y))^b * (P_teacher(y))^a
            student_uncertainty = 1.0 - student_conf
            scores = (student_uncertainty ** student_power) * (teacher_conf ** teacher_power)

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

            # Get original indices if available
            if 'idx' in data:
                all_indices.append(data['idx'].cpu())
            else:
                # Fallback: sequential indices
                batch_start = len(all_indices) * dataloader.batch_size
                batch_indices = torch.arange(
                    batch_start,
                    batch_start + labels.size(0),
                    dtype=torch.long
                )
                all_indices.append(batch_indices)

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    indices = torch.cat(all_indices, dim=0)

    logger.info(f"Computed {len(scores)} uncertainty-confidence scores")
    logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    logger.info(f"  Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")

    return scores, labels, indices


def select_samples_by_scores(
    dataset, scores, labels, indices,
    num_samples, per_class, mode, num_classes
):
    """Select samples based on uncertainty-confidence scores.

    Args:
        dataset: Original dataset
        scores: (N,) tensor of scores
        labels: (N,) tensor of labels
        indices: (N,) tensor of sample indices
        num_samples: Number of samples to select (per-class or global)
        per_class: If True, select num_samples per class; else global
        mode: 'max' (high scores) or 'min' (low scores)
        num_classes: Total number of classes

    Returns:
        pruned_dataset: Subset of dataset with selected samples
        stats: Dict with selection statistics
    """
    from torch.utils.data import Subset

    assert mode in ['max', 'min'], f"mode must be 'max' or 'min', got {mode}"

    selected_indices = []
    stats = {'per_class': {}, 'total': 0}

    if per_class:
        # Per-class selection
        for cls in range(num_classes):
            cls_mask = labels == cls
            cls_scores = scores[cls_mask]
            cls_indices = indices[cls_mask]

            if len(cls_scores) == 0:
                logger.warning(f"Class {cls}: no samples found!")
                stats['per_class'][cls] = {
                    'available': 0,
                    'selected': 0,
                    'score_range': (0.0, 0.0)
                }
                continue

            # Select top-k or bottom-k
            k = min(num_samples, len(cls_scores))
            if mode == 'max':
                _, topk_idx = torch.topk(cls_scores, k, largest=True)
            else:
                _, topk_idx = torch.topk(cls_scores, k, largest=False)

            selected_cls_indices = cls_indices[topk_idx]
            selected_indices.append(selected_cls_indices)

            selected_scores = cls_scores[topk_idx]
            stats['per_class'][cls] = {
                'available': len(cls_scores),
                'selected': k,
                'score_range': (selected_scores.min().item(), selected_scores.max().item()),
                'score_mean': selected_scores.mean().item()
            }

        selected_indices = torch.cat(selected_indices).tolist()

    else:
        # Global selection
        k = min(num_samples, len(scores))
        if mode == 'max':
            _, topk_idx = torch.topk(scores, k, largest=True)
        else:
            _, topk_idx = torch.topk(scores, k, largest=False)

        selected_indices = indices[topk_idx].tolist()

        selected_scores = scores[topk_idx]
        stats['global'] = {
            'available': len(scores),
            'selected': k,
            'score_range': (selected_scores.min().item(), selected_scores.max().item()),
            'score_mean': selected_scores.mean().item()
        }

    stats['total'] = len(selected_indices)

    # Create subset
    pruned_dataset = Subset(dataset, selected_indices)

    logger.info(f"Selected {stats['total']} samples (mode={mode})")

    return pruned_dataset, stats


def log_selection_stats(stats, num_classes):
    """Log detailed selection statistics."""
    logger.info("\n" + "="*80)
    logger.info("SELECTION STATISTICS")
    logger.info("="*80)

    if 'per_class' in stats:
        logger.info("\nPer-Class Selection:")
        logger.info(f"{'Class':<8} {'Available':<12} {'Selected':<10} {'Score Range':<25} {'Mean Score':<12}")
        logger.info("-" * 80)

        for cls in range(num_classes):
            if cls in stats['per_class']:
                s = stats['per_class'][cls]
                score_range = f"[{s['score_range'][0]:.4f}, {s['score_range'][1]:.4f}]"
                mean_score = f"{s['score_mean']:.4f}" if 'score_mean' in s else "N/A"
                logger.info(
                    f"{cls:<8} {s['available']:<12} {s['selected']:<10} {score_range:<25} {mean_score:<12}"
                )

    elif 'global' in stats:
        s = stats['global']
        logger.info(f"\nGlobal Selection:")
        logger.info(f"  Available: {s['available']}")
        logger.info(f"  Selected: {s['selected']}")
        logger.info(f"  Score range: [{s['score_range'][0]:.4f}, {s['score_range'][1]:.4f}]")
        logger.info(f"  Mean score: {s['score_mean']:.4f}")

    logger.info(f"\nTotal selected samples: {stats['total']}")
    logger.info("="*80 + "\n")
