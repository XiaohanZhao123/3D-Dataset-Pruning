"""Prototype computation utilities for Proto-RKD.

Computes class prototypes by averaging embeddings over multiple passes
to handle data augmentation noise.
"""

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.data_prep import prepare_batch

logger = logging.getLogger(__name__)


def compute_class_prototypes(
    model,
    dataloader,
    cfg,
    device,
    num_classes: int,
    num_passes: int = 5,
) -> torch.Tensor:
    """Compute class prototypes by averaging embeddings over multiple passes.

    Since data augmentation causes different embeddings for the same sample,
    we run multiple forward passes and average to get stable prototypes.

    Args:
        model: Teacher model with encoder.forward_cls_feat() method
        dataloader: Non-shuffled dataloader (may have augmentation)
        cfg: OpenPoint config with num_points
        device: Torch device
        num_classes: Number of classes
        num_passes: Number of forward passes to average (default 5)

    Returns:
        prototypes: [num_classes, embed_dim] L2-normalized class prototypes
    """
    model.eval()

    # Get actual model (handle DataParallel)
    actual_model = model.module if hasattr(model, "module") else model

    # Accumulators (lazy init)
    class_sum = None  # [num_classes, embed_dim]
    class_count = torch.zeros(num_classes, device=device)

    npoints = cfg.num_points

    # Use inference_mode for efficient forward passes
    with torch.inference_mode():
        for pass_idx in range(num_passes):
            pbar = tqdm(
                dataloader,
                desc=f"Prototype pass {pass_idx + 1}/{num_passes}",
                leave=False,
            )
            for data in pbar:
                data, labels = prepare_batch(
                    data,
                    cfg,
                    device,
                    npoints=npoints,
                    resample=False,
                    truncate=True,
                )

                # Extract embeddings
                emb = actual_model.encoder.forward_cls_feat(data)  # [B, D]

                # Lazy init class_sum with correct embed_dim
                if class_sum is None:
                    embed_dim = emb.shape[1]
                    class_sum = torch.zeros(num_classes, embed_dim, device=device)
                    logger.info(f"Embedding dimension: {embed_dim}")

                # Accumulate per class
                for c in range(num_classes):
                    mask = labels == c
                    if mask.any():
                        class_sum[c] += emb[mask].sum(dim=0)
                        class_count[c] += mask.sum()

        # Compute average
        # Avoid division by zero for classes with no samples
        valid_mask = class_count > 0
        prototypes = torch.zeros_like(class_sum)
        prototypes[valid_mask] = class_sum[valid_mask] / class_count[
            valid_mask
        ].unsqueeze(1)

        # Warn about missing classes
        if not valid_mask.all():
            missing = (~valid_mask).nonzero(as_tuple=True)[0].tolist()
            logger.warning(f"No samples found for classes: {missing}")

        # L2 normalize for cosine similarity
        prototypes = F.normalize(prototypes, p=2, dim=1)

    # Clone OUTSIDE inference_mode to get normal tensor for autograd
    prototypes = prototypes.clone().detach()

    logger.info(
        f"Computed prototypes: shape={prototypes.shape}, "
        f"classes_with_samples={valid_mask.sum().item()}/{num_classes}"
    )

    return prototypes
