"""Trainer classes for PointNeXt pruning framework.

Provides modular trainers for different training modes:
- StandardTrainer: No knowledge distillation
- LogitKDTrainer: Soft-label knowledge distillation
- RKDTrainer: Relational knowledge distillation (with variants)

Usage:
    from utils.trainers import get_trainer

    trainer = get_trainer(model, cfg, device, pruning_cfg, teacher_model)

    for epoch in range(epochs):
        metrics = trainer.train_one_epoch(train_loader, optimizer, scheduler, epoch)
"""

import logging
from typing import Optional

import torch.nn as nn

from .base import BaseTrainer, TrainMetrics
from .logit_kd import LogitKDTrainer
from .rkd import RKDTrainer
from .standard import StandardTrainer

logger = logging.getLogger(__name__)

__all__ = [
    "BaseTrainer",
    "TrainMetrics",
    "StandardTrainer",
    "LogitKDTrainer",
    "RKDTrainer",
    "get_trainer",
]


def get_trainer(
    model: nn.Module,
    cfg,
    device,
    pruning_cfg,
    teacher_model: Optional[nn.Module] = None,
    prototypes=None,
) -> BaseTrainer:
    """Create appropriate trainer based on configuration.

    Factory function that examines pruning_cfg and returns the right trainer:
    - No KD → StandardTrainer
    - use_rkd=True → RKDTrainer (with optional logit KD)
    - use_kd=True → LogitKDTrainer

    Args:
        model: Student model to train
        cfg: OpenPoint config (cfg.openpoint from merged config)
        device: Training device
        pruning_cfg: Pruning config with KD settings
        teacher_model: Teacher model (required if use_kd=True)

    Returns:
        Configured trainer instance

    Raises:
        ValueError: If KD enabled but no teacher provided
    """
    use_kd = pruning_cfg.get("use_kd", False)

    if not use_kd:
        logger.info("Creating StandardTrainer (no KD)")
        return StandardTrainer(model, cfg, device)

    # KD enabled - need teacher
    if teacher_model is None:
        raise ValueError("use_kd=True but no teacher_model provided")

    use_rkd = pruning_cfg.get("use_rkd", False)

    if use_rkd:
        # RKD trainer (handles all RKD variants)
        memory_rkd = _create_memory_rkd(pruning_cfg)

        logger.info("Creating RKDTrainer")
        return RKDTrainer(
            model=model,
            cfg=cfg,
            device=device,
            teacher_model=teacher_model,
            rkd_distance_weight=pruning_cfg.get("rkd_distance_weight", 1.0),
            rkd_angle_weight=pruning_cfg.get("rkd_angle_weight", 2.0),
            anchor_size=pruning_cfg.get("rkd_anchor_size", 0),
            memory_rkd=memory_rkd,
            use_logit_kd=pruning_cfg.get("use_logit_kd", False),
            kd_alpha=pruning_cfg.get("kd_alpha", 0.5),
            kd_temperature=pruning_cfg.get("kd_temperature", 3.0),
            rkd_loss_scale=pruning_cfg.get("rkd_loss_scale", 1.0),
            # Proto-RKD
            prototypes=prototypes,
            proto_weight=pruning_cfg.get("proto_weight", 1.0),
            proto_tau=pruning_cfg.get("proto_tau", 0.1),
        )

    # Standard logit-based KD
    logger.info("Creating LogitKDTrainer")
    return LogitKDTrainer(
        model=model,
        cfg=cfg,
        device=device,
        teacher_model=teacher_model,
        kd_alpha=pruning_cfg.get("kd_alpha", 0.5),
        kd_temperature=pruning_cfg.get("kd_temperature", 3.0),
    )


def _create_memory_rkd(pruning_cfg):
    """Create MemoryAugmentedRKD if enabled.

    Args:
        pruning_cfg: Pruning config

    Returns:
        MemoryAugmentedRKD instance or None
    """
    # Anchor mode takes precedence over memory mode
    if pruning_cfg.get("rkd_anchor_size", 0) > 0:
        return None

    if not pruning_cfg.get("use_memory_rkd", False):
        return None

    from utils.rkd import MemoryAugmentedRKD

    queue_size = pruning_cfg.get("rkd_queue_size", 384)
    sample_size = pruning_cfg.get("rkd_sample_size", 72)

    logger.info(
        f"Creating MemoryAugmentedRKD: queue_size={queue_size}, sample_size={sample_size}"
    )

    return MemoryAugmentedRKD(
        queue_size=queue_size,
        embedding_dim=256,  # Placeholder, actual dim determined on first push
        sample_size=sample_size,
        distance_weight=pruning_cfg.get("rkd_distance_weight", 1.0),
        angle_weight=pruning_cfg.get("rkd_angle_weight", 2.0),
    )
