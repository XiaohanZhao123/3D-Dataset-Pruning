"""Relational Knowledge Distillation (RKD) trainer.

Handles all RKD variants:
- Standard RKD (batch-only relations)
- Anchor-augmented RKD (additional forward pass with anchor samples)
- Memory-augmented RKD (MoCo-style queue of past embeddings)
- Combined RKD + Logit KD
"""

import logging

import torch
import torch.nn as nn

from openpoints.utils import AverageMeter, ConfusionMatrix
from utils.data_prep import prepare_data_dict, resample_points_fps
from utils.kd import distill_kl_loss
from utils.model_loading import get_logits
from utils.rkd import proto_kd_loss, rkd_loss, rkd_loss_with_anchor_mask

from .base import BaseTrainer, TrainMetrics

logger = logging.getLogger(__name__)


def collate_anchor_batch(data_list, device):
    """Collate list of dataset items into a batch dict for anchor samples.

    Args:
        data_list: List of dicts from dataset[i]
        device: Target device for tensors

    Returns:
        Batched dict with stacked tensors on device
    """
    pos = torch.stack([d["pos"] for d in data_list]).to(device)
    x = torch.stack([d["x"] for d in data_list]).to(device)

    # Handle labels (tensor or int)
    y_list = [d["y"] for d in data_list]
    y_tensors = []
    for y_item in y_list:
        if torch.is_tensor(y_item):
            y_tensors.append(y_item.view(-1))
        else:
            y_tensors.append(torch.tensor([y_item]))
    y = torch.cat(y_tensors).to(device)

    batch = {"pos": pos, "x": x, "y": y}

    # Optional heights key
    if "heights" in data_list[0]:
        heights = torch.stack([d["heights"] for d in data_list]).to(device)
        batch["heights"] = heights

    return batch


class RKDTrainer(BaseTrainer):
    """Relational Knowledge Distillation trainer.

    Transfers structural knowledge (pairwise distances, angular relations)
    from teacher to student embeddings.

    Supports three augmentation modes:
    1. Standard (batch-only): Relations within current batch
    2. Anchor-augmented: Additional samples for richer relations
    3. Memory-augmented: MoCo-style queue of past embeddings

    Can optionally combine with logit-based KD for hybrid distillation.

    Args:
        model: Student model to train
        cfg: OpenPoint config
        device: Training device
        teacher_model: Teacher model for distillation
        rkd_distance_weight: Weight for distance-wise loss (β)
        rkd_angle_weight: Weight for angle-wise loss (γ)
        anchor_size: Number of anchor samples per step (0 = disabled)
        memory_rkd: MemoryAugmentedRKD instance (None = disabled)
        use_logit_kd: Whether to also use logit-based KD
        kd_alpha: Weight for logit distillation loss
        kd_temperature: Temperature for logit softening
        rkd_loss_scale: Multiplier for RKD loss when combined with logit KD
    """

    def __init__(
        self,
        model: nn.Module,
        cfg,
        device,
        teacher_model: nn.Module,
        rkd_distance_weight: float = 1.0,
        rkd_angle_weight: float = 2.0,
        anchor_size: int = 0,
        memory_rkd=None,
        use_logit_kd: bool = False,
        kd_alpha: float = 0.5,
        kd_temperature: float = 3.0,
        rkd_loss_scale: float = 1.0,
        # Proto-RKD parameters
        prototypes: torch.Tensor = None,
        proto_weight: float = 1.0,
        proto_tau: float = 0.1,
    ):
        super().__init__(model, cfg, device)
        self.teacher = teacher_model
        self.rkd_distance_weight = rkd_distance_weight
        self.rkd_angle_weight = rkd_angle_weight
        self.anchor_size = anchor_size
        self.memory_rkd = memory_rkd
        self.use_logit_kd = use_logit_kd
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.rkd_loss_scale = rkd_loss_scale

        # Proto-RKD
        self.prototypes = prototypes  # [K, D] or None
        self.proto_weight = proto_weight
        self.proto_tau = proto_tau

        # Get actual teacher model
        self.actual_teacher = (
            teacher_model.module if hasattr(teacher_model, "module") else teacher_model
        )

        # Log configuration
        self._log_config()

    def _log_config(self):
        """Log RKD configuration."""
        mode = "standard (batch-only)"
        if self.anchor_size > 0:
            mode = f"anchor-augmented (size={self.anchor_size})"
        elif self.memory_rkd is not None:
            mode = f"memory-augmented (queue={self.memory_rkd.queue_size})"

        logger.info(f"RKD Trainer initialized: {mode}")
        logger.info(
            f"  Weights: distance={self.rkd_distance_weight}, angle={self.rkd_angle_weight}"
        )
        if self.use_logit_kd:
            logger.info(
                f"  Combined with logit KD: α={self.kd_alpha}, T={self.kd_temperature}, "
                f"rkd_scale={self.rkd_loss_scale}"
            )
        if self.prototypes is not None:
            logger.info(
                f"  Proto-RKD: enabled (K={self.prototypes.shape[0]}, "
                f"weight={self.proto_weight}, τ={self.proto_tau})"
            )

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        scheduler,
        epoch: int,
    ) -> TrainMetrics:
        """Train one epoch with RKD.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            TrainMetrics with loss components and accuracy
        """
        self.model.train()
        self.teacher.eval()

        # Meters
        loss_meter = AverageMeter()
        hard_loss_meter = AverageMeter()
        distill_loss_meter = AverageMeter()
        rkd_dist_meter = AverageMeter()
        rkd_angle_meter = AverageMeter()
        proto_loss_meter = AverageMeter()
        cm = ConfusionMatrix(num_classes=self.cfg.num_classes)

        # Dataset reference for anchor sampling
        dataset = train_loader.dataset
        dataset_size = len(dataset)

        pbar = self.create_progress_bar(train_loader, epoch)
        for idx, data in pbar:
            # Prepare batch
            data, target = self.prepare_batch(data)

            # Student forward pass
            logits, hard_loss = self.actual_model.get_logits_loss(data, target)

            # Extract embeddings (use get_embeddings if available for Point-MAE support)
            with torch.no_grad():
                if hasattr(self.teacher, "get_embeddings"):
                    teacher_emb = self.teacher.get_embeddings(data)
                else:
                    teacher_emb = self.actual_teacher.encoder.forward_cls_feat(data)
            if hasattr(self.model, "get_embeddings"):
                student_emb = self.model.get_embeddings(data)
            else:
                student_emb = self.actual_model.encoder.forward_cls_feat(data)

            # Compute RKD loss based on augmentation mode
            if self.anchor_size > 0:
                rkd_total, rkd_dist, rkd_angle = self._compute_anchor_rkd(
                    teacher_emb, student_emb, dataset, dataset_size
                )
            elif self.memory_rkd is not None:
                rkd_total, rkd_dist, rkd_angle = self.memory_rkd.compute_loss(
                    teacher_emb, student_emb
                )
            else:
                rkd_total, rkd_dist, rkd_angle = rkd_loss(
                    student_emb,
                    teacher_emb,
                    distance_weight=self.rkd_distance_weight,
                    angle_weight=self.rkd_angle_weight,
                )

            # Compute Proto-RKD loss if prototypes available
            if self.prototypes is not None:
                proto_loss = proto_kd_loss(
                    student_emb, teacher_emb, self.prototypes, self.proto_tau
                )
                rkd_total = rkd_total + self.proto_weight * proto_loss
            else:
                proto_loss = torch.tensor(0.0, device=self.device)

            # Compute final loss
            if self.use_logit_kd:
                # Combined: RKD + Logit KD
                teacher_logits = self._get_teacher_logits(data)
                distill_loss = self._compute_distill_loss(logits, teacher_logits)

                loss = (
                    self.kd_alpha * distill_loss
                    + (1.0 - self.kd_alpha) * hard_loss
                    + self.rkd_loss_scale * rkd_total
                )
                distill_loss_meter.update(distill_loss.item())
            else:
                # RKD only
                loss = hard_loss + rkd_total
                distill_loss_meter.update(0.0)

            # Backward pass
            self.optimizer_step(loss, optimizer, scheduler, epoch)

            # Update memory queue AFTER backward
            if self.memory_rkd is not None:
                self.memory_rkd.push(teacher_emb, student_emb)

            # Update metrics
            self.compute_metrics(logits, target, loss, cm, loss_meter)
            hard_loss_meter.update(hard_loss.item())
            rkd_dist_meter.update(rkd_dist.item())
            rkd_angle_meter.update(rkd_angle.item())
            proto_loss_meter.update(proto_loss.item())

            # Update progress bar
            pbar_kwargs = {
                "loss": f"{loss_meter.val:.3f}",
                "hard": f"{hard_loss_meter.val:.3f}",
                "rkd_d": f"{rkd_dist_meter.val:.4f}",
                "rkd_a": f"{rkd_angle_meter.val:.4f}",
            }
            if self.prototypes is not None:
                pbar_kwargs["proto"] = f"{proto_loss_meter.val:.4f}"
            self.update_progress_bar(pbar, idx, **pbar_kwargs)

        return self.log_epoch_summary(
            epoch,
            loss_meter,
            cm,
            hard_loss=hard_loss_meter,
            distill_loss=distill_loss_meter,
            rkd_dist=rkd_dist_meter,
            rkd_angle=rkd_angle_meter,
            proto_loss=proto_loss_meter,
        )

    def _compute_anchor_rkd(
        self,
        teacher_emb_active: torch.Tensor,
        student_emb_active: torch.Tensor,
        dataset,
        dataset_size: int,
    ):
        """Compute RKD with anchor augmentation.

        Args:
            teacher_emb_active: Teacher embeddings for current batch [B, D]
            student_emb_active: Student embeddings for current batch [B, D]
            dataset: Training dataset for sampling anchors
            dataset_size: Total dataset size

        Returns:
            (total_loss, distance_loss, angle_loss)
        """
        # Sample anchor indices
        anchor_indices = torch.randperm(dataset_size)[: self.anchor_size]
        anchor_data_list = [dataset[i] for i in anchor_indices.tolist()]
        anchor_batch = collate_anchor_batch(anchor_data_list, self.device)

        # Prepare anchor data
        anchor_points = resample_points_fps(anchor_batch["x"], self.npoints)
        anchor_data_dict = prepare_data_dict(anchor_points, self.cfg)

        # Forward anchors (no grad) - use get_embeddings if available for Point-MAE
        with torch.inference_mode():
            if hasattr(self.teacher, "get_embeddings"):
                teacher_emb_anchor = self.teacher.get_embeddings(anchor_data_dict)
            else:
                teacher_emb_anchor = self.actual_teacher.encoder.forward_cls_feat(
                    anchor_data_dict
                )
            if hasattr(self.model, "get_embeddings"):
                student_emb_anchor = self.model.get_embeddings(anchor_data_dict)
            else:
                student_emb_anchor = self.actual_model.encoder.forward_cls_feat(
                    anchor_data_dict
                )

        # Combine active + anchors
        teacher_all = torch.cat([teacher_emb_active, teacher_emb_anchor], dim=0)
        student_all = torch.cat([student_emb_active, student_emb_anchor], dim=0)

        n_active = student_emb_active.size(0)

        return rkd_loss_with_anchor_mask(
            student_all,
            teacher_all,
            n_active=n_active,
            distance_weight=self.rkd_distance_weight,
            angle_weight=self.rkd_angle_weight,
        )

    @torch.inference_mode()
    def _get_teacher_logits(self, data: dict) -> torch.Tensor:
        """Get teacher logits for combined KD."""
        return get_logits(self.teacher, data)

    def _compute_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence distillation loss."""
        return distill_kl_loss(student_logits, teacher_logits, self.kd_temperature)
