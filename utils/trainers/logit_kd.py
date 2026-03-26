"""Logit-based Knowledge Distillation trainer."""

import logging

import torch
import torch.nn as nn

from openpoints.utils import AverageMeter, ConfusionMatrix
from utils.kd import distill_kl_loss
from utils.model_loading import get_logits

from .base import BaseTrainer, TrainMetrics

logger = logging.getLogger(__name__)


class LogitKDTrainer(BaseTrainer):
    """Knowledge distillation trainer with soft labels.

    Trains student model to match teacher's softened output distribution
    while also learning from ground truth labels.

    Loss = α * KL_div(student, teacher) + (1-α) * CE(student, labels)

    Args:
        model: Student model to train
        cfg: OpenPoint config
        device: Training device
        teacher_model: Teacher model for distillation
        kd_alpha: Weight for distillation loss (1-alpha for hard loss)
        kd_temperature: Temperature for softening distributions
    """

    def __init__(
        self,
        model: nn.Module,
        cfg,
        device,
        teacher_model: nn.Module,
        kd_alpha: float = 0.5,
        kd_temperature: float = 3.0,
    ):
        super().__init__(model, cfg, device)
        self.teacher = teacher_model
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        scheduler,
        epoch: int,
    ) -> TrainMetrics:
        """Train one epoch with logit-based knowledge distillation.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            TrainMetrics with loss, hard_loss, distill_loss, and accuracy
        """
        self.model.train()
        self.teacher.eval()

        loss_meter = AverageMeter()
        hard_loss_meter = AverageMeter()
        distill_loss_meter = AverageMeter()
        cm = ConfusionMatrix(num_classes=self.cfg.num_classes)

        pbar = self.create_progress_bar(train_loader, epoch)
        for idx, data in pbar:
            # Prepare batch
            data, target = self.prepare_batch(data)

            # Student forward pass
            logits, hard_loss = self.actual_model.get_logits_loss(data, target)

            # Teacher forward pass (no grad)
            teacher_logits = self._get_teacher_logits(data)

            # Compute distillation loss
            distill_loss = self._compute_distill_loss(logits, teacher_logits)

            # Combined loss
            loss = self.kd_alpha * distill_loss + (1.0 - self.kd_alpha) * hard_loss

            # Backward pass
            self.optimizer_step(loss, optimizer, scheduler, epoch)

            # Update metrics
            self.compute_metrics(logits, target, loss, cm, loss_meter)
            hard_loss_meter.update(hard_loss.item())
            distill_loss_meter.update(distill_loss.item())

            # Update progress bar
            self.update_progress_bar(
                pbar,
                idx,
                loss=f"{loss_meter.val:.3f}",
                hard=f"{hard_loss_meter.val:.3f}",
                dist=f"{distill_loss_meter.val:.3f}",
                acc=f"{cm.overall_accuray:.2f}",
            )

        return self.log_epoch_summary(
            epoch,
            loss_meter,
            cm,
            hard_loss=hard_loss_meter,
            distill_loss=distill_loss_meter,
        )

    @torch.inference_mode()
    def _get_teacher_logits(self, data: dict) -> torch.Tensor:
        """Get teacher logits for distillation.

        Args:
            data: Prepared input data dict

        Returns:
            Teacher logits [B, num_classes]
        """
        return get_logits(self.teacher, data)

    def _compute_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence distillation loss.

        Args:
            student_logits: Student predictions [B, num_classes]
            teacher_logits: Teacher predictions [B, num_classes]

        Returns:
            Distillation loss scalar
        """
        return distill_kl_loss(student_logits, teacher_logits, self.kd_temperature)
