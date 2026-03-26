"""Base trainer class with shared utilities.

All trainers inherit from BaseTrainer which provides common functionality:
- Batch preparation (GPU transfer, FPS resampling, data dict)
- Metrics computation (confusion matrix, loss meters)
- Epoch summary logging
- Optimizer step with gradient clipping
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from openpoints.utils import AverageMeter, ConfusionMatrix
from utils.data_prep import prepare_batch

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Training metrics returned by train_one_epoch()."""

    loss: float
    macc: float
    overall_acc: float
    extra: Dict[str, float] = field(default_factory=dict)

    def to_wandb_dict(self, prefix: str = "train") -> Dict[str, float]:
        """Convert to wandb logging dict."""
        result = {
            f"{prefix}/loss": self.loss,
            f"{prefix}/macc": self.macc,
            f"{prefix}/oa": self.overall_acc,
        }
        for key, value in self.extra.items():
            result[f"{prefix}/{key}"] = value
        return result


class BaseTrainer(ABC):
    """Base trainer with shared utilities.

    Subclasses must implement train_one_epoch().

    Args:
        model: Student model to train
        cfg: OpenPoint config (cfg.openpoint from merged config)
        device: Training device
    """

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.npoints = cfg.num_points

        # Get actual model (handle DataParallel)
        self.actual_model = model.module if hasattr(model, "module") else model

    @abstractmethod
    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        scheduler,
        epoch: int,
    ) -> TrainMetrics:
        """Train one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            TrainMetrics with loss, accuracy, and optional extra metrics
        """
        pass

    def prepare_batch(self, data: dict) -> Tuple[dict, torch.Tensor]:
        """Move data to device, resample points, prepare data dict.

        Args:
            data: Raw batch dict from dataloader

        Returns:
            (prepared_data, target) tuple
        """
        return prepare_batch(
            data,
            self.cfg,
            self.device,
            npoints=self.npoints,
            resample=True,
            truncate=False,
        )

    def compute_metrics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor,
        cm: ConfusionMatrix,
        loss_meter: AverageMeter,
    ):
        """Update confusion matrix and loss meter.

        Args:
            logits: Model predictions [B, num_classes]
            target: Ground truth labels [B]
            loss: Loss value (scalar tensor)
            cm: Confusion matrix to update
            loss_meter: Loss meter to update
        """
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

    def optimizer_step(
        self,
        loss: torch.Tensor,
        optimizer,
        scheduler,
        epoch: int,
    ):
        """Backward pass, gradient clipping, optimizer step.

        Args:
            loss: Loss to backpropagate
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch (for step-wise scheduling)
        """
        loss.backward()

        if self.cfg.get("grad_norm_clip") is not None and self.cfg.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_norm_clip, norm_type=2
            )

        optimizer.step()
        self.model.zero_grad()

        # Step-wise scheduling (if not epoch-based)
        if not self.cfg.sched_on_epoch:
            scheduler.step(epoch)

    def log_epoch_summary(
        self,
        epoch: int,
        loss_meter: AverageMeter,
        cm: ConfusionMatrix,
        **extra_meters: AverageMeter,
    ) -> TrainMetrics:
        """Log epoch summary and return metrics.

        Args:
            epoch: Current epoch number
            loss_meter: Loss meter
            cm: Confusion matrix
            **extra_meters: Additional meters to log (hard_loss, distill_loss, etc.)

        Returns:
            TrainMetrics dataclass
        """
        macc, overall_acc, accs = cm.all_acc()

        # Build log message
        msg = f"Epoch {epoch}: loss={loss_meter.avg:.3f}"
        extra_dict = {}

        for name, meter in extra_meters.items():
            msg += f", {name}={meter.avg:.3f}"
            extra_dict[name] = meter.avg

        msg += f", train_oa={overall_acc:.2f}%, train_macc={macc:.2f}%"
        logger.info(msg)

        return TrainMetrics(
            loss=loss_meter.avg,
            macc=macc,
            overall_acc=overall_acc,
            extra=extra_dict,
        )

    def create_progress_bar(self, train_loader, epoch: int):
        """Create tqdm progress bar for epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            tqdm progress bar wrapping enumerated loader
        """
        return tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
        )

    def update_progress_bar(self, pbar, idx: int, **metrics):
        """Update progress bar postfix every 10 iterations.

        Args:
            pbar: tqdm progress bar
            idx: Current batch index
            **metrics: Metrics to display (name -> formatted string)
        """
        if idx % 10 == 0:
            pbar.set_postfix(metrics)
