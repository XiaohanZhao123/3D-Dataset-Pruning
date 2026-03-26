"""Standard trainer without knowledge distillation."""

import torch.nn as nn

from openpoints.utils import AverageMeter, ConfusionMatrix

from .base import BaseTrainer, TrainMetrics


class StandardTrainer(BaseTrainer):
    """Standard training without knowledge distillation.

    Simply trains the model with cross-entropy loss on ground truth labels.

    Args:
        model: Model to train
        cfg: OpenPoint config
        device: Training device
    """

    def __init__(self, model: nn.Module, cfg, device):
        super().__init__(model, cfg, device)

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        scheduler,
        epoch: int,
    ) -> TrainMetrics:
        """Train one epoch with standard cross-entropy loss.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            TrainMetrics with loss and accuracy
        """
        self.model.train()

        loss_meter = AverageMeter()
        cm = ConfusionMatrix(num_classes=self.cfg.num_classes)

        pbar = self.create_progress_bar(train_loader, epoch)
        for idx, data in pbar:
            # Prepare batch
            data, target = self.prepare_batch(data)

            # Forward pass
            logits, loss = self.actual_model.get_logits_loss(data, target)

            # Backward pass
            self.optimizer_step(loss, optimizer, scheduler, epoch)

            # Update metrics
            self.compute_metrics(logits, target, loss, cm, loss_meter)

            # Update progress bar
            self.update_progress_bar(
                pbar,
                idx,
                loss=f"{loss_meter.val:.3f}",
                acc=f"{cm.overall_accuray:.2f}",
            )

        return self.log_epoch_summary(epoch, loss_meter, cm)
