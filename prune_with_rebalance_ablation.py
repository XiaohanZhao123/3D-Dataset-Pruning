"""Ablation study: Rebalancing strategies during Knowledge Distillation.

Research Question: Does reweighting during soft KD hurt performance?
Hypothesis: Instance-balanced (no reweighting) is the best strategy.

Strategies:
    - instance: No rebalancing (baseline hypothesis)
    - cb_loss: Class-balanced loss weighting (CB weights on KD + CE)
    - cb_sampling: Uniform class sampling (each class equally likely)
    - sqrt_sampling: Square-root balanced sampling (p_c ∝ n_c^0.5)

Loss structure:
    total_loss = rkd_weight * rkd_loss + kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss

    - RKD: Always unweighted (not instance-based, computes pairwise relations)
    - KD + CE: Reweighted based on strategy (for cb_loss only)
    - Sampling: Modified for cb_sampling and sqrt_sampling
"""

import logging

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix, save_checkpoint
from utils.builders import build_scorer_model
from utils.config_loader import load_pointnext_config, merge_pruning_config
from utils.rkd import rkd_loss
from utils.samplers import BalancedClassSampler, UniformClassSampler
from utils.train import build_student_model, prepare_data_dict, setup_experiment

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def compute_class_counts(dataloader, num_classes: int) -> torch.Tensor:
    """Compute class counts from dataloader."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for data in dataloader:
        labels = data["y"]
        if labels.dim() > 1:
            labels = labels.squeeze()
        for label in labels:
            counts[label.item()] += 1
    return counts


def compute_cb_weights(
    class_counts: torch.Tensor, beta: float = 0.9999
) -> torch.Tensor:
    """Compute Class-Balanced weights.

    CB weight for class c: (1 - beta) / (1 - beta^n_c)
    Normalized so mean weight = 1.
    """
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights[class_counts == 0] = 0.0
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


def get_dataset_labels(dataset) -> torch.Tensor:
    """Extract labels from dataset for sampler."""
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "label"):
        labels = dataset.label
    else:
        raise ValueError("Dataset must have 'targets', 'labels', or 'label' attribute")

    if isinstance(labels, torch.Tensor):
        return labels
    return torch.tensor(labels)


# =============================================================================
# Validation
# =============================================================================


@torch.no_grad()
def validate(model, val_loader, cfg):
    """Validate model."""
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)

        target = data["y"]
        points = data["x"]
        points = points[:, :npoints]

        data.update(prepare_data_dict(points, cfg))

        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    macc, overall_acc, accs = cm.cal_acc(cm.tp, cm.count)
    return macc, overall_acc, accs


# =============================================================================
# Model Loading
# =============================================================================


def load_scorer_model(cfg, device):
    """Load scorer model for sample selection."""
    logger.info("Loading scorer model...")
    logger.info(f"  Checkpoint: {cfg.pruning.scorer_checkpoint}")

    model, model_type = build_scorer_model(cfg, device, freeze=True)

    logger.info(f"  Loaded scorer (type={model_type})")
    return model


def load_teacher_model(cfg, device, scorer_model):
    """Load teacher model for KD, or reuse scorer if not specified."""
    teacher_ckpt = cfg.pruning.get("teacher_checkpoint")

    if not teacher_ckpt:
        logger.info("No separate teacher specified, reusing scorer model for KD")
        return scorer_model

    logger.info("Loading separate KD teacher...")
    logger.info(f"  Checkpoint: {teacher_ckpt}")

    model, model_type = build_scorer_model(cfg, device, freeze=True)

    logger.info(f"  Loaded teacher (type={model_type})")
    return model


# =============================================================================
# Trainer for Rebalancing Ablation
# =============================================================================


class RebalanceAblationTrainer:
    """Trainer for rebalancing ablation study.

    Strategies:
        - instance: No rebalancing (baseline)
        - cb_loss: CB-weighted KD + CE losses
        - cb_sampling: Uniform class sampling (handled externally via sampler)
        - sqrt_sampling: Sqrt-balanced sampling (handled externally via sampler)

    For sampling strategies, loss computation is standard.
    For cb_loss, per-sample weights are applied to KD + CE.
    RKD is always unweighted.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        cfg,
        device,
        strategy: str = "instance",
        cb_weights: torch.Tensor = None,
        rkd_weight: float = 1.0,
        rkd_distance_weight: float = 1.0,
        rkd_angle_weight: float = 2.0,
        kd_alpha: float = 0.5,
        kd_temperature: float = 3.0,
    ):
        self.model = model
        self.teacher = teacher_model
        self.cfg = cfg
        self.device = device
        self.strategy = strategy

        # Loss weights
        self.rkd_weight = rkd_weight
        self.rkd_distance_weight = rkd_distance_weight
        self.rkd_angle_weight = rkd_angle_weight
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

        # CB weights (only used for cb_loss strategy)
        self.cb_weights = cb_weights
        if cb_weights is not None:
            self.cb_weights = cb_weights.to(device)

        # Get actual models (handle DataParallel)
        self.actual_model = model.module if hasattr(model, "module") else model
        self.actual_teacher = (
            teacher_model.module if hasattr(teacher_model, "module") else teacher_model
        )

        self.npoints = cfg.num_points

        self._log_config()

    def _log_config(self):
        """Log trainer configuration."""
        logger.info(f"RebalanceAblationTrainer initialized:")
        logger.info(f"  Strategy: {self.strategy}")
        logger.info(f"  RKD weight: {self.rkd_weight}")
        logger.info(
            f"  RKD components: distance={self.rkd_distance_weight}, "
            f"angle={self.rkd_angle_weight}"
        )
        logger.info(f"  KD alpha: {self.kd_alpha}")
        logger.info(f"  KD temperature: {self.kd_temperature}")
        if self.strategy == "cb_loss" and self.cb_weights is not None:
            logger.info(
                f"  CB weights range: [{self.cb_weights.min():.3f}, {self.cb_weights.max():.3f}]"
            )

    def _prepare_batch(self, data: dict):
        """Move data to GPU and prepare."""
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)

        target = data["y"]
        points = data["x"][:, : self.npoints]
        data.update(prepare_data_dict(points, self.cfg))

        return data, target

    @torch.inference_mode()
    def _get_teacher_outputs(self, data: dict):
        """Get teacher embeddings and logits."""
        teacher_emb = self.actual_teacher.encoder.forward_cls_feat(data)
        teacher_logits = self.actual_teacher.prediction(teacher_emb)
        return teacher_emb, teacher_logits

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KD loss (optionally CB-weighted)."""
        T = self.kd_temperature
        student_log_prob = F.log_softmax(student_logits / T, dim=1)
        teacher_prob = F.softmax(teacher_logits / T, dim=1)

        if self.strategy == "cb_loss" and self.cb_weights is not None:
            # Per-sample KL
            kl_per_sample = F.kl_div(
                student_log_prob, teacher_prob, reduction="none"
            ).sum(dim=1)
            kl_per_sample = kl_per_sample * (T**2)
            # Apply CB weights
            weights = self.cb_weights[targets]
            return (kl_per_sample * weights).mean()
        else:
            # Standard KL
            return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (
                T**2
            )

    def _compute_ce_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute CE loss (optionally CB-weighted)."""
        if self.strategy == "cb_loss" and self.cb_weights is not None:
            # Per-sample CE
            ce_per_sample = F.cross_entropy(logits, targets, reduction="none")
            weights = self.cb_weights[targets]
            return (ce_per_sample * weights).mean()
        else:
            return F.cross_entropy(logits, targets)

    def train_one_epoch(self, train_loader, optimizer, scheduler, epoch: int) -> dict:
        """Train one epoch."""
        self.model.train()
        self.teacher.eval()

        # Meters
        loss_meter = AverageMeter()
        rkd_meter = AverageMeter()
        kd_meter = AverageMeter()
        ce_meter = AverageMeter()
        cm = ConfusionMatrix(num_classes=self.cfg.num_classes)

        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
        )

        for idx, data in pbar:
            data, target = self._prepare_batch(data)

            # Forward passes
            student_emb = self.actual_model.encoder.forward_cls_feat(data)
            student_logits = self.actual_model.prediction(student_emb)

            with torch.no_grad():
                teacher_emb, teacher_logits = self._get_teacher_outputs(data)

            # RKD loss (always unweighted)
            rkd_total, _, _ = rkd_loss(
                student_emb,
                teacher_emb,
                distance_weight=self.rkd_distance_weight,
                angle_weight=self.rkd_angle_weight,
            )

            # KD and CE losses (optionally weighted for cb_loss strategy)
            kd_loss_val = self._compute_kd_loss(student_logits, teacher_logits, target)
            ce_loss_val = self._compute_ce_loss(student_logits, target)

            # Total loss
            loss = (
                self.rkd_weight * rkd_total
                + self.kd_alpha * kd_loss_val
                + (1.0 - self.kd_alpha) * ce_loss_val
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if self.cfg.get("grad_norm_clip") and self.cfg.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_norm_clip
                )
            optimizer.step()

            if not self.cfg.sched_on_epoch:
                scheduler.step(epoch)

            # Update meters
            loss_meter.update(loss.item())
            rkd_meter.update(rkd_total.item())
            kd_meter.update(kd_loss_val.item())
            ce_meter.update(ce_loss_val.item())
            cm.update(student_logits.argmax(dim=1), target)

            if idx % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss_meter.val:.3f}",
                    rkd=f"{rkd_meter.val:.4f}",
                    kd=f"{kd_meter.val:.3f}",
                    ce=f"{ce_meter.val:.3f}",
                )

        macc, overall_acc, _ = cm.all_acc()
        logger.info(
            f"Epoch {epoch}: loss={loss_meter.avg:.3f}, rkd={rkd_meter.avg:.4f}, "
            f"kd={kd_meter.avg:.3f}, ce={ce_meter.avg:.3f}, "
            f"train_oa={overall_acc:.2f}%, train_macc={macc:.2f}%"
        )

        return {
            "loss": loss_meter.avg,
            "rkd_loss": rkd_meter.avg,
            "kd_loss": kd_meter.avg,
            "ce_loss": ce_meter.avg,
            "train_oa": overall_acc,
            "train_macc": macc,
        }


# =============================================================================
# Main
# =============================================================================


@hydra.main(
    config_path="cfgs_pruning", config_name="rebalance_ablation", version_base=None
)
def main(cfg: DictConfig):
    """Main workflow for rebalancing ablation study."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    strategy = cfg.pruning.get("rebalance_strategy", "instance")

    logger.info("=" * 80)
    logger.info(f"Rebalancing Ablation Study (strategy={strategy})")
    logger.info("  Hypothesis: Instance-balanced (no reweighting) is best")
    logger.info("=" * 80)

    # 1. Load and merge configs
    logger.info(f"Loading PointNeXt config: {cfg.pointnext_config}")
    pointnext_cfg = load_pointnext_config(cfg.pointnext_config)
    full_cfg = merge_pruning_config(pointnext_cfg, cfg)

    # 2. Initialize WandB
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb.get("use_wandb", True) else "disabled",
    )

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. Load scorer model
    logger.info("=" * 80)
    scorer_model = load_scorer_model(cfg, device)

    # 4. Load dataset
    logger.info("=" * 80)
    logger.info("Loading training dataset...")
    train_loader = build_dataloader_from_cfg(
        full_cfg.openpoint.batch_size,
        full_cfg.openpoint.dataset,
        full_cfg.openpoint.dataloader,
        datatransforms_cfg=full_cfg.openpoint.datatransforms,
        split="train",
        distributed=False,
    )
    original_dataset = train_loader.dataset
    logger.info(f"Original dataset size: {len(original_dataset)}")

    # 5. Compute class counts for original dataset
    logger.info("Computing class counts for original dataset...")
    original_class_counts = compute_class_counts(
        train_loader, full_cfg.openpoint.num_classes
    )
    logger.info(f"Original class counts: {original_class_counts.tolist()}")

    # 6. Score and select samples (pruning)
    logger.info("=" * 80)
    from pruning.balanced_scorers import SCORER_REGISTRY, get_scorer

    scorer_method = str(cfg.pruning.get("scorer", "loss")).lower()
    scorer_cls = SCORER_REGISTRY[scorer_method]

    # Create non-shuffled loader for scoring
    scoring_loader = DataLoader(
        original_dataset,
        batch_size=full_cfg.openpoint.batch_size,
        shuffle=False,
        num_workers=full_cfg.openpoint.dataloader.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    scorer = get_scorer(
        scorer_method,
        scorer_model,
        full_cfg.openpoint,
        device=str(device),
    )

    logger.info(f"Computing {scorer_method} scores...")
    scores, labels, indices = scorer.compute(
        scoring_loader,
        total_samples=cfg.pruning.total_samples,
        per_class=cfg.pruning.per_class,
        num_classes=full_cfg.openpoint.num_classes,
        grad_scope=cfg.pruning.get("grad_norm_scope", "head"),
        sigma=cfg.pruning.get("submodular_sigma"),
        space=cfg.pruning.get("submodular_space", "embedding"),
        loss_type=cfg.pruning.get("loss_type", "ce"),
        focal_gamma=cfg.pruning.get("focal_gamma", 2.0),
        cb_beta=cfg.pruning.get("cb_beta", 0.9999),
    )

    # Select samples
    selection_mode = scorer_cls.mode_override or cfg.pruning.mode
    selected_indices = scorer.select(
        scores,
        labels,
        indices,
        total_samples=cfg.pruning.total_samples,
        per_class=cfg.pruning.per_class,
        mode=selection_mode,
        num_classes=full_cfg.openpoint.num_classes,
    )

    pruned_dataset = Subset(original_dataset, selected_indices)
    logger.info(f"Pruned dataset size: {len(pruned_dataset)}")

    # 7. Compute class counts for pruned dataset
    pruned_loader_temp = DataLoader(pruned_dataset, batch_size=64, shuffle=False)
    pruned_class_counts = compute_class_counts(
        pruned_loader_temp, full_cfg.openpoint.num_classes
    )
    logger.info(f"Pruned class counts: {pruned_class_counts.tolist()}")

    # 8. Build training dataloader based on strategy
    logger.info("=" * 80)
    logger.info(f"Building dataloader for strategy: {strategy}")

    num_workers = full_cfg.openpoint.dataloader.num_workers
    batch_size = full_cfg.openpoint.batch_size

    if strategy == "cb_sampling":
        # Uniform class sampling
        sampler = UniformClassSampler(
            pruned_dataset,
            batch_size=batch_size,
            num_classes=full_cfg.openpoint.num_classes,
        )
        pruned_loader = DataLoader(
            pruned_dataset,
            batch_sampler=None,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("  Using UniformClassSampler (class-balanced)")

    elif strategy == "sqrt_sampling":
        # Square-root balanced sampling
        sqrt_alpha = cfg.pruning.get("sqrt_alpha", 0.5)
        sampler = BalancedClassSampler(
            pruned_dataset,
            num_classes=full_cfg.openpoint.num_classes,
            alpha=sqrt_alpha,
        )
        pruned_loader = DataLoader(
            pruned_dataset,
            batch_sampler=None,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info(f"  Using BalancedClassSampler (alpha={sqrt_alpha})")

    else:
        # instance or cb_loss: random sampling
        pruned_loader = DataLoader(
            pruned_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("  Using random sampling")

    # 9. Compute CB weights (for cb_loss strategy, always from pruned dataset)
    cb_weights = None
    if strategy == "cb_loss":
        cb_beta = cfg.pruning.get("cb_beta", 0.9999)
        cb_weights = compute_cb_weights(pruned_class_counts, cb_beta)
        logger.info(f"  CB weights from pruned dataset (beta={cb_beta})")
        logger.info(
            f"  CB weights range: [{cb_weights.min():.3f}, {cb_weights.max():.3f}]"
        )

    # 10. Load validation dataset
    val_loader = build_dataloader_from_cfg(
        full_cfg.openpoint.get("val_batch_size", batch_size),
        full_cfg.openpoint.dataset,
        full_cfg.openpoint.dataloader,
        datatransforms_cfg=full_cfg.openpoint.datatransforms,
        split="val",
        distributed=False,
    )

    # 11. Build student model
    logger.info("=" * 80)
    logger.info("Building student model...")
    student_model, model_size = build_student_model(full_cfg.openpoint, device)

    # 12. Build optimizer and scheduler
    optimizer = build_optimizer_from_cfg(
        student_model, lr=full_cfg.openpoint.lr, **full_cfg.openpoint.optimizer
    )
    scheduler = build_scheduler_from_cfg(full_cfg.openpoint, optimizer)

    exp_name = f"rebalance_{strategy}_{scorer_method}_{cfg.pruning.total_samples}"
    run_name = setup_experiment(full_cfg.openpoint, exp_name)

    # 13. Load teacher model
    logger.info("=" * 80)
    teacher_model = load_teacher_model(cfg, device, scorer_model)

    # 14. Create trainer
    logger.info("=" * 80)
    trainer = RebalanceAblationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        cfg=full_cfg.openpoint,
        device=device,
        strategy=strategy,
        cb_weights=cb_weights,
        rkd_weight=cfg.pruning.get("rkd_weight", 1.0),
        rkd_distance_weight=cfg.pruning.get("rkd_distance_weight", 1.0),
        rkd_angle_weight=cfg.pruning.get("rkd_angle_weight", 2.0),
        kd_alpha=cfg.pruning.get("kd_alpha", 0.5),
        kd_temperature=cfg.pruning.get("kd_temperature", 3.0),
    )

    # Log config summary
    logger.info("Training Configuration:")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  RKD weight: {cfg.pruning.get('rkd_weight', 1.0)}")
    logger.info(f"  KD alpha: {cfg.pruning.get('kd_alpha', 0.5)}")
    logger.info(f"  KD temperature: {cfg.pruning.get('kd_temperature', 3.0)}")
    logger.info(f"  Epochs: {full_cfg.openpoint.epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {full_cfg.openpoint.lr}")
    logger.info("=" * 80)

    # 15. Training loop
    best_val_acc = 0.0
    best_epoch = 0
    best_macc = 0.0

    for epoch in range(1, full_cfg.openpoint.epochs + 1):
        metrics = trainer.train_one_epoch(pruned_loader, optimizer, scheduler, epoch)

        if epoch % full_cfg.openpoint.val_freq == 0:
            val_macc, val_oa, val_accs = validate(
                student_model, val_loader, full_cfg.openpoint
            )

            is_best = val_oa > best_val_acc
            if is_best:
                best_val_acc = val_oa
                best_epoch = epoch
                best_macc = val_macc
                logger.info(f"*** New best @ epoch {epoch}: val_oa={val_oa:.2f}% ***")

            wandb_log = {
                "epoch": epoch,
                "val/macc": val_macc,
                "val/oa": val_oa,
                "best_val_oa": best_val_acc,
                "macc_at_best_oa": best_macc,
                "lr": optimizer.param_groups[0]["lr"],
                "train/loss": metrics["loss"],
                "train/rkd_loss": metrics["rkd_loss"],
                "train/kd_loss": metrics["kd_loss"],
                "train/ce_loss": metrics["ce_loss"],
                "train/oa": metrics["train_oa"],
                "train/macc": metrics["train_macc"],
            }
            wandb.log(wandb_log)

            # Save checkpoint (only if enabled in config)
            if is_best and cfg.pruning.get("save_checkpoint", False):
                save_checkpoint(
                    full_cfg.openpoint,
                    student_model,
                    epoch,
                    optimizer,
                    scheduler,
                    additioanl_dict={"best_val": best_val_acc},
                    is_best=True,
                )

            logger.info(
                f"Epoch {epoch}: val_oa={val_oa:.2f}, val_macc={val_macc:.2f}, "
                f"best_val_oa={best_val_acc:.2f} (@epoch {best_epoch})"
            )

        if full_cfg.openpoint.sched_on_epoch:
            scheduler.step(epoch)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% @ epoch {best_epoch}")
    logger.info("=" * 80)

    wandb.finish()


if __name__ == "__main__":
    main()
