"""Main pruning script for PointNeXt models.

This script:
1. Loads PointNeXt training config and merges with pruning settings
2. Loads a pre-trained teacher model for scoring
3. Scores and prunes the training dataset
4. Trains a new model on the pruned dataset
"""

import logging
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
import wandb

from utils.config_loader import (
    load_pointnext_config,
    merge_pruning_config,
    load_model_from_checkpoint,
)
from utils.train import (
    build_student_model,
    setup_experiment,
)
from utils.data_prep import prepare_batch
from utils.trainers import get_trainer
from pruning import ScoreBasedDataPruner
from pruning.utils import build_score_fn
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.utils import ConfusionMatrix, save_checkpoint

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate(model, val_loader, cfg, device):
    """Validate model.

    Adapted from PointNeXt examples/classification/train.py

    Args:
        model: Model to validate
        val_loader: Validation data loader
        cfg: OpenPoint config

    Returns:
        Tuple of (macc, overall_acc, accs)
    """
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
    for idx, data in pbar:
        data, target = prepare_batch(
            data,
            cfg,
            device,
            resample=False,
            truncate=True,
        )

        # Forward pass
        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    macc, overall_acc, accs = cm.cal_acc(cm.tp, cm.count)
    return macc, overall_acc, accs


@hydra.main(config_path="cfgs_pruning", config_name="pruning", version_base=None)
def main(cfg: DictConfig):
    """Main pruning workflow."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("Starting Pruning Workflow")
    logger.info("="*80)

    # 1. Load and merge configs
    # pointnext_config is auto-constructed via Hydra interpolation in pruning.yaml
    logger.info(f"Loading PointNeXt config: {cfg.pointnext_config}")
    pointnext_cfg = load_pointnext_config(cfg.pointnext_config)

    logger.info("Merging with pruning config...")
    full_cfg = merge_pruning_config(pointnext_cfg, cfg)

    # Log config
    logger.info("Merged config:")
    logger.info(full_cfg)

    # 2. Initialize WandB
    # Convert EasyConfig to dict for wandb (it's dict-like)
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get('entity'),
        name=cfg.wandb.name,
        config=dict(full_cfg),
        mode='online' if cfg.wandb.get('use_wandb', True) else 'disabled'
    )

    seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 3. Load scoring model (teacher)
    logger.info("Loading scoring model...")
    score_model, score_cfg = load_model_from_checkpoint(
        cfg.pruning.score_ckpt,
        cfg.pruning.get('score_config')
    )

    # 4. Build scorer (feature extraction + scoring function)
    logger.info("Building scoring pipeline...")
    score_fn = build_score_fn(full_cfg, score_model, score_cfg, device, 
                             num_classes=full_cfg.openpoint.num_classes)
    pruner = ScoreBasedDataPruner(score_fn)

    # Keep teacher model for KD if enabled
    use_kd = cfg.pruning.get('use_kd', False)
    teacher_model = score_model if use_kd else None
    if use_kd:
        logger.info("Knowledge Distillation enabled")
        logger.info(f"  Alpha: {cfg.pruning.kd_alpha}")
        logger.info(f"  Temperature: {cfg.pruning.kd_temperature}")
        teacher_model.eval()  # Set to eval mode for KD
    else:
        logger.info("Standard training (no KD)")

    # 5. Load dataset
    logger.info("Loading training dataset...")
    train_loader = build_dataloader_from_cfg(
        full_cfg.openpoint.batch_size,
        full_cfg.openpoint.dataset,
        full_cfg.openpoint.dataloader,
        datatransforms_cfg=full_cfg.openpoint.datatransforms,
        split='train',
        distributed=False
    )
    logger.info(f"Original dataset size: {len(train_loader.dataset)}")

    # 6. Prune dataset
    logger.info("="*80)
    logger.info(f"Pruning dataset:")
    logger.info(f"  Method: {cfg.pruning.score_fn.name}")
    logger.info(f"  Total samples: {cfg.pruning.total_samples}")
    logger.info(f"  Per class: {cfg.pruning.per_class}")
    logger.info(f"  Mode: {cfg.pruning.mode}")
    logger.info("="*80)

    pruned_dataset = pruner.prune_dataset(
        train_loader.dataset,
        total_samples=cfg.pruning.total_samples,
        per_class=cfg.pruning.per_class,
        mode=cfg.pruning.mode
    )
    logger.info(f"Pruned dataset size: {len(pruned_dataset)}")

    # 7. Build pruned dataloader
    # Create DataLoader directly from the pruned Subset
    from torch.utils.data import DataLoader
    pruned_loader = DataLoader(
        pruned_dataset,
        batch_size=full_cfg.openpoint.batch_size,
        shuffle=True,
        num_workers=full_cfg.openpoint.dataloader.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 8. Load validation dataset
    val_loader = build_dataloader_from_cfg(
        full_cfg.openpoint.get('val_batch_size', full_cfg.openpoint.batch_size),
        full_cfg.openpoint.dataset,
        full_cfg.openpoint.dataloader,
        datatransforms_cfg=full_cfg.openpoint.datatransforms,
        split='val',
        distributed=False
    )

    # 9. Build training model
    logger.info("Building training model...")
    model, model_size = build_student_model(full_cfg.openpoint, device)

    # Build optimizer and scheduler
    optimizer = build_optimizer_from_cfg(
        model,
        lr=full_cfg.openpoint.lr,
        **full_cfg.openpoint.optimizer
    )
    scheduler = build_scheduler_from_cfg(full_cfg.openpoint, optimizer)

    # Set up experiment directory and run_name
    exp_name = f'pruning_{full_cfg.pruning.score_fn.name}_{full_cfg.pruning.total_samples}'
    run_name = setup_experiment(full_cfg.openpoint, exp_name)

    # 10. Training loop
    logger.info("="*80)
    logger.info("Starting training on pruned dataset...")
    if use_kd:
        logger.info("Training Mode: Knowledge Distillation")
        logger.info(f"  KD Alpha: {cfg.pruning.kd_alpha}")
        logger.info(f"  KD Temperature: {cfg.pruning.kd_temperature}")
        logger.info(f"  Loss = {cfg.pruning.kd_alpha}*distill + {1-cfg.pruning.kd_alpha}*hard")
    else:
        logger.info("Training Mode: Standard (No KD)")
    logger.info(f"Epochs: {full_cfg.openpoint.epochs}")
    logger.info(f"Batch size: {full_cfg.openpoint.batch_size}")
    logger.info(f"Learning rate: {full_cfg.openpoint.lr}")
    logger.info("="*80)

    best_val_acc = 0.0
    best_epoch = 0

    trainer = get_trainer(
        model=model,
        cfg=full_cfg.openpoint,
        device=device,
        pruning_cfg=cfg.pruning,
        teacher_model=teacher_model if use_kd else None,
        teacher_num_heads=1,
    )

    # Create checkpoint directory
    ckpt_dir = full_cfg.openpoint.get('ckpt_dir', './checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, full_cfg.openpoint.epochs + 1):
        # Train
        metrics = trainer.train_one_epoch(pruned_loader, optimizer, scheduler, epoch)
        train_loss = metrics.loss
        train_macc = metrics.macc
        train_oa = metrics.overall_acc

        # Validate
        if epoch % full_cfg.openpoint.val_freq == 0:
            val_macc, val_oa, val_accs = validate(
                model, val_loader, full_cfg.openpoint, device
            )

            is_best = val_oa > best_val_acc
            if is_best:
                best_val_acc = val_oa
                best_epoch = epoch
                logger.info(f'*** New best @ epoch {epoch}: val_oa={val_oa:.2f}% ***')

            # Log to WandB
            wandb_log = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/macc': train_macc,
                'train/oa': train_oa,
                'val/macc': val_macc,
                'val/oa': val_oa,
                'best_val_oa': best_val_acc,
                'lr': optimizer.param_groups[0]['lr']
            }

            # Add extra training metrics if available
            for key, value in metrics.extra.items():
                wandb_log[f"train/{key}"] = value

            if use_kd:
                wandb_log.update({
                    'train/kd_alpha': cfg.pruning.kd_alpha,
                    'train/kd_temperature': cfg.pruning.kd_temperature
                })

            wandb.log(wandb_log)

            # Save checkpoint (only if enabled in config)
            if is_best and cfg.pruning.get("save_checkpoint", False):
                save_checkpoint(
                    full_cfg.openpoint,
                    model,
                    epoch,
                    optimizer,
                    scheduler,
                    additioanl_dict={'best_val': best_val_acc},
                    is_best=True
                )

            logger.info(
                f'Epoch {epoch}: '
                f'train_loss={train_loss:.3f}, train_oa={train_oa:.2f}, '
                f'val_oa={val_oa:.2f}, best_val_oa={best_val_acc:.2f} (@epoch {best_epoch})'
            )

        if full_cfg.openpoint.sched_on_epoch:
            scheduler.step(epoch)

    logger.info("="*80)
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% @ epoch {best_epoch}")
    logger.info("="*80)

    wandb.finish()


if __name__ == "__main__":
    main()
