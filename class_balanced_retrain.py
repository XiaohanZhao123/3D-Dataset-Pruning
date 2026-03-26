"""Class-balanced retraining script for PointNeXt models.

This script:
1. Loads a pre-trained model
2. Freezes the encoder and re-initializes the classifier
3. Re-trains the classifier using uniform class sampling
4. Evaluates and saves before/after models for comparison
"""

import logging
import os

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.utils import ConfusionMatrix, save_checkpoint
from utils.config_loader import (
    load_model_from_checkpoint,
    load_pointnext_config,
    merge_class_balanced_config,
)
from utils.samplers import UniformClassSampler
from utils.data_prep import prepare_batch
from utils.trainers import StandardTrainer
from utils.pointmae_utils import (
    is_pointmae_model,
    load_pointmae_model,
    add_get_logits_loss,
    freeze_pointmae_encoder,
    reinit_pointmae_classifier,
    PointMAEDatasetWrapper,
)
from utils.pointmae_dataloader import PointMAEModelNet40, PointMAEScanObjectNN, build_pointmae_dataloader

logger = logging.getLogger(__name__)


def is_pointmlp_model(model):
    """Check if model is PointMLP architecture."""
    return (
        hasattr(model, "classifier")
        and hasattr(model, "local_grouper_list")
        and hasattr(model, "embedding")
    )


def get_classifier_heads(model):
    """
    Get classifier head(s) from model, handling both BaseCls and PointMLP.

    Returns:
        - For BaseCls with single head: the Linear layer
        - For BaseCls with multi-head: ModuleList of Linear layers
        - For PointMLP: the Sequential classifier (multi-head not supported)
    """
    if is_pointmlp_model(model):
        return model.classifier
    elif hasattr(model, "prediction") and hasattr(model.prediction, "head"):
        return model.prediction.head
    else:
        raise ValueError("Cannot find classifier head in model")


def freeze_encoder_reinit_classifier(model, num_classes, device):
    """
    Freeze encoder and re-initialize classifier.

    Args:
        model: PointNeXt model (BaseCls with .encoder and .prediction) or PointMLP
        num_classes: Number of classes for classifier
        device: Device to move new classifier to

    Returns:
        Modified model with frozen encoder and fresh classifier
    """
    logger.info("Freezing encoder and re-initializing classifier...")

    # Check model architecture type
    is_pointmlp = is_pointmlp_model(model)
    is_pointmae = is_pointmae_model(model)

    if is_pointmae:
        # Point-MAE architecture: freeze encoder components, reinit classifier
        logger.info("  Detected Point-MAE PointTransformer architecture")

        # Freeze encoder
        model, frozen_param_count, frozen_components = freeze_pointmae_encoder(model)

        # Re-initialize classifier
        model, trainable_param_count = reinit_pointmae_classifier(model)

        # Verify frozen/trainable status
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"\n  Parameter Summary:")
        logger.info(
            f"    Frozen params:     {frozen_param_count:,} ({frozen_param_count / total_params * 100:.1f}%)"
        )
        logger.info(
            f"    Trainable params:  {trainable_param_count:,} ({trainable_param_count / total_params * 100:.1f}%)"
        )
        logger.info(f"    Total params:      {total_params:,}")

        return model

    elif is_pointmlp:
        # PointMLP architecture: freeze encoder components
        logger.info("  Detected PointMLP architecture")

        frozen_components = []
        frozen_param_count = 0

        # Freeze embedding layer
        for param in model.embedding.parameters():
            param.requires_grad = False
            frozen_param_count += param.numel()
        frozen_components.append(
            f"embedding ({sum(p.numel() for p in model.embedding.parameters()):,} params)"
        )

        # Freeze local grouper list
        for param in model.local_grouper_list.parameters():
            param.requires_grad = False
            frozen_param_count += param.numel()
        frozen_components.append(
            f"local_grouper_list ({sum(p.numel() for p in model.local_grouper_list.parameters()):,} params)"
        )

        # Freeze pre-extraction blocks
        for param in model.pre_blocks_list.parameters():
            param.requires_grad = False
            frozen_param_count += param.numel()
        frozen_components.append(
            f"pre_blocks_list ({sum(p.numel() for p in model.pre_blocks_list.parameters()):,} params)"
        )

        # Freeze post-extraction blocks
        for param in model.pos_blocks_list.parameters():
            param.requires_grad = False
            frozen_param_count += param.numel()
        frozen_components.append(
            f"pos_blocks_list ({sum(p.numel() for p in model.pos_blocks_list.parameters()):,} params)"
        )

        # Freeze activation (no learnable params, but included for completeness)
        if hasattr(model, "act"):
            for param in model.act.parameters():
                param.requires_grad = False

        logger.info("  PointMLP encoder components frozen ❄️:")
        for comp in frozen_components:
            logger.info(f"    - {comp}")
        logger.info(f"  Total frozen params: {frozen_param_count:,}")

    elif hasattr(model, "encoder"):
        # BaseCls architecture: standard encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen_param_count = sum(p.numel() for p in model.encoder.parameters())
        logger.info(f"  Encoder frozen ❄️ ({frozen_param_count:,} params)")
    else:
        raise ValueError(
            "Model does not have 'encoder' attribute or recognized PointMLP structure"
        )

    # Re-initialize classifier
    if is_pointmlp and hasattr(model, "classifier"):
        # PointMLP architecture: classifier is nn.Sequential
        logger.info("  Re-initializing PointMLP classifier...")

        trainable_components = []
        trainable_param_count = 0

        # Re-initialize all Linear layers in the classifier
        for module in model.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Ensure classifier parameters are trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
            trainable_param_count += param.numel()

        trainable_components.append(f"classifier ({trainable_param_count:,} params)")

        logger.info("  PointMLP classifier re-initialized 🔄:")
        for comp in trainable_components:
            logger.info(f"    - {comp}")

    elif hasattr(model, "prediction"):
        # BaseCls architecture: prediction head
        old_prediction = model.prediction

        # Re-build prediction head with same architecture but fresh weights
        if hasattr(old_prediction, "head"):
            # ClsHead structure - rebuild the head
            from openpoints.models.classification.cls_base import ClsHead

            # Extract original parameters
            in_channels = (
                old_prediction.head[0].in_features
                if hasattr(old_prediction.head[0], "in_features")
                else None
            )

            if in_channels is None:
                # Fallback: get from encoder output
                in_channels = (
                    model.encoder.out_channels
                    if hasattr(model.encoder, "out_channels")
                    else 512
                )

            # Re-initialize existing head
            for module in old_prediction.head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            logger.info(
                f"  Classifier re-initialized 🔄 (in_channels={in_channels}, num_classes={num_classes})"
            )
        else:
            raise ValueError("Prediction head does not have expected structure")

        # Ensure classifier parameters are trainable
        for param in model.prediction.parameters():
            param.requires_grad = True

    else:
        raise ValueError("Model does not have 'prediction' or 'classifier' attribute")

    # Verify frozen/trainable status
    if is_pointmlp:
        # PointMLP: calculate from encoder components
        frozen_params = (
            sum(p.numel() for p in model.embedding.parameters())
            + sum(p.numel() for p in model.local_grouper_list.parameters())
            + sum(p.numel() for p in model.pre_blocks_list.parameters())
            + sum(p.numel() for p in model.pos_blocks_list.parameters())
        )
        trainable_params = sum(
            p.numel() for p in model.classifier.parameters() if p.requires_grad
        )
    else:
        # BaseCls: calculate from encoder and prediction
        frozen_params = sum(
            p.numel() for p in model.encoder.parameters() if not p.requires_grad
        )
        trainable_params = sum(
            p.numel() for p in model.prediction.parameters() if p.requires_grad
        )

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"\n  📊 Parameter Summary:")
    logger.info(
        f"    Frozen params:     {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)"
    )
    logger.info(
        f"    Trainable params:  {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)"
    )
    logger.info(f"    Total params:      {total_params:,}")

    return model


@torch.no_grad()
def validate(model, val_loader, cfg, device, is_pointmae=False):
    """Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        cfg: OpenPoint config
        device: Device
        is_pointmae: If True, model expects tensor input instead of dict
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

        # Forward pass - Point-MAE expects tensor, others expect dict
        if is_pointmae:
            logits = model(data["pos"])
        else:
            logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    macc, overall_acc, accs = cm.cal_acc(cm.tp, cm.count)
    return macc, overall_acc, accs


@hydra.main(config_path="cfgs_pruning", config_name="class_balanced", version_base=None)
def main(cfg: DictConfig):
    """Main class-balanced retraining workflow."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("Starting Class-Balanced Retraining")
    logger.info("=" * 80)

    # 1. Load and merge configs
    logger.info(f"Loading PointNeXt config: {cfg.pointnext_config}")
    pointnext_cfg = load_pointnext_config(cfg.pointnext_config)

    logger.info("Merging with class-balanced config...")
    full_cfg = merge_class_balanced_config(pointnext_cfg, cfg)

    # Set up structured checkpoint directory: checkpoints/class_balanced/{dataset}/{model}/
    dataset_name = cfg.dataset
    model_name = cfg.model
    base_ckpt_dir = os.path.join(
        "checkpoints", "class_balanced", dataset_name, model_name
    )

    # Generate unique run name and create run-specific subdirectory
    import time

    import shortuuid

    exp_name = "class_balanced_retrain"
    expid = time.strftime("%Y%m%d-%H%M%S-") + str(shortuuid.uuid())
    run_name = f"{exp_name}-{expid}"

    # Set ckpt_dir to run-specific directory to avoid checkpoint overlap
    full_cfg.openpoint.ckpt_dir = os.path.join(base_ckpt_dir, run_name)
    full_cfg.openpoint.run_name = run_name
    os.makedirs(full_cfg.openpoint.ckpt_dir, exist_ok=True)
    logger.info(f"Base checkpoint directory: {base_ckpt_dir}")
    logger.info(f"Run-specific directory: {full_cfg.openpoint.ckpt_dir}")
    logger.info(f"Run name: {run_name}")

    # Log config
    logger.info("Merged config:")
    logger.info(full_cfg)

    # 2. Initialize WandB (log only Hydra config, not merged PointNeXt config)
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

    # 3. Load pre-trained model
    logger.info("Loading pre-trained model...")

    # Check model type for architecture-specific loading
    model_type = cfg.class_balanced.get("model_type", "auto")
    use_pointmae = model_type == "pointmae"

    if use_pointmae:
        # Point-MAE specific loading
        logger.info("Loading Point-MAE model...")
        pretrained_model, pretrained_cfg = load_pointmae_model(
            cfg.class_balanced.pretrained_ckpt,
            config=cfg.class_balanced.get("pointmae_config"),
            device=device,
        )
        # Add get_logits_loss() method for StandardTrainer compatibility
        add_get_logits_loss(pretrained_model)
    else:
        # Standard PointNeXt/PointMLP loading
        pretrained_model, pretrained_cfg = load_model_from_checkpoint(
            cfg.class_balanced.pretrained_ckpt, cfg.class_balanced.get("pretrained_config")
        )
        pretrained_model.to(device)

    # Auto-detect if model is Point-MAE (for validation)
    if model_type == "auto":
        use_pointmae = is_pointmae_model(pretrained_model)
        if use_pointmae:
            logger.info("Auto-detected Point-MAE architecture")
            add_get_logits_loss(pretrained_model)

    # Save before-retraining checkpoint for visualization
    logger.info("Saving BEFORE model checkpoint...")
    os.makedirs(full_cfg.openpoint.ckpt_dir, exist_ok=True)
    before_ckpt_path = os.path.join(
        full_cfg.openpoint.ckpt_dir, "model_before_retrain.pth"
    )
    torch.save(
        {"model": pretrained_model.state_dict(), "config": pretrained_cfg},
        before_ckpt_path,
    )
    logger.info(f"  Saved: {before_ckpt_path}")

    # 4. Set up criterion (loaded model may not have it)
    from openpoints.loss import build_criterion_from_cfg

    if not hasattr(pretrained_model, "criterion") or pretrained_model.criterion is None:
        logger.info("Setting up criterion for loaded model...")
        pretrained_model.criterion = build_criterion_from_cfg(
            full_cfg.openpoint.criterion_args
        )

    # 5. Freeze encoder and re-initialize classifier
    model = freeze_encoder_reinit_classifier(
        pretrained_model,
        num_classes=full_cfg.openpoint.num_classes,
        device=device,
    )

    from openpoints.utils import cal_model_parm_nums

    model_size = cal_model_parm_nums(model)
    trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {model_size / 1e6:.2f}M")
    logger.info(
        f"Trainable parameters: {trainable_size / 1e6:.2f}M ({100 * trainable_size / model_size:.1f}%)"
    )

    # 6. Load dataset
    logger.info("Loading training dataset...")
    num_points = full_cfg.openpoint.get("num_points", 8192)

    if use_pointmae and cfg.class_balanced.get("pointmae_data_dir"):
        # Use Point-MAE dataset loaders directly
        pointmae_data_dir = cfg.class_balanced.pointmae_data_dir

        # Detect dataset type: ScanObjectNN vs ModelNet40
        dataset_name = cfg.get("dataset", "").lower()
        is_scanobjectnn = "scanobjectnn" in dataset_name or "scanobjectnn" in pointmae_data_dir.lower()

        logger.info(f"Loading Point-MAE dataset from {pointmae_data_dir}")
        logger.info(f"  num_points={num_points}, dataset_type={'ScanObjectNN' if is_scanobjectnn else 'ModelNet40'}")

        if is_scanobjectnn:
            # Use ScanObjectNN loader for Point-MAE
            variant = cfg.class_balanced.get("scanobjectnn_variant", "hardest")
            logger.info(f"  ScanObjectNN variant: {variant}")
            train_dataset = PointMAEScanObjectNN(
                data_dir=pointmae_data_dir,
                split="train",
                variant=variant,
                num_points=num_points,
            )
        else:
            # Use ModelNet40 loader with FPS cache
            logger.info(f"  cache_only=True")
            train_dataset = PointMAEModelNet40(
                data_dir=pointmae_data_dir,
                split="train",
                num_points=num_points,
                use_normals=False,
                cache_only=True,  # Use pre-built cache
            )
        logger.info(f"Training dataset size: {len(train_dataset)}")
    else:
        # Use OpenPoints dataloader for non-Point-MAE models
        train_dataset_loader = build_dataloader_from_cfg(
            full_cfg.openpoint.batch_size,
            full_cfg.openpoint.dataset,
            full_cfg.openpoint.dataloader,
            datatransforms_cfg=full_cfg.openpoint.datatransforms,
            split="train",
            distributed=False,
        )
        train_dataset = train_dataset_loader.dataset
        logger.info(f"Training dataset size: {len(train_dataset)}")

        # Wrap dataset for Point-MAE if using OpenPoints loader
        if use_pointmae:
            logger.info(f"Wrapping dataset with PointMAEDatasetWrapper (num_points={num_points})")
            train_dataset = PointMAEDatasetWrapper(train_dataset, num_points=num_points)

    # 7. Build class-balanced dataloader
    if cfg.class_balanced.use_uniform_sampler:
        logger.info("Using UniformClassSampler for class-balanced training")
        sampler = UniformClassSampler(
            dataset=train_dataset,
            batch_size=full_cfg.openpoint.batch_size,
            num_classes=full_cfg.openpoint.num_classes,
            num_samples=cfg.class_balanced.get("samples_per_epoch")
            or len(train_dataset),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=full_cfg.openpoint.batch_size,
            sampler=sampler,
            num_workers=full_cfg.openpoint.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        logger.info("Using standard DataLoader (no class balancing)")
        train_loader = train_dataset_loader

    # 8. Load validation dataset
    val_batch_size = full_cfg.openpoint.get("val_batch_size", full_cfg.openpoint.batch_size)

    if use_pointmae and cfg.class_balanced.get("pointmae_data_dir"):
        # Use Point-MAE dataset loaders directly
        logger.info(f"Loading Point-MAE validation dataset from {pointmae_data_dir}")

        if is_scanobjectnn:
            # Use ScanObjectNN loader for Point-MAE
            val_dataset = PointMAEScanObjectNN(
                data_dir=pointmae_data_dir,
                split="test",
                variant=variant,
                num_points=num_points,
            )
        else:
            # Use ModelNet40 loader with FPS cache
            val_dataset = PointMAEModelNet40(
                data_dir=pointmae_data_dir,
                split="test",  # ModelNet40 uses 'test' split for validation
                num_points=num_points,
                use_normals=False,
                cache_only=True,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=full_cfg.openpoint.dataloader.num_workers,
            pin_memory=True,
        )
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    else:
        # Use OpenPoints dataloader
        val_dataset_loader = build_dataloader_from_cfg(
            val_batch_size,
            full_cfg.openpoint.dataset,
            full_cfg.openpoint.dataloader,
            datatransforms_cfg=full_cfg.openpoint.datatransforms,
            split="val",
            distributed=False,
        )

        # Wrap validation dataset for Point-MAE if needed
        if use_pointmae:
            val_dataset = PointMAEDatasetWrapper(val_dataset_loader.dataset, num_points=num_points)
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=full_cfg.openpoint.dataloader.num_workers,
                pin_memory=True,
            )
        else:
            val_loader = val_dataset_loader

    # 9. Build optimizer and scheduler (only for classifier parameters)
    optimizer = build_optimizer_from_cfg(
        model, lr=full_cfg.openpoint.lr, **full_cfg.openpoint.optimizer
    )
    scheduler = build_scheduler_from_cfg(full_cfg.openpoint, optimizer)

    # Verify optimizer is only training classifier
    opt_param_count = sum(
        p.numel() for group in optimizer.param_groups for p in group["params"]
    )
    logger.info(f"Optimizer tracking {opt_param_count / 1e6:.2f}M parameters")

    # 10. Training loop
    logger.info("=" * 80)
    logger.info("Starting class-balanced retraining...")
    logger.info(f"Epochs: {full_cfg.openpoint.epochs}")
    logger.info(f"Batch size: {full_cfg.openpoint.batch_size}")
    logger.info(f"Learning rate: {full_cfg.openpoint.lr}")
    logger.info("=" * 80)

    best_val_acc = 0.0
    best_epoch = 0

    trainer = StandardTrainer(model, full_cfg.openpoint, device)

    for epoch in range(1, full_cfg.openpoint.epochs + 1):
        # Train
        metrics = trainer.train_one_epoch(train_loader, optimizer, scheduler, epoch)
        train_loss = metrics.loss
        train_macc = metrics.macc
        train_oa = metrics.overall_acc

        # Validate
        if epoch % full_cfg.openpoint.val_freq == 0:
            val_macc, val_oa, val_accs = validate(
                model, val_loader, full_cfg.openpoint, device, is_pointmae=use_pointmae
            )

            is_best = val_oa > best_val_acc
            if is_best:
                best_val_acc = val_oa
                best_epoch = epoch
                logger.info(f"*** New best @ epoch {epoch}: val_oa={val_oa:.2f}% ***")

            # Log to WandB
            wandb_log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/macc": train_macc,
                "train/oa": train_oa,
                "val/macc": val_macc,
                "val/oa": val_oa,
                "best_val_oa": best_val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }

            wandb.log(wandb_log)

            # Save checkpoint (based on average performance)
            if is_best:
                save_checkpoint(
                    full_cfg.openpoint,
                    model,
                    epoch,
                    optimizer,
                    scheduler,
                    additioanl_dict={"best_val": best_val_acc},
                    is_best=True,
                )

            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.3f}, train_oa={train_oa:.2f}, "
                f"val_oa={val_oa:.2f}, best_val_oa={best_val_acc:.2f} (@epoch {best_epoch})"
            )

        if full_cfg.openpoint.sched_on_epoch:
            scheduler.step(epoch)

    # 11. Save after-retraining checkpoint for visualization
    logger.info("=" * 80)
    logger.info("Saving AFTER model checkpoint...")
    after_ckpt_path = os.path.join(
        full_cfg.openpoint.ckpt_dir, "model_after_retrain.pth"
    )

    save_dict = {
        "model": model.state_dict(),
        "config": full_cfg.openpoint,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }

    torch.save(save_dict, after_ckpt_path)
    logger.info(f"  Saved: {after_ckpt_path}")

    logger.info("=" * 80)
    logger.info("Class-balanced retraining complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% @ epoch {best_epoch}")
    logger.info(f"Before checkpoint: {before_ckpt_path}")
    logger.info(f"After checkpoint: {after_ckpt_path}")
    logger.info("=" * 80)

    # 12. Auto-generate comparison visualization if enabled
    if cfg.class_balanced.get("auto_plot", True):
        from utils.visualization import auto_generate_comparison

        vis_model_name = "Point-MAE" if use_pointmae else "PointNeXt"
        auto_generate_comparison(
            before_ckpt_path=before_ckpt_path,
            after_model=model,
            train_dataset=train_dataset,
            full_cfg=full_cfg,
            device=device,
            model_name=vis_model_name,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
