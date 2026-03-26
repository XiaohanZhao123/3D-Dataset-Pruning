"""Pruning script with flexible model loading.

This script:
1. Loads a scorer model (single or multi-head) for sample selection
2. Scores training samples using configurable methods
3. Selects samples based on scoring
4. Trains a student model on pruned data with optional KD
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.utils import ConfusionMatrix, save_checkpoint
from utils.config_loader import load_pointnext_config, merge_pruning_config
from utils.metrics import (
    build_hmt_wandb_log,
    compute_class_counts,
    compute_group_accuracies,
    define_hmt_classes,
)
from utils.model_loading import load_standard_model
from utils.train import prepare_data_dict, setup_experiment
from utils.trainers import get_trainer
from utils.builders import (
    build_scorer_model,
    build_train_dataset,
    build_dataloader,
    build_student_model,
    detect_model_type,
)
from pruning.functional import hybrid_select_score_based, double_budget_hybrid_merge

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate(model, val_loader, cfg):
    """Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        cfg: OpenPoint config
    """
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

        # Point-MAE expects tensor, PointNeXt expects dict
        if hasattr(model, "get_embeddings"):
            logits = model(data["pos"])
        else:
            logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    macc, overall_acc, accs = cm.cal_acc(cm.tp, cm.count)
    return macc, overall_acc, accs


def load_scorer_model(cfg, device):
    """Load scorer model based on config.

    Returns:
        (model, requires_grad)
    """
    from pruning.balanced_scorers import SCORER_REGISTRY
    from utils.model_loading import load_standard_model

    scorer_method = str(cfg.pruning.get("scorer", "loss")).lower()
    scorer_cls = SCORER_REGISTRY.get(scorer_method)
    if scorer_cls is None:
        available = sorted(set(SCORER_REGISTRY.keys()))
        raise ValueError(f"Unknown scorer '{scorer_method}'. Available: {available}")

    requires_grad = scorer_cls.requires_grad
    freeze = not requires_grad

    ckpt = cfg.pruning.scorer_checkpoint
    config = cfg.pruning.scorer_config

    logger.info("Loading scorer model...")
    logger.info(f"  Checkpoint: {ckpt}")
    if not freeze:
        logger.info(f"  Gradient-based scorer '{scorer_method}' -> params trainable")

    model, _ = load_standard_model(ckpt, config, device, freeze=freeze)

    logger.info("✓ Loaded scorer model")
    return model, requires_grad


def load_teacher_model(cfg, device, scorer_model):
    """Load teacher model for KD, or reuse scorer if not specified.

    Auto-detects model type (Point-MAE vs PointNeXt) from checkpoint path.

    Returns:
        model or None if KD disabled
    """
    from utils.builders import detect_model_type
    from utils.model_loading import load_standard_model

    if not cfg.pruning.get("use_kd", False):
        return None

    teacher_ckpt = cfg.pruning.get("teacher_checkpoint")

    # If no separate teacher specified, reuse scorer
    if not teacher_ckpt:
        logger.info("No separate teacher specified, reusing scorer model for KD")
        return scorer_model

    logger.info("Loading separate KD teacher...")
    logger.info(f"  Checkpoint: {teacher_ckpt}")

    # Detect model type from checkpoint path
    model_type = detect_model_type(teacher_ckpt)
    logger.info(f"  Detected type: {model_type}")

    if model_type == "pointmae":
        # Load Point-MAE model
        from utils.pointmae_utils import load_pointmae_model, add_get_logits_loss
        from utils.builders import add_get_embeddings

        model, _ = load_pointmae_model(teacher_ckpt, device=device)
        add_get_logits_loss(model)
        add_get_embeddings(model)

        # Freeze for KD
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    else:
        # Load PointNeXt/PointMLP model
        teacher_config = cfg.pruning.get("teacher_config", cfg.pruning.scorer_config)
        model, _ = load_standard_model(teacher_ckpt, teacher_config, device, freeze=True)

    logger.info("✓ Loaded teacher model")
    return model


@hydra.main(
    config_path="cfgs_pruning", config_name="pruning_balanced", version_base=None
)
def main(cfg: DictConfig):
    """Main pruning workflow."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("Dataset Pruning Pipeline")
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Auto-tune for fixed input sizes
    logger.info(f"Using device: {device}")

    # 3. Load scorer model (unified builder handles Point-MAE vs PointNeXt)
    logger.info("=" * 80)
    # Check if scorer needs gradients (e.g., grad_norm, grad_herding)
    from pruning.balanced_scorers import SCORER_REGISTRY
    scorer_method = str(cfg.pruning.get("scorer", "loss")).lower()
    scorer_cls = SCORER_REGISTRY.get(scorer_method)
    freeze = not (scorer_cls.requires_grad if scorer_cls else False)
    if not freeze:
        logger.info(f"Gradient-based scorer '{scorer_method}' -> model params trainable")
    scorer_model, model_type = build_scorer_model(cfg, device, freeze=freeze)

    # 4. Load dataset (unified builder handles Point-MAE vs OpenPoints data)
    logger.info("=" * 80)
    logger.info("Loading training dataset...")
    train_dataset = build_train_dataset(full_cfg, "train")
    train_loader = build_dataloader(full_cfg, train_dataset, "train")
    logger.info(f"Original dataset size: {len(train_dataset)}")

    # Compute original class distribution for HMT grouping
    num_classes = full_cfg.openpoint.num_classes
    original_counts = compute_class_counts(train_loader.dataset, num_classes)
    original_hmt = define_hmt_classes(original_counts)
    logger.info(
        f"Original HMT split: head={len(original_hmt['head'])}, "
        f"medium={len(original_hmt['medium'])}, tail={len(original_hmt['tail'])} classes"
    )

    # 5. Create non-shuffled dataloader for scoring (larger batch for inference)
    scoring_batch_size = cfg.pruning.get(
        "scoring_batch_size", full_cfg.openpoint.batch_size * 2
    )
    num_workers = full_cfg.openpoint.dataloader.num_workers
    scoring_loader = DataLoader(
        train_dataset,
        batch_size=scoring_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    logger.info(f"Scoring batch size: {scoring_batch_size}")

    # 5b. Build validation loader early (needed for DRoP per-class metrics)
    val_batch_size = cfg.pruning.get(
        "val_batch_size", full_cfg.openpoint.batch_size * 2
    )
    val_dataset = build_train_dataset(full_cfg, "val")
    val_loader_for_scoring = build_dataloader(
        full_cfg, val_dataset, "val", batch_size=val_batch_size
    )
    logger.info(f"Validation dataset for scoring: {len(val_dataset)} samples")

    # 6. Compute scores
    logger.info("=" * 80)
    from pruning.balanced_scorers import SCORER_REGISTRY, get_scorer

    scorer_method = str(cfg.pruning.get("scorer", "loss")).lower()
    scorer_cls = SCORER_REGISTRY[scorer_method]

    # Warn if mode will be overridden
    if scorer_cls.mode_override and cfg.pruning.mode != scorer_cls.mode_override:
        logger.warning(
            f"⚠️ {scorer_method} scorer ignores mode='{cfg.pruning.mode}', "
            f"will use '{scorer_cls.mode_override}'"
        )

    # Build scorer kwargs - only include CCS parameters when using CCS mode
    scorer_kwargs = {"device": str(device)}
    if cfg.pruning.mode == "ccs":
        scorer_kwargs["mislabel_ratio"] = cfg.pruning.get("mislabel_ratio", 0.0)
        scorer_kwargs["num_strata"] = cfg.pruning.get("num_strata", 50)
        scorer_kwargs["ccscp_min_samples"] = cfg.pruning.get("ccscp_min_samples", 1)

    scorer = get_scorer(
        scorer_method,
        scorer_model,
        full_cfg.openpoint,
        **scorer_kwargs,
    )

    # Check for hybrid selection mode (needed before compute to avoid wasted work)
    use_hybrid = cfg.pruning.get("hybrid", False)
    hybrid_ratio = cfg.pruning.get("hybrid_per_class_ratio", 0.5)
    hybrid_phase2_scorer_name = cfg.pruning.get("hybrid_phase2_scorer")

    # Fail fast if scorer doesn't support hybrid
    if use_hybrid and not scorer_cls.supports_hybrid:
        raise ValueError(
            f"Scorer '{scorer_method}' does not support hybrid mode. "
            f"It has its own budget allocation logic."
        )

    # Build phase 2 scorer if different from phase 1
    scorer_phase2 = None
    if use_hybrid and hybrid_phase2_scorer_name:
        phase2_scorer_cls = SCORER_REGISTRY.get(hybrid_phase2_scorer_name)
        if phase2_scorer_cls is None:
            available = sorted(set(SCORER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown phase2 scorer '{hybrid_phase2_scorer_name}'. Available: {available}"
            )
        if not phase2_scorer_cls.supports_hybrid:
            raise ValueError(
                f"Phase2 scorer '{hybrid_phase2_scorer_name}' does not support hybrid mode."
            )

        logger.info(f"Building phase 2 scorer: {hybrid_phase2_scorer_name}")
        scorer_phase2 = get_scorer(
            hybrid_phase2_scorer_name,
            scorer_model,
            full_cfg.openpoint,
            **scorer_kwargs,
        )

    # For selection-based scorers with hybrid mode, skip initial compute
    # (hybrid mode will recompute in two phases anyway)
    skip_initial_compute = use_hybrid and not scorer_cls.is_score_based

    if skip_initial_compute:
        logger.info(f"Skipping initial {scorer_method} compute (hybrid mode will recompute)")
        scores, labels, indices = None, None, None
    else:
        logger.info(f"Computing {scorer_method} scores...")
        scores, labels, indices = scorer.compute(
            scoring_loader,
            total_samples=cfg.pruning.total_samples,
            per_class=cfg.pruning.per_class,
            num_classes=full_cfg.openpoint.num_classes,
            grad_scope=cfg.pruning.get("grad_norm_scope", "head"),
            sigma=cfg.pruning.get("submodular_sigma"),
            space=cfg.pruning.get("submodular_space", "embedding"),
            rbf_algorithm=cfg.pruning.get("rbf_algorithm", "apricot"),
            # Loss scorer options
            loss_type=cfg.pruning.get("loss_type", "ce"),
            focal_gamma=cfg.pruning.get("focal_gamma", 2.0),
            cb_beta=cfg.pruning.get("cb_beta", 0.9999),
            # DRoP-specific: pass validation loader for per-class metrics
            val_loader=val_loader_for_scoring,
        )

    # 7. Select samples
    logger.info("=" * 80)
    logger.info("Selecting samples:")
    logger.info(f"  Method: {scorer_method}")
    logger.info(f"  Total samples: {cfg.pruning.total_samples}")

    if use_hybrid:
        logger.info(f"  Hybrid mode: {hybrid_ratio*100:.0f}% per-class, {(1-hybrid_ratio)*100:.0f}% global")
        logger.info(f"  Phase 1 scorer: {scorer_method}")
        if scorer_phase2 is not None:
            logger.info(f"  Phase 2 scorer: {hybrid_phase2_scorer_name}")
        else:
            logger.info(f"  Phase 2 scorer: {scorer_method} (same as phase 1)")
    else:
        logger.info(f"  Per class: {cfg.pruning.per_class}")

    selection_mode = scorer_cls.mode_override or cfg.pruning.mode
    if scorer_cls.mode_override:
        logger.info(f"  Mode: {selection_mode} (forced by {scorer_method})")
    else:
        logger.info(f"  Mode: {selection_mode}")

    # Hybrid selection: two-phase selection (per-class then global)
    if use_hybrid:
        if scorer_cls.is_score_based:
            # Score-based scorers: use existing scores for both phases
            logger.info("Using score-based hybrid selection...")
            _, _, selected_indices = hybrid_select_score_based(
                scores,
                labels,
                indices,
                total_samples=cfg.pruning.total_samples,
                hybrid_per_class_ratio=hybrid_ratio,
                mode=selection_mode,
                num_classes=full_cfg.openpoint.num_classes,
            )
        else:
            # Selection-based scorers: need to recompute for each phase
            logger.info("Using selection-based hybrid selection...")
            import math

            phase1_per_class = math.floor(
                hybrid_ratio * cfg.pruning.total_samples / full_cfg.openpoint.num_classes
            )
            phase1_budget = phase1_per_class * full_cfg.openpoint.num_classes
            phase2_budget = cfg.pruning.total_samples - phase1_budget

            logger.info(f"  Phase 1: {phase1_per_class} per class × {full_cfg.openpoint.num_classes} = {phase1_budget}")
            logger.info(f"  Phase 2: {phase2_budget} global")

            # Phase 1: Per-class selection (recompute with phase1 budget)
            scores1, labels1, indices1 = scorer.compute(
                scoring_loader,
                total_samples=phase1_budget,
                per_class=True,
                num_classes=full_cfg.openpoint.num_classes,
                grad_scope=cfg.pruning.get("grad_norm_scope", "head"),
                sigma=cfg.pruning.get("submodular_sigma"),
                space=cfg.pruning.get("submodular_space", "embedding"),
                rbf_algorithm=cfg.pruning.get("rbf_algorithm", "apricot"),
            )
            phase1_indices = scorer.select(
                scores1, labels1, indices1,
                total_samples=phase1_budget,
                per_class=True,
                mode=selection_mode,
                num_classes=full_cfg.openpoint.num_classes,
            )
            logger.info(f"  Phase 1 selected: {len(phase1_indices)} indices")

            # Phase 2: Global selection on FULL dataset with DOUBLE budget
            # This ensures we have enough non-overlapping samples to fill the gap
            # even if Phase 1 and Phase 2 have significant overlap
            if phase2_budget > 0:
                # Use phase 2 scorer if specified, else use phase 1 scorer
                phase2_scorer = scorer_phase2 if scorer_phase2 is not None else scorer

                # Select with full total_samples budget (2x phase2_budget)
                # This guarantees enough unique samples after overlap removal
                phase2_selection_budget = cfg.pruning.total_samples
                logger.info(f"  Phase 2: selecting {phase2_selection_budget} on FULL dataset (double budget strategy)")

                scores2, labels2, indices2 = phase2_scorer.compute(
                    scoring_loader,  # FULL dataset, not subset!
                    total_samples=phase2_selection_budget,
                    per_class=False,
                    num_classes=full_cfg.openpoint.num_classes,
                    grad_scope=cfg.pruning.get("grad_norm_scope", "head"),
                    sigma=cfg.pruning.get("submodular_sigma"),
                    space=cfg.pruning.get("submodular_space", "embedding"),
                    rbf_algorithm=cfg.pruning.get("rbf_algorithm", "apricot"),
                )
                # Phase 2 returns indices in ranked order (important for merge)
                phase2_indices = phase2_scorer.select(
                    scores2, labels2, indices2,
                    total_samples=phase2_selection_budget,
                    per_class=False,
                    mode=selection_mode,
                    num_classes=full_cfg.openpoint.num_classes,
                )
                logger.info(f"  Phase 2 selected: {len(phase2_indices)} indices")

                # Compute overlap statistics
                phase1_set = set(phase1_indices)
                overlap_count = sum(1 for idx in phase2_indices if idx in phase1_set)
                overlap_pct = 100 * overlap_count / len(phase2_indices)
                logger.info(f"  Overlap with Phase 1: {overlap_count} ({overlap_pct:.1f}%)")
            else:
                phase2_indices = []

            # Merge using double-budget strategy
            selected_indices = double_budget_hybrid_merge(
                phase1_indices, phase2_indices, cfg.pruning.total_samples
            )
            logger.info(f"  Final selection: {len(selected_indices)} samples")
    else:
        # Standard selection (non-hybrid)
        selected_indices = scorer.select(
            scores,
            labels,
            indices,
            total_samples=cfg.pruning.total_samples,
            per_class=cfg.pruning.per_class,
            mode=selection_mode,
            num_classes=full_cfg.openpoint.num_classes,
            # NUCS-specific parameters (ignored by other scorers)
            nucs_aggregation=cfg.pruning.get("nucs_aggregation", "mean"),
            nucs_min_samples=cfg.pruning.get("nucs_min_samples", 1),
            nucs_endpoint=cfg.pruning.get("nucs_endpoint", 0.75),
            nucs_use_krr=cfg.pruning.get("nucs_use_krr", False),
            nucs_endpoint_candidates=cfg.pruning.get("nucs_endpoint_candidates"),
        )

    pruned_dataset = Subset(train_loader.dataset, selected_indices)
    logger.info(f"Pruned dataset size: {len(pruned_dataset)}")

    # Compute pruned class distribution for HMT grouping
    # For hybrid mode, always compute since it's a mix of per-class and global
    pruned_hmt = None
    if use_hybrid or not cfg.pruning.per_class:
        pruned_counts = compute_class_counts(pruned_dataset, num_classes)
        pruned_hmt = define_hmt_classes(pruned_counts)
        logger.info(
            f"Pruned HMT split: head={len(pruned_hmt['head'])}, "
            f"medium={len(pruned_hmt['medium'])}, tail={len(pruned_hmt['tail'])} classes"
        )

    # 8. Build pruned dataloader
    # drop_last=True prevents BatchNorm failure when last batch has size 1
    pruned_loader = DataLoader(
        pruned_dataset,
        batch_size=full_cfg.openpoint.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    # 9. Reuse validation loader built earlier (for DRoP and other scorers)
    val_loader = val_loader_for_scoring
    logger.info(f"Validation dataset size: {len(val_dataset)}, batch size: {val_batch_size}")

    # 10. Build student model (unified builder)
    logger.info("=" * 80)
    logger.info("Building student model...")
    student_model, model_size = build_student_model(full_cfg, device)

    optimizer = build_optimizer_from_cfg(
        student_model, lr=full_cfg.openpoint.lr, **full_cfg.openpoint.optimizer
    )
    scheduler = build_scheduler_from_cfg(full_cfg.openpoint, optimizer)

    exp_name = f"pruning_{scorer_method}_{cfg.pruning.mode}_{cfg.pruning.total_samples}"
    run_name = setup_experiment(full_cfg.openpoint, exp_name)

    # 11. Load teacher model for KD
    logger.info("=" * 80)
    teacher_model = load_teacher_model(cfg, device, scorer_model)

    # 11b. Compute prototypes for Proto-RKD if enabled
    prototypes = None
    if cfg.pruning.get("use_proto_rkd", False) and teacher_model is not None:
        from utils.prototypes import compute_class_prototypes

        logger.info("=" * 80)
        logger.info("Computing class prototypes for Proto-RKD...")
        prototypes = compute_class_prototypes(
            model=teacher_model,
            dataloader=scoring_loader,  # Full dataset, no shuffle
            cfg=full_cfg.openpoint,
            device=device,
            num_classes=full_cfg.openpoint.num_classes,
            num_passes=cfg.pruning.get("proto_num_passes", 5),
        )
        logger.info(f"✓ Prototypes computed: {prototypes.shape}")

    # 12. Create trainer
    logger.info("=" * 80)
    trainer = get_trainer(
        student_model,
        full_cfg.openpoint,
        device,
        cfg.pruning,
        teacher_model=teacher_model,
        prototypes=prototypes,
    )

    # Log training config
    use_kd = cfg.pruning.get("use_kd", False)

    if use_kd:
        logger.info("Training Mode: Knowledge Distillation")
        logger.info(f"  KD Alpha: {cfg.pruning.get('kd_alpha', 0.5)}")
        logger.info(f"  KD Temperature: {cfg.pruning.get('kd_temperature', 3.0)}")
        if cfg.pruning.get("use_rkd", False):
            logger.info("  RKD: Enabled")
        if cfg.pruning.get("use_proto_rkd", False):
            logger.info(
                f"  Proto-RKD: Enabled (weight={cfg.pruning.get('proto_weight', 1.0)}, "
                f"τ={cfg.pruning.get('proto_tau', 0.1)}, "
                f"passes={cfg.pruning.get('proto_num_passes', 5)})"
            )
    else:
        logger.info("Training Mode: Standard (No KD)")

    logger.info(f"Epochs: {full_cfg.openpoint.epochs}")
    logger.info(f"Batch size: {full_cfg.openpoint.batch_size}")
    logger.info(f"Learning rate: {full_cfg.openpoint.lr}")
    logger.info("=" * 80)

    # 13. Training loop
    best_val_acc = 0.0
    best_epoch = 0
    best_macc = 0.0
    best_head_acc = 0.0
    best_medium_acc = 0.0
    best_tail_acc = 0.0

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
                # Capture HMT at this epoch (original grouping only)
                orig_group_accs = compute_group_accuracies(val_accs, original_hmt)
                best_head_acc = orig_group_accs["head"]
                best_medium_acc = orig_group_accs["medium"]
                best_tail_acc = orig_group_accs["tail"]
                logger.info(f"*** New best @ epoch {epoch}: val_oa={val_oa:.2f}% ***")

            wandb_log = {
                "epoch": epoch,
                "val/macc": val_macc,
                "val/oa": val_oa,
                "best_val_oa": best_val_acc,
                "macc_at_best_oa": best_macc,
                "head_acc_at_best_oa": best_head_acc,
                "medium_acc_at_best_oa": best_medium_acc,
                "tail_acc_at_best_oa": best_tail_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            wandb_log.update(metrics.to_wandb_dict("train"))
            wandb_log.update(build_hmt_wandb_log(val_accs, original_hmt, pruned_hmt))
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
                f"Epoch {epoch}: train_loss={metrics.loss:.3f}, train_oa={metrics.overall_acc:.2f}, "
                f"val_oa={val_oa:.2f}, best_val_oa={best_val_acc:.2f} (@epoch {best_epoch})"
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
