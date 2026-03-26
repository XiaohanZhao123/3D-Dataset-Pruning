"""Cross-architecture pruning script.

This script supports different architectures for scorer/teacher vs student:
1. Loads a scorer model for sample selection (architecture A)
2. Scores training samples using configurable methods
3. Selects samples based on scoring
4. Trains a student model (architecture B) on pruned data with RKD + Logit KD

Key feature: RKD (Relational Knowledge Distillation) works across different
embedding dimensions, enabling cross-architecture knowledge transfer.
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
from pruning.functional import hybrid_select_score_based
from utils.model_loading import load_model_by_architecture
from utils.train import prepare_data_dict, setup_experiment
from utils.trainers import get_trainer
from utils.builders import (
    build_scorer_model,
    build_train_dataset,
    build_dataloader,
    build_student_model,
    detect_model_type,
    add_get_embeddings_pointnext,
)

logger = logging.getLogger(__name__)


def build_student_model_cross_arch(student_cfg, training_cfg, device, pruning_cfg=None):
    """Build student model from separate config (for cross-architecture).

    Supports both PointNeXt and Point-MAE student architectures with
    multiple initialization modes (random, pretrain, finetune).

    Args:
        student_cfg: EasyConfig for student model architecture
        training_cfg: Training config (merged) for criterion_args, num_classes, etc.
        device: Target device
        pruning_cfg: Optional pruning config for student_init, checkpoints, etc.

    Returns:
        (model, model_size_M): Model and size in millions of parameters
    """
    from utils.builders import add_get_embeddings, detect_model_type

    # Detect student model type from student config path or content
    student_config_path = pruning_cfg.get("student_config", "") if pruning_cfg else ""
    model_type = _detect_student_type(student_cfg, student_config_path)
    logger.info(f"Detected student model type: {model_type}")

    student_init = pruning_cfg.get("student_init", "pretrain") if pruning_cfg else "pretrain"
    logger.info(f"Student initialization mode: {student_init}")

    if model_type == "pointmae":
        model, model_size = _build_pointmae_student_cross_arch(
            student_cfg, training_cfg, device, pruning_cfg, student_init
        )
    else:
        model, model_size = _build_pointnext_student_cross_arch(
            student_cfg, training_cfg, device, pruning_cfg, student_init
        )

    return model, model_size


def _detect_student_type(student_cfg, student_config_path: str) -> str:
    """Detect student model type from config.

    Args:
        student_cfg: EasyConfig for student model
        student_config_path: Path to student config file

    Returns:
        Model type: "pointmae" or "pointnext"
    """
    path_lower = student_config_path.lower()

    # Check path for hints
    if "pointmae" in path_lower or "mae" in path_lower:
        return "pointmae"

    # Check if config has Point-MAE specific fields
    if hasattr(student_cfg, "pointmae_config") or hasattr(student_cfg, "trans_dim"):
        return "pointmae"

    # Check model config for PointTransformer indicators
    if hasattr(student_cfg, "model"):
        model_name = student_cfg.model.get("NAME", "").lower()
        if "pointtransformer" in model_name or "mae" in model_name:
            return "pointmae"

    return "pointnext"


def _build_pointmae_student_cross_arch(student_cfg, training_cfg, device, pruning_cfg, student_init):
    """Build Point-MAE student model.

    Args:
        student_cfg: EasyConfig for student model
        training_cfg: Training config
        device: Target device
        pruning_cfg: Pruning config with checkpoint paths
        student_init: Initialization mode (random, pretrain, finetune)

    Returns:
        (model, model_size_M)
    """
    from utils.builders import add_get_embeddings
    from utils.pointmae_utils import add_get_logits_loss, load_pointmae_model

    if student_init == "random":
        # Build model with random initialization
        from utils.builders import _build_pointmae_random

        # Create a minimal config for random build
        class MinimalCfg:
            def __init__(self):
                self.pruning = {}

        model = _build_pointmae_random(MinimalCfg(), device)
        logger.info("Built Point-MAE student with RANDOM initialization")

    elif student_init == "pretrain":
        # Load pre-trained encoder (self-supervised), random classifier
        pretrain_ckpt = pruning_cfg.get("student_pretrain_checkpoint")
        if pretrain_ckpt is None:
            pretrain_ckpt = pruning_cfg.get("pretrain_checkpoint")
        if pretrain_ckpt is None:
            # Try to find in standard location
            from pathlib import Path
            scorer_ckpt = pruning_cfg.get("scorer_checkpoint", "")
            if scorer_ckpt:
                pretrain_dir = Path(scorer_ckpt).parent
                possible_pretrain = pretrain_dir / "model_pretrain.pth"
                if possible_pretrain.exists():
                    pretrain_ckpt = str(possible_pretrain)

        if pretrain_ckpt is None:
            raise ValueError(
                "Point-MAE student_init='pretrain' requires a pretrain checkpoint. "
                "Set pruning.student_pretrain_checkpoint or pruning.pretrain_checkpoint."
            )

        logger.info(f"Loading pre-trained Point-MAE encoder from: {pretrain_ckpt}")
        model, _ = load_pointmae_model(pretrain_ckpt, device=device)

    elif student_init == "finetune":
        # Load fine-tuned checkpoint (NOT recommended)
        finetune_ckpt = pruning_cfg.get("student_checkpoint")
        if finetune_ckpt is None:
            finetune_ckpt = pruning_cfg.get("scorer_checkpoint")

        logger.warning("⚠️ student_init='finetune' loads task-specific weights!")
        logger.warning("   This may cause inflated accuracy due to prior label exposure.")
        logger.info(f"Loading fine-tuned Point-MAE from: {finetune_ckpt}")
        model, _ = load_pointmae_model(finetune_ckpt, device=device)

    else:
        raise ValueError(f"Unknown student_init mode: '{student_init}'")

    add_get_logits_loss(model)
    add_get_embeddings(model)

    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    return model, model_size


def _build_pointnext_student_cross_arch(student_cfg, training_cfg, device, pruning_cfg, student_init):
    """Build PointNeXt/PointMLP student model.

    Args:
        student_cfg: EasyConfig for student model
        training_cfg: Training config
        device: Target device
        pruning_cfg: Pruning config
        student_init: Initialization mode (random, pretrain, finetune)

    Returns:
        (model, model_size_M)
    """
    from openpoints.models import build_model_from_cfg

    # Setup model config with training settings
    if not hasattr(student_cfg.model, "criterion_args"):
        student_cfg.model.criterion_args = training_cfg.get(
            "criterion_args", {"NAME": "CrossEntropyLoss", "label_smoothing": 0.0}
        )
    if hasattr(student_cfg.model, "encoder_args"):
        if "in_channels" not in student_cfg.model.encoder_args:
            student_cfg.model.encoder_args.in_channels = training_cfg.get("in_channels", 3)
    if hasattr(student_cfg.model, "cls_args"):
        if "num_classes" not in student_cfg.model.cls_args:
            student_cfg.model.cls_args.num_classes = training_cfg.num_classes

    # Build model
    model = build_model_from_cfg(student_cfg.model).to(device)

    # Handle initialization modes
    if student_init == "random":
        # Default build_model_from_cfg already does random init
        logger.info("Built PointNeXt student with RANDOM initialization")

    elif student_init == "pretrain":
        # Load pre-trained encoder weights if available
        pretrain_ckpt = pruning_cfg.get("student_pretrain_checkpoint") if pruning_cfg else None
        if pretrain_ckpt:
            from openpoints.utils import load_checkpoint
            logger.info(f"Loading pre-trained PointNeXt encoder from: {pretrain_ckpt}")
            load_checkpoint(model, pretrain_ckpt, strict=False)
        else:
            logger.info("No student_pretrain_checkpoint provided, using random init")

    elif student_init == "finetune":
        # Load fine-tuned checkpoint (NOT recommended)
        finetune_ckpt = pruning_cfg.get("student_checkpoint") if pruning_cfg else None
        if finetune_ckpt is None and pruning_cfg:
            finetune_ckpt = pruning_cfg.get("scorer_checkpoint")
        if finetune_ckpt:
            from openpoints.utils import load_checkpoint
            logger.warning("⚠️ student_init='finetune' loads task-specific weights!")
            logger.info(f"Loading fine-tuned PointNeXt from: {finetune_ckpt}")
            load_checkpoint(model, finetune_ckpt, strict=False)

    # Add get_embeddings method for RKD
    add_get_embeddings_pointnext(model)

    # Count parameters
    model_size = sum(p.numel() for p in model.parameters()) / 1e6

    return model, model_size


def _is_pointmae_model(model) -> bool:
    """Detect if model is Point-MAE architecture.

    Point-MAE has group_divider, cls_token, pos_embed attributes.
    PointNeXt has encoder, prediction attributes.
    """
    actual_model = model.module if hasattr(model, "module") else model
    return hasattr(actual_model, "group_divider") and hasattr(actual_model, "cls_token")


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
    is_pointmae = _is_pointmae_model(model)

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)

        target = data["y"]
        points = data["x"]
        points = points[:, :npoints]

        data.update(prepare_data_dict(points, cfg))

        # Point-MAE expects tensor [B, N, 3], PointNeXt expects dict
        if is_pointmae:
            logits = model(data["pos"])
        else:
            logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    macc, overall_acc, accs = cm.cal_acc(cm.tp, cm.count)
    return macc, overall_acc, accs


def load_teacher_model(cfg, device, scorer_model):
    """Load teacher model for KD, or reuse scorer if not specified.

    Returns:
        model or None if KD disabled
    """
    if not cfg.pruning.get("use_kd", False):
        return None

    teacher_ckpt = cfg.pruning.get("teacher_checkpoint")

    # If no separate teacher specified, reuse scorer
    if not teacher_ckpt:
        logger.info("No separate teacher specified, reusing scorer model for KD")
        return scorer_model

    teacher_config = cfg.pruning.get("teacher_config", cfg.pruning.scorer_config)

    logger.info("Loading separate KD teacher...")
    logger.info(f"  Checkpoint: {teacher_ckpt}")

    model, _ = load_model_by_architecture(
        teacher_ckpt, teacher_config, device, architecture="single", freeze=True
    )

    logger.info("✓ Loaded teacher model")
    return model


@hydra.main(
    config_path="cfgs_pruning", config_name="pruning_cross_arch", version_base=None
)
def main(cfg: DictConfig):
    """Main cross-architecture pruning workflow."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("Cross-Architecture Dataset Pruning Pipeline")
    logger.info("=" * 80)

    # 1. Load and merge configs (scorer/teacher config)
    logger.info(f"Loading scorer/teacher config: {cfg.pointnext_config}")
    pointnext_cfg = load_pointnext_config(cfg.pointnext_config)
    full_cfg = merge_pruning_config(pointnext_cfg, cfg)

    # 1b. Load student config (may be different architecture)
    student_config_path = cfg.get("student_config", cfg.pointnext_config)
    logger.info(f"Loading student config: {student_config_path}")
    student_cfg = load_pointnext_config(student_config_path)

    # Log architecture info
    scorer_model = cfg.get("scorer_model", cfg.get("model", "unknown"))
    student_model_name = cfg.get("student_model", scorer_model)
    logger.info(f"Scorer/Teacher architecture: {scorer_model}")
    logger.info(f"Student architecture: {student_model_name}")
    if scorer_model != student_model_name:
        logger.info(">>> Cross-architecture distillation enabled <<<")

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

    scorer = get_scorer(
        scorer_method,
        scorer_model,
        full_cfg.openpoint,
        device=str(device),
    )

    # Check for hybrid selection mode
    use_hybrid = cfg.pruning.get("hybrid", False)
    hybrid_ratio = cfg.pruning.get("hybrid_per_class_ratio", 0.5)

    # Fail fast if scorer doesn't support hybrid
    if use_hybrid and not scorer_cls.supports_hybrid:
        raise ValueError(
            f"Scorer '{scorer_method}' does not support hybrid mode. "
            f"It has its own budget allocation logic."
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
        )

    # 7. Select samples
    logger.info("=" * 80)
    logger.info("Selecting samples:")
    logger.info(f"  Method: {scorer_method}")
    logger.info(f"  Total samples: {cfg.pruning.total_samples}")

    if use_hybrid:
        logger.info(f"  Hybrid mode: {hybrid_ratio*100:.0f}% per-class, {(1-hybrid_ratio)*100:.0f}% global")
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
                scores, labels, indices,
                total_samples=cfg.pruning.total_samples,
                hybrid_per_class_ratio=hybrid_ratio,
                mode=selection_mode,
                num_classes=full_cfg.openpoint.num_classes,
            )
        else:
            # Selection-based scorers: INCREMENTAL HYBRID for RBF submodular
            logger.info("Using INCREMENTAL hybrid selection for RBF submodular...")
            import math

            phase1_per_class = math.floor(
                hybrid_ratio * cfg.pruning.total_samples / full_cfg.openpoint.num_classes
            )
            phase1_budget = phase1_per_class * full_cfg.openpoint.num_classes
            phase2_budget = cfg.pruning.total_samples - phase1_budget

            logger.info(f"  Phase 1: {phase1_per_class} per class × {full_cfg.openpoint.num_classes} = {phase1_budget}")
            logger.info(f"  Phase 2: {phase2_budget} global (on FULL dataset)")

            # Phase 1: Per-class selection (independent)
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
            logger.info(f"  Phase 1 selected: {len(phase1_indices)} samples")

            # Phase 2: Global selection on FULL dataset (independent, stateless)
            if phase2_budget > 0:
                logger.info("  Phase 2: Global selection on FULL dataset (stateless)...")
                scores2, labels2, indices2 = scorer.compute(
                    scoring_loader,  # FULL dataset, not subset!
                    total_samples=phase2_budget,
                    per_class=False,
                    num_classes=full_cfg.openpoint.num_classes,
                    grad_scope=cfg.pruning.get("grad_norm_scope", "head"),
                    sigma=cfg.pruning.get("submodular_sigma"),
                    space=cfg.pruning.get("submodular_space", "embedding"),
                    rbf_algorithm=cfg.pruning.get("rbf_algorithm", "apricot"),
                )
                phase2_indices = scorer.select(
                    scores2, labels2, indices2,
                    total_samples=phase2_budget,
                    per_class=False,
                    mode=selection_mode,
                    num_classes=full_cfg.openpoint.num_classes,
                )
                logger.info(f"  Phase 2 selected: {len(phase2_indices)} samples")
            else:
                phase2_indices = []

            # Merge phase 1 and phase 2 (may have overlap)
            phase1_set = set(phase1_indices)
            phase2_set = set(phase2_indices)
            merged_set = phase1_set | phase2_set
            overlap = phase1_set & phase2_set

            logger.info(f"  Merge: {len(phase1_set)} + {len(phase2_set)} = {len(merged_set)} unique")
            logger.info(f"  Overlap: {len(overlap)} samples ({100*len(overlap)/max(len(phase2_set),1):.1f}% of phase2)")

            # Calculate remaining budget after merge
            remaining_budget = cfg.pruning.total_samples - len(merged_set)
            logger.info(f"  Remaining budget after merge: {remaining_budget}")

            # Incremental fill using initial_subset (ONLY HERE do we use warm-start)
            if remaining_budget > 0:
                logger.info(f"  Incremental fill: selecting {remaining_budget} more samples...")
                merged_list = list(merged_set)

                # Use initial_subset for warm-start selection
                scores_fill, labels_fill, indices_fill = scorer.compute(
                    scoring_loader,
                    total_samples=remaining_budget,
                    per_class=False,
                    num_classes=full_cfg.openpoint.num_classes,
                    grad_scope=cfg.pruning.get("grad_norm_scope", "head"),
                    sigma=cfg.pruning.get("submodular_sigma"),
                    space=cfg.pruning.get("submodular_space", "embedding"),
                    rbf_algorithm=cfg.pruning.get("rbf_algorithm", "apricot"),
                    initial_subset=merged_list,  # Warm-start with merged selection!
                )
                fill_indices = scorer.select(
                    scores_fill, labels_fill, indices_fill,
                    total_samples=remaining_budget,
                    per_class=False,
                    mode=selection_mode,
                    num_classes=full_cfg.openpoint.num_classes,
                )
                logger.info(f"  Incremental fill selected: {len(fill_indices)} samples")

                # Final selection = merged + fill
                selected_indices = merged_list + fill_indices
            else:
                # No fill needed (or over budget due to rounding)
                selected_indices = list(merged_set)
                if remaining_budget < 0:
                    logger.warning(
                        f"  Over budget by {-remaining_budget} samples "
                        f"(due to perfect overlap not happening)"
                    )
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
    pruned_loader = DataLoader(
        pruned_dataset,
        batch_size=full_cfg.openpoint.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,  # Prevents BatchNorm failure when last batch has size 1
    )

    # 9. Load validation dataset (unified builder)
    val_batch_size = cfg.pruning.get(
        "val_batch_size", full_cfg.openpoint.batch_size * 2
    )
    val_dataset = build_train_dataset(full_cfg, "val")
    val_loader = build_dataloader(full_cfg, val_dataset, "val", batch_size=val_batch_size)
    logger.info(f"Validation dataset size: {len(val_dataset)}, batch size: {val_batch_size}")

    # 10. Build student model (using separate student config for cross-arch)
    logger.info("=" * 80)
    logger.info("Building student model...")
    student_model, model_size = build_student_model_cross_arch(
        student_cfg, full_cfg.openpoint, device, cfg.pruning
    )
    logger.info(f"Student model: {model_size:.2f}M parameters")

    optimizer = build_optimizer_from_cfg(
        student_model, lr=full_cfg.openpoint.lr, **full_cfg.openpoint.optimizer
    )
    scheduler = build_scheduler_from_cfg(full_cfg.openpoint, optimizer)

    exp_name = f"pruning_{scorer_method}_{cfg.pruning.mode}_{cfg.pruning.total_samples}"
    run_name = setup_experiment(full_cfg.openpoint, exp_name)

    # 11. Load teacher model for KD
    logger.info("=" * 80)
    teacher_model = load_teacher_model(cfg, device, scorer_model)

    # 12. Create trainer (RKD + Logit KD, no Proto-RKD for cross-arch)
    logger.info("=" * 80)
    trainer = get_trainer(
        student_model,
        full_cfg.openpoint,
        device,
        cfg.pruning,
        teacher_model=teacher_model,
        prototypes=None,  # Proto-RKD not used in cross-arch
    )

    # Log training config
    use_kd = cfg.pruning.get("use_kd", False)

    if use_kd:
        logger.info("Training Mode: Cross-Architecture Knowledge Distillation")
        logger.info(f"  Teacher: {scorer_model}")
        logger.info(f"  Student: {student_model_name}")
        logger.info(f"  KD Alpha: {cfg.pruning.get('kd_alpha', 0.5)}")
        logger.info(f"  KD Temperature: {cfg.pruning.get('kd_temperature', 3.0)}")
        if cfg.pruning.get("use_rkd", False):
            logger.info(f"  RKD: Enabled (distance={cfg.pruning.get('rkd_distance_weight', 1.0)}, "
                       f"angle={cfg.pruning.get('rkd_angle_weight', 2.0)})")
        if cfg.pruning.get("use_logit_kd", False):
            logger.info(f"  Logit KD: Enabled (scale={cfg.pruning.get('rkd_loss_scale', 1.0)})")
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
