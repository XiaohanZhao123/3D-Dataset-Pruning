"""Unified builder functions for model and dataset loading.

This module provides a unified interface for loading models and datasets,
hiding the complexity of different architectures (PointNeXt, Point-MAE, etc.)
behind simple builder functions.

Usage:
    from utils.builders import build_scorer_model, build_train_dataset

    # Model loading - auto-detects Point-MAE vs PointNeXt
    model, model_type = build_scorer_model(cfg, device)

    # Dataset loading - uses PointMAEModelNet40 for Point-MAE, else OpenPoints
    dataset = build_train_dataset(cfg, "train")
"""

import logging
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def detect_model_type(cfg_or_path: Union[str, object]) -> str:
    """Auto-detect model type from checkpoint path or config.

    Detection rules (in order):
    1. Config has pruning.model_type set explicitly -> use that
    2. Path contains "pointmae/" or checkpoint in pointmae folder -> "pointmae"
    3. Filename contains "mae" or "pretrain" -> "pointmae"
    4. Filename contains "pointmlp" -> "pointmlp"
    5. Otherwise -> "pointnext"

    Args:
        cfg_or_path: Either a config object with pruning.scorer_checkpoint,
                     or a direct path string

    Returns:
        Model type string: "pointmae", "pointmlp", or "pointnext"
    """
    if isinstance(cfg_or_path, str):
        ckpt_path = cfg_or_path
        explicit_type = None
    else:
        # Check for explicit model_type in config
        explicit_type = cfg_or_path.pruning.get("model_type")
        if explicit_type:
            return explicit_type
        ckpt_path = cfg_or_path.pruning.scorer_checkpoint

    # Normalize path for detection
    path_lower = ckpt_path.lower()
    filename = os.path.basename(ckpt_path).lower()

    # Check if in pointmae folder
    if "/pointmae/" in path_lower or "\\pointmae\\" in path_lower:
        return "pointmae"

    # Check filename patterns
    if "mae" in filename or "pretrain" in filename:
        return "pointmae"
    elif "pointmlp" in filename:
        return "pointmlp"
    else:
        return "pointnext"


def build_scorer_model(
    cfg,
    device: torch.device,
    freeze: bool = True,
) -> Tuple[nn.Module, str]:
    """Build scorer model with unified interface.

    Auto-detects model type and loads appropriately:
    - Point-MAE: Uses load_pointmae_model + add_get_embeddings
    - PointNeXt/PointMLP: Delegates to existing load_scorer_model

    Args:
        cfg: Full config with pruning.scorer_checkpoint, pruning.scorer_config, etc.
        device: Target device
        freeze: Whether to freeze model parameters (default True)

    Returns:
        (model, model_type): Loaded model and detected type
    """
    model_type = detect_model_type(cfg)
    logger.info(f"Detected model type: {model_type}")

    if model_type == "pointmae":
        return _build_pointmae_scorer(cfg, device, freeze)
    else:
        return _build_pointnext_scorer(cfg, device, freeze)


def _build_pointmae_scorer(
    cfg,
    device: torch.device,
    freeze: bool = True,
) -> Tuple[nn.Module, str]:
    """Build Point-MAE scorer model.

    Args:
        cfg: Config with pruning.scorer_checkpoint
        device: Target device
        freeze: Whether to freeze parameters

    Returns:
        (model, "pointmae")
    """
    from utils.pointmae_utils import (
        add_get_logits_loss,
        load_pointmae_model,
    )

    ckpt_path = cfg.pruning.scorer_checkpoint
    logger.info(f"Loading Point-MAE scorer from: {ckpt_path}")

    # Load model
    model, model_config = load_pointmae_model(ckpt_path, device=device)

    # Add interface methods
    add_get_logits_loss(model)
    add_get_embeddings(model)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info("Point-MAE scorer loaded with frozen parameters")
    else:
        logger.info("Point-MAE scorer loaded with trainable parameters")

    return model, "pointmae"


def _build_pointnext_scorer(
    cfg,
    device: torch.device,
    freeze: bool = True,
) -> Tuple[nn.Module, str]:
    """Build PointNeXt/PointMLP scorer model by delegating to existing function.

    Args:
        cfg: Config with pruning settings
        device: Target device
        freeze: Whether to freeze parameters

    Returns:
        (model, model_type)
    """
    from pruning.balanced_scorers import SCORER_REGISTRY
    from utils.model_loading import load_standard_model

    scorer_method = str(cfg.pruning.get("scorer", "loss")).lower()
    scorer_cls = SCORER_REGISTRY.get(scorer_method)

    requires_grad = scorer_cls.requires_grad if scorer_cls else False
    effective_freeze = not requires_grad if not freeze else freeze

    ckpt = cfg.pruning.scorer_checkpoint
    config = cfg.pruning.scorer_config

    logger.info("Loading PointNeXt/PointMLP scorer...")
    logger.info(f"  Checkpoint: {ckpt}")

    model, _ = load_standard_model(ckpt, config, device, freeze=effective_freeze)

    # Add get_embeddings method for PointNeXt
    add_get_embeddings_pointnext(model)

    model_type = detect_model_type(ckpt)
    logger.info(f"Loaded {model_type} scorer")

    return model, model_type


def add_get_embeddings(model: nn.Module):
    """Add get_embeddings() method to Point-MAE model.

    Extracts CLS token embedding after transformer blocks.

    Args:
        model: Point-MAE PointTransformer model

    Returns:
        Model with get_embeddings() method added
    """
    import types

    def get_embeddings(self, data):
        """Extract CLS token embedding from Point-MAE.

        Args:
            data: Dict with 'pos' key [B, N, 3] or tensor [B, N, 3]

        Returns:
            CLS token embedding [B, trans_dim * 2] (768 for default config)
        """
        # Extract points
        if isinstance(data, dict):
            pts = data["pos"]
        else:
            pts = data

        # Ensure on same device as model
        device = next(self.parameters()).device
        if pts.device != device:
            pts = pts.to(device)

        # Point-MAE forward through encoder (same as forward method)
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # [B, G, C]

        # Get batch size
        B = group_input_tokens.size(0)

        # Prepare CLS token and position
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        cls_pos = self.cls_pos.expand(B, -1, -1)  # [B, 1, C]

        # pos_embed is a module that maps center coordinates to embeddings
        pos = self.pos_embed(center)  # [B, G, C]

        # Concatenate CLS with patch tokens
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # [B, 1+G, C]
        pos = torch.cat((cls_pos, pos), dim=1)  # [B, 1+G, C]

        # Through transformer blocks
        x = self.blocks(x, pos)
        x = self.norm(x)

        # Extract CLS token and concatenate with max-pooled features
        # (Point-MAE uses concat_cls_feat = True by default)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)  # [B, C*2]

        return concat_f

    model.get_embeddings = types.MethodType(get_embeddings, model)
    logger.info("Added get_embeddings() method to Point-MAE model")
    return model


def add_get_embeddings_pointnext(model: nn.Module):
    """Add get_embeddings() method to PointNeXt/PointMLP model.

    Extracts encoder output features.

    Args:
        model: PointNeXt BaseCls model

    Returns:
        Model with get_embeddings() method added
    """
    import types

    def get_embeddings(self, data):
        """Extract encoder features from PointNeXt.

        Args:
            data: Dict with model inputs

        Returns:
            Global feature embedding [B, C]
        """
        actual_model = self.module if hasattr(self, "module") else self

        if hasattr(actual_model, "encoder"):
            # BaseCls architecture
            return actual_model.encoder.forward_cls_feat(data)
        elif hasattr(actual_model, "embedding"):
            # PointMLP architecture
            xyz = data["pos"]
            x = actual_model.embedding(xyz)
            for block in actual_model.local_grouper_list:
                x, xyz = block(x, xyz)
            x = x.mean(-1)  # Global pooling
            return x
        else:
            raise ValueError("Cannot extract embeddings from this model")

    model.get_embeddings = types.MethodType(get_embeddings, model)
    return model


def build_train_dataset(cfg, split: str = "train") -> Dataset:
    """Build training/validation dataset with unified interface.

    Auto-detects whether to use Point-MAE loader or OpenPoints loader.

    Args:
        cfg: Config object with:
            - pruning.pointmae_data_dir (optional): Path to Point-MAE format data
            - openpoint.dataset: OpenPoints dataset config
        split: "train" or "val"/"test"

    Returns:
        Dataset instance
    """
    # Check if Point-MAE data directory is specified
    pointmae_data_dir = cfg.pruning.get("pointmae_data_dir")

    if pointmae_data_dir:
        return _build_pointmae_dataset(cfg, split, pointmae_data_dir)
    else:
        return _build_openpoints_dataset(cfg, split)


def _build_pointmae_dataset(cfg, split: str, data_dir: str) -> Dataset:
    """Build Point-MAE format dataset.

    Auto-detects dataset type based on config or data_dir path:
    - ModelNet40: Uses PointMAEModelNet40 (8192 points, FPS cache)
    - ScanObjectNN: Uses PointMAEScanObjectNN (2048 points, h5 files)

    Args:
        cfg: Config object
        split: "train" or "val"/"test"
        data_dir: Path to Point-MAE format data directory

    Returns:
        Dataset instance (PointMAEModelNet40 or PointMAEScanObjectNN)
    """
    from utils.pointmae_dataloader import PointMAEModelNet40, PointMAEScanObjectNN

    num_points = cfg.openpoint.get("num_points", 8192)

    # Map "val" to "test"
    actual_split = "test" if split in ("val", "test") else "train"

    # Detect dataset type from config or path
    dataset_name = cfg.get("dataset", "").lower()
    data_dir_lower = data_dir.lower()

    if "scanobjectnn" in dataset_name or "scanobjectnn" in data_dir_lower:
        # ScanObjectNN dataset
        variant = cfg.pruning.get("scanobjectnn_variant", "hardest")
        num_points = cfg.openpoint.get("num_points", 2048)  # ScanObjectNN default

        logger.info(f"Loading ScanObjectNN ({variant}) from {data_dir}")
        logger.info(f"  split={actual_split}, num_points={num_points}")

        dataset = PointMAEScanObjectNN(
            data_dir=data_dir,
            split=actual_split,
            variant=variant,
            num_points=num_points,
        )
    else:
        # ModelNet40 dataset (default)
        logger.info(f"Loading Point-MAE ModelNet40 dataset from {data_dir}")
        logger.info(f"  split={actual_split}, num_points={num_points}")

        dataset = PointMAEModelNet40(
            data_dir=data_dir,
            split=actual_split,
            num_points=num_points,
            use_normals=False,
            cache_only=True,  # Use pre-built FPS cache
        )

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def _build_openpoints_dataset(cfg, split: str) -> Dataset:
    """Build OpenPoints format dataset.

    Args:
        cfg: Config with openpoint.dataset settings
        split: "train" or "val"

    Returns:
        OpenPoints dataset
    """
    from openpoints.dataset import build_dataloader_from_cfg

    logger.info(f"Loading OpenPoints dataset for split={split}")

    loader = build_dataloader_from_cfg(
        cfg.openpoint.batch_size,
        cfg.openpoint.dataset,
        cfg.openpoint.dataloader,
        datatransforms_cfg=cfg.openpoint.datatransforms,
        split=split,
        distributed=False,
    )

    logger.info(f"Loaded {len(loader.dataset)} samples")
    return loader.dataset


def build_dataloader(
    cfg,
    dataset: Dataset,
    split: str = "train",
    sampler=None,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Build DataLoader with unified interface.

    Args:
        cfg: Config with openpoint.dataloader settings
        dataset: Dataset instance
        split: "train" or "val" (affects shuffle default)
        sampler: Optional sampler (overrides shuffle)
        batch_size: Override batch size (default from config)

    Returns:
        DataLoader instance
    """
    if batch_size is None:
        if split == "train":
            batch_size = cfg.openpoint.batch_size
        else:
            batch_size = cfg.openpoint.get("val_batch_size", cfg.openpoint.batch_size)

    num_workers = cfg.openpoint.dataloader.num_workers
    shuffle = (split == "train") and (sampler is None)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=(split == "train"),
    )

    return loader


def build_student_model(cfg, device: torch.device) -> Tuple[nn.Module, float]:
    """Build student model with unified interface.

    Auto-detects model type and builds appropriately.

    Args:
        cfg: Config with model settings
        device: Target device

    Returns:
        (model, model_size_M): Model and size in millions of parameters
    """
    model_type = detect_model_type(cfg)

    if model_type == "pointmae":
        return _build_pointmae_student(cfg, device)
    else:
        # Delegate to existing function
        from utils.train import build_student_model as _build_student

        return _build_student(cfg.openpoint, device)


def _build_pointmae_student(cfg, device: torch.device) -> Tuple[nn.Module, float]:
    """Build Point-MAE student model.

    Supports three initialization modes via `pruning.student_init`:
      - "random": Fully random initialization (true from-scratch training)
      - "pretrain": Load pre-trained encoder, random classifier (default, recommended)
      - "finetune": Load fine-tuned checkpoint (NOT recommended, causes data leakage)

    Args:
        cfg: Config with pointmae settings
        device: Target device

    Returns:
        (model, model_size_M)
    """
    from utils.pointmae_utils import add_get_logits_loss, load_pointmae_model

    student_init = cfg.pruning.get("student_init", "pretrain")
    logger.info(f"Student initialization mode: {student_init}")

    if student_init == "random":
        # Fully random initialization - build model without loading any weights
        model = _build_pointmae_random(cfg, device)
        logger.info("Built Point-MAE student with RANDOM initialization")

    elif student_init == "pretrain":
        # Load pre-trained encoder (self-supervised), random classifier
        # This is the correct default for dataset pruning experiments
        pretrain_ckpt = cfg.pruning.get(
            "student_pretrain_checkpoint", cfg.pruning.get("pretrain_checkpoint")
        )
        if pretrain_ckpt is None:
            # Try to find model_pretrain.pth in same directory as scorer checkpoint
            from pathlib import Path

            scorer_ckpt = cfg.pruning.scorer_checkpoint
            if isinstance(scorer_ckpt, (list, tuple)):
                scorer_ckpt = scorer_ckpt[0]
            pretrain_dir = Path(scorer_ckpt).parent
            pretrain_ckpt = pretrain_dir / "model_pretrain.pth"
            if not pretrain_ckpt.exists():
                raise ValueError(
                    f"student_init='pretrain' but no pretrain checkpoint found. "
                    f"Set pruning.student_pretrain_checkpoint or pruning.pretrain_checkpoint. "
                    f"Looked for: {pretrain_ckpt}"
                )
            pretrain_ckpt = str(pretrain_ckpt)

        logger.info(f"Loading pre-trained encoder from: {pretrain_ckpt}")
        logger.info("  -> Encoder: pre-trained weights (self-supervised, no labels)")
        logger.info("  -> Classifier: random initialization")
        model, _ = load_pointmae_model(pretrain_ckpt, device=device)

    elif student_init == "finetune":
        # Load fine-tuned checkpoint (legacy behavior, NOT recommended)
        ckpt_path = cfg.pruning.get("student_checkpoint", cfg.pruning.scorer_checkpoint)
        if isinstance(ckpt_path, (list, tuple)):
            ckpt_path = ckpt_path[-1]  # Use last checkpoint (usually fine-tuned)
        logger.warning("⚠️ student_init='finetune' loads task-specific weights!")
        logger.warning(
            "   This may cause inflated accuracy due to prior label exposure."
        )
        logger.warning("   Consider using 'pretrain' or 'random' for fair evaluation.")
        logger.info(f"Loading fine-tuned checkpoint from: {ckpt_path}")
        model, _ = load_pointmae_model(ckpt_path, device=device)

    else:
        raise ValueError(
            f"Unknown student_init mode: '{student_init}'. "
            f"Valid options: 'random', 'pretrain', 'finetune'"
        )

    add_get_logits_loss(model)
    add_get_embeddings(model)  # Required for RKD training

    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Point-MAE student: {model_size:.2f}M parameters")

    return model, model_size


def _build_pointmae_random(cfg, device: torch.device) -> nn.Module:
    """Build Point-MAE model with random initialization (no checkpoint loading).

    Args:
        cfg: Config with pointmae settings
        device: Target device

    Returns:
        Randomly initialized PointTransformer model
    """
    import sys
    from pathlib import Path

    # We need to build PointTransformer directly without loading checkpoint
    # This requires importing from Point-MAE codebase
    POINTMAE_PATH = Path(__file__).parent.parent / "third_party" / "pointmae"

    # Save current modules to avoid conflicts
    saved_modules = {}
    for mod_name in list(sys.modules.keys()):
        if mod_name == "utils" or mod_name.startswith("utils."):
            saved_modules[mod_name] = sys.modules.pop(mod_name)
        elif mod_name == "models" or mod_name.startswith("models."):
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    original_path = sys.path.copy()
    sys.path.insert(0, str(POINTMAE_PATH))

    try:
        from easydict import EasyDict
        from models.Point_MAE import PointTransformer

        # Default Point-MAE config for ModelNet40
        config = EasyDict(
            {
                "trans_dim": 384,
                "depth": 12,
                "drop_path_rate": 0.1,
                "cls_dim": 40,
                "num_heads": 6,
                "group_size": 32,
                "num_group": 512,
                "encoder_dims": 384,
            }
        )

        # Build model with random initialization
        model = PointTransformer(config)
        model.to(device)

        logger.info("Built PointTransformer with random initialization")
        logger.info(
            f"  Config: trans_dim={config.trans_dim}, depth={config.depth}, "
            f"num_heads={config.num_heads}"
        )

        return model

    finally:
        sys.path = original_path
        sys.modules.update(saved_modules)
