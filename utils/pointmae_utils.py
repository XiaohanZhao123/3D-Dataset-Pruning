"""Point-MAE integration utilities for class-balanced retraining.

This module provides:
1. Model detection and loading helpers
2. Encoder/classifier component identification
3. Interface adapter for StandardTrainer compatibility
4. Dataset wrapper for openpoints format conversion

Usage:
    from utils.pointmae_utils import (
        is_pointmae_model,
        load_pointmae_model,
        add_get_logits_loss,
        get_pointmae_encoder_modules,
        get_pointmae_classifier,
    )
"""

import logging
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Path to Point-MAE source
POINTMAE_PATH = Path(__file__).parent.parent / "third_party" / "pointmae"


def _ensure_pointmae_path():
    """Add Point-MAE to sys.path if not already present."""
    path_str = str(POINTMAE_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def is_pointmae_model(model) -> bool:
    """Check if model is Point-MAE PointTransformer architecture.

    Identifies by presence of:
    - group_divider (FPS + KNN grouping)
    - cls_head_finetune (classification head)
    - blocks (TransformerEncoder)

    Args:
        model: PyTorch model to check

    Returns:
        True if model is Point-MAE PointTransformer
    """
    return (
        hasattr(model, "group_divider")
        and hasattr(model, "cls_head_finetune")
        and hasattr(model, "blocks")
    )


def get_pointmae_encoder_modules(model) -> list:
    """Get list of encoder module names for freezing.

    Point-MAE encoder consists of:
    - group_divider: FPS + KNN grouping
    - encoder: Patch embedding (Conv1d network)
    - pos_embed: Position embedding MLP
    - cls_token: Classification token (Parameter)
    - cls_pos: Classification position embedding (Parameter)
    - blocks: TransformerEncoder
    - norm: Final LayerNorm

    Args:
        model: Point-MAE PointTransformer model

    Returns:
        List of attribute names that should be frozen
    """
    encoder_modules = [
        "group_divider",  # FPS + KNN grouping
        "encoder",  # Patch embedding
        "pos_embed",  # Position embedding MLP
        "cls_token",  # Classification token (nn.Parameter)
        "cls_pos",  # Classification position embedding (nn.Parameter)
        "blocks",  # TransformerEncoder
        "norm",  # Final LayerNorm
    ]
    return [m for m in encoder_modules if hasattr(model, m)]


def get_pointmae_classifier(model) -> nn.Module:
    """Get the classifier head from Point-MAE model.

    Args:
        model: Point-MAE PointTransformer model

    Returns:
        cls_head_finetune module (nn.Sequential)

    Raises:
        ValueError: If model does not have cls_head_finetune
    """
    if hasattr(model, "cls_head_finetune"):
        return model.cls_head_finetune
    raise ValueError("Model does not have cls_head_finetune attribute")


def add_get_logits_loss(model, criterion=None):
    """Add get_logits_loss() method to Point-MAE model for StandardTrainer compatibility.

    The StandardTrainer expects models to have a get_logits_loss(data, target) method
    that returns (logits, loss). Point-MAE has get_loss_acc(ret, gt) instead.

    This function monkey-patches a compatible get_logits_loss() method onto the model.

    Args:
        model: Point-MAE PointTransformer model
        criterion: Optional loss function. If None, uses model's built-in loss_ce.

    Returns:
        Model with get_logits_loss() method added
    """

    def get_logits_loss(self, data, target):
        """Forward pass + loss computation for StandardTrainer compatibility.

        Args:
            data: Dict with 'pos' key containing point cloud [B, N, 3]
            target: Ground truth labels [B]

        Returns:
            (logits, loss) tuple
        """
        # Extract points from dict (prepare_batch returns dict with 'pos')
        if isinstance(data, dict):
            points = data["pos"]
        else:
            # Already a tensor
            points = data

        # Point-MAE forward expects [B, N, 3] tensor
        logits = self.forward(points)

        # Compute loss using model's criterion or external one
        if criterion is not None:
            loss = criterion(logits, target)
        else:
            # Use model's built-in cross-entropy loss
            loss = self.loss_ce(logits, target.long())

        return logits, loss

    # Monkey-patch the method
    model.get_logits_loss = types.MethodType(get_logits_loss, model)

    # Also add criterion attribute for compatibility
    if criterion is not None:
        model.criterion = criterion
    elif not hasattr(model, "criterion"):
        model.criterion = model.loss_ce

    logger.info("Added get_logits_loss() method to Point-MAE model")
    return model


def _load_pointmae_via_subprocess(ckpt_path: str, config: dict = None, cache_path: str = None):
    """Load Point-MAE model using subprocess to avoid import conflicts.

    Args:
        ckpt_path: Path to Point-MAE checkpoint
        config: Optional config dict with model parameters (cls_dim, num_group, etc.)
        cache_path: Optional path to cache the loaded state dict

    Returns:
        Dict containing 'state_dict' and 'config'
    """
    import subprocess
    import tempfile

    if cache_path is None:
        # Create temp file
        fd, cache_path = tempfile.mkstemp(suffix=".pth")
        import os
        os.close(fd)
        cleanup = True
    else:
        cleanup = False

    try:
        # Build command with config arguments
        loader_script = Path(__file__).parent / "_pointmae_loader.py"
        cmd = [sys.executable, str(loader_script), str(ckpt_path), cache_path]

        # Add config parameters if provided
        if config:
            if "cls_dim" in config:
                cmd.extend(["--cls-dim", str(config["cls_dim"])])
            if "num_group" in config:
                cmd.extend(["--num-group", str(config["num_group"])])
            if "trans_dim" in config:
                cmd.extend(["--trans-dim", str(config["trans_dim"])])
            if "depth" in config:
                cmd.extend(["--depth", str(config["depth"])])
            if "num_heads" in config:
                cmd.extend(["--num-heads", str(config["num_heads"])])
            if "group_size" in config:
                cmd.extend(["--group-size", str(config["group_size"])])

        # Run the loader script in a subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),  # PointNeXt root
        )

        if result.returncode != 0:
            logger.error(f"Subprocess stderr: {result.stderr}")
            raise RuntimeError(f"Failed to load Point-MAE model: {result.stderr}")

        logger.info(result.stdout.strip())

        # Load the saved model
        # We need Point-MAE in path to unpickle the model class
        # Also need to temporarily hide our utils module to avoid conflicts
        original_path = sys.path.copy()
        saved_modules = {}

        # Save and remove conflicting modules
        for mod_name in list(sys.modules.keys()):
            if mod_name == 'utils' or mod_name.startswith('utils.'):
                saved_modules[mod_name] = sys.modules.pop(mod_name)
            elif mod_name == 'models' or mod_name.startswith('models.'):
                saved_modules[mod_name] = sys.modules.pop(mod_name)

        sys.path.insert(0, str(POINTMAE_PATH))
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
        finally:
            sys.path = original_path
            # Restore our modules
            sys.modules.update(saved_modules)

        return data

    finally:
        if cleanup:
            import os
            try:
                os.unlink(cache_path)
            except Exception:
                pass


def load_pointmae_model(ckpt_path: str, config: dict = None, device="cuda"):
    """Load Point-MAE PointTransformer model from checkpoint.

    Uses subprocess isolation to build the model, then loads via pickle.
    Point-MAE's models/__init__.py uses lazy imports to avoid conflicts.

    Args:
        ckpt_path: Path to checkpoint file (e.g., modelnet_8k.pth)
        config: Optional config dict. If None, uses default 8k config.
        device: Device to load model to

    Returns:
        (model, config): Loaded PointTransformer model and config used
    """
    from easydict import EasyDict

    # Load via subprocess - this returns full model + config
    logger.info(f"Loading Point-MAE model from {ckpt_path} via subprocess...")
    data = _load_pointmae_via_subprocess(ckpt_path, config=config)

    # Get the full model object
    model = data["model"]
    model_config = EasyDict(data["config"]) if config is None else EasyDict(config)

    # Move to device
    model.to(device)

    logger.info(f"Loaded Point-MAE model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    return model, model_config


def freeze_pointmae_encoder(model):
    """Freeze all encoder components of Point-MAE model.

    Freezes:
    - group_divider (FPS + KNN)
    - encoder (patch embedding)
    - pos_embed (position embedding MLP)
    - cls_token, cls_pos (classification tokens)
    - blocks (transformer encoder)
    - norm (final layer norm)

    Args:
        model: Point-MAE PointTransformer model

    Returns:
        (model, frozen_count, frozen_components): Model with frozen encoder,
            total frozen parameter count, and list of frozen component names
    """
    frozen_components = []
    frozen_param_count = 0

    encoder_modules = get_pointmae_encoder_modules(model)

    for module_name in encoder_modules:
        module = getattr(model, module_name)

        if isinstance(module, nn.Parameter):
            # Handle nn.Parameter directly (cls_token, cls_pos)
            module.requires_grad = False
            frozen_param_count += module.numel()
        else:
            # Handle nn.Module
            for param in module.parameters():
                param.requires_grad = False
                frozen_param_count += param.numel()

        frozen_components.append(module_name)

    logger.info("Point-MAE encoder frozen:")
    for comp in frozen_components:
        logger.info(f"  - {comp}")
    logger.info(f"Total frozen params: {frozen_param_count:,}")

    return model, frozen_param_count, frozen_components


def reinit_pointmae_classifier(model):
    """Re-initialize Point-MAE classifier head with Xavier uniform.

    Point-MAE's cls_head_finetune is a 3-layer MLP:
    - Linear(trans_dim * 2, 256) + BatchNorm + ReLU + Dropout
    - Linear(256, 256) + BatchNorm + ReLU + Dropout
    - Linear(256, cls_dim)

    Args:
        model: Point-MAE PointTransformer model

    Returns:
        (model, trainable_count): Model with reinitialized classifier,
            and trainable parameter count
    """
    classifier = get_pointmae_classifier(model)

    # Re-initialize all Linear and BatchNorm layers
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # Ensure classifier parameters are trainable
    trainable_count = 0
    for param in classifier.parameters():
        param.requires_grad = True
        trainable_count += param.numel()

    logger.info(f"Point-MAE classifier re-initialized: {trainable_count:,} trainable params")

    return model, trainable_count


class PointMAEDatasetWrapper:
    """Wrapper to make openpoints datasets compatible with Point-MAE.

    Converts openpoints format to Point-MAE format while maintaining
    compatibility with UniformClassSampler (via targets property).

    Openpoints format: {'pos': [N, 3], 'y': label, 'x': [N, C]}
    Point-MAE format: (taxonomy_id, model_id, (points, label))

    Note: For class-balanced retraining, we keep openpoints format
    since prepare_batch() handles the conversion. This wrapper is
    provided for cases where direct Point-MAE format is needed.

    Args:
        dataset: Openpoints dataset instance
        num_points: Number of points to use (default 8192 for Point-MAE)
    """

    def __init__(self, dataset, num_points: int = 8192):
        self.dataset = dataset
        self.num_points = num_points
        self._extract_labels()

    def _extract_labels(self):
        """Extract labels array for sampler compatibility."""
        if hasattr(self.dataset, "label"):
            self.labels = self.dataset.label
        elif hasattr(self.dataset, "labels"):
            self.labels = self.dataset.labels
        elif hasattr(self.dataset, "targets"):
            self.labels = self.dataset.targets
        else:
            # Fallback: iterate through dataset
            self.labels = []
            for i in range(len(self.dataset)):
                data = self.dataset[i]
                if isinstance(data, dict):
                    self.labels.append(data["y"])
                else:
                    self.labels.append(data[1])
            self.labels = torch.tensor(self.labels)

    @property
    def targets(self):
        """Compatibility with UniformClassSampler."""
        return self.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return data in openpoints dict format (for prepare_batch compatibility)."""
        data = self.dataset[idx]

        if isinstance(data, dict):
            points = data["pos"]
            label = data["y"]
        else:
            # Handle tuple format
            points, label = data[0], data[1]

        # Ensure points is tensor
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).float()

        # Take only xyz if more channels
        if points.shape[-1] > 3:
            points = points[:, :3]

        # Ensure correct number of points
        if points.shape[0] > self.num_points:
            # Random subsample
            idx_pts = torch.randperm(points.shape[0])[: self.num_points]
            points = points[idx_pts]
        elif points.shape[0] < self.num_points:
            # Pad by repeating
            repeat_times = (self.num_points // points.shape[0]) + 1
            points = points.repeat(repeat_times, 1)[: self.num_points]

        # Return in openpoints format
        return {"pos": points, "y": label, "x": points}
