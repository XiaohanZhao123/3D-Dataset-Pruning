"""Model loading utilities for PointNeXt pruning framework.

Provides functions for loading models from checkpoints.

Usage:
    from utils.model_loading import load_model, load_standard_model

    # Load from checkpoint (uses config from checkpoint if available)
    model = load_model(ckpt_path, device, config_path)

    # Load with explicit config
    model, cfg = load_standard_model(ckpt_path, config_path, device)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_model(
    ckpt_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
    freeze: bool = True,
) -> nn.Module:
    """Load model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Target device
        config_path: Model config path (uses checkpoint config if not provided)
        freeze: Whether to freeze parameters (default True)

    Returns:
        Loaded model

    Raises:
        ValueError: If config not found in checkpoint and config_path not provided
    """
    logger.info(f"Loading model from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Get config
    model_cfg = _get_model_config(checkpoint, config_path)

    # Build model
    from openpoints.models import build_model_from_cfg

    model = build_model_from_cfg(model_cfg)

    # Load weights
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info("✓ Loaded model with frozen parameters")
    else:
        logger.info("✓ Loaded model with trainable parameters")

    return model


def load_standard_model(
    ckpt_path: str,
    config_path: str,
    device: torch.device,
    freeze: bool = True,
) -> Tuple[nn.Module, dict]:
    """Load model with explicit config file.

    Args:
        ckpt_path: Path to checkpoint
        config_path: Path to config file
        device: Target device
        freeze: Whether to freeze parameters (default True)

    Returns:
        (model, config)
    """
    from openpoints.models import build_model_from_cfg
    from utils.config_loader import load_pointnext_config

    logger.info(f"Loading standard model from: {ckpt_path}")
    cfg = load_pointnext_config(config_path)
    model = build_model_from_cfg(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info("✓ Loaded model with frozen parameters")
    else:
        logger.info("✓ Loaded model with trainable parameters")

    return model, cfg


def _get_model_config(checkpoint: dict, config_path: Optional[str]):
    """Extract model config from checkpoint or file."""
    if "config" in checkpoint:
        logger.info("Using config from checkpoint")
        return checkpoint["config"]["model"]

    if config_path:
        from utils.config_loader import load_pointnext_config

        logger.info(f"Loading config from: {config_path}")
        cfg = load_pointnext_config(config_path)
        return cfg.model

    raise ValueError(
        "Checkpoint missing 'config' and no config_path provided. "
        "Specify the config path explicitly."
    )


def get_encoder_output_dim(model: nn.Module) -> int:
    """Get the output dimension of the model's encoder."""
    actual_model = model.module if hasattr(model, "module") else model
    head = actual_model.prediction.head

    if isinstance(head, nn.Sequential):
        for m in head.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        raise ValueError("No Linear layer found in Sequential head")
    else:
        return head.in_features


def get_logits(model: nn.Module, data: dict) -> torch.Tensor:
    """Get model logits for input data.

    Handles both PointNeXt (expects dict) and Point-MAE (expects tensor).

    Args:
        model: The model
        data: Input data dict with 'pos' and 'x' keys

    Returns:
        Logits tensor
    """
    # Get the actual model (unwrap DataParallel if needed)
    actual_model = model.module if hasattr(model, "module") else model

    # Point-MAE models have group_divider - they expect tensor [B, N, 3]
    # PointNeXt models have encoder - they expect dict with 'pos' and 'x'
    if hasattr(actual_model, "group_divider") and isinstance(data, dict):
        return model(data["pos"])
    return model(data)


# Backward compatibility aliases
def load_model_by_architecture(
    ckpt_path: str,
    config_path: str,
    device: torch.device,
    architecture: str = "single",
    freeze: bool = True,
) -> Tuple[nn.Module, int]:
    """Deprecated: Use load_model() instead.

    Kept for backward compatibility. Always returns num_heads=1.
    """
    if architecture != "single":
        logger.warning(
            f"Multi-head architecture is deprecated. Loading as single-head."
        )
    model, _ = load_standard_model(ckpt_path, config_path, device, freeze=freeze)
    return model, 1


def load_model_with_head_detection(
    ckpt_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
) -> Tuple[nn.Module, int]:
    """Deprecated: Use load_model() instead.

    Kept for backward compatibility. Always returns num_heads=1.
    """
    model = load_model(ckpt_path, device, config_path, freeze=True)
    return model, 1


def get_head0_logits(model: nn.Module, data: dict) -> torch.Tensor:
    """Deprecated: Use get_logits() instead.

    Kept for backward compatibility.
    """
    return get_logits(model, data)


def get_averaged_logits(
    model: nn.Module, data: dict, num_heads: int = 1
) -> torch.Tensor:
    """Deprecated: Use get_logits() instead.

    Kept for backward compatibility.
    """
    return get_logits(model, data)
