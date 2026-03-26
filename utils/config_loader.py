"""Config loading utilities for integrating PointNeXt configs with pruning settings."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig

logger = logging.getLogger(__name__)


def load_pointnext_config(config_path: str) -> EasyConfig:
    """Load PointNeXt YAML config using their EasyConfig system.

    Handles hierarchical merging:
    - cfgs/default.yaml (base)
    - cfgs/dataset_name/default.yaml
    - cfgs/dataset_name/model_name.yaml

    Args:
        config_path: Path to PointNeXt config (e.g., cfgs/modelnet40ply2048/pointnext-s.yaml)

    Returns:
        EasyConfig object (PointNeXt's native config format)

    Raises:
        FileNotFoundError: If config not found
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"PointNeXt config not found: {config_path}\n"
            f"Please ensure the path is correct."
        )

    # Load using PointNeXt's hierarchical merging
    cfg = EasyConfig()
    cfg.load(str(config_path), recursive=True)

    return cfg


def merge_pruning_config(
    pointnext_cfg: EasyConfig, pruning_cfg: DictConfig
) -> EasyConfig:
    """Merge pruning config into PointNeXt config.

    Strategy:
    1. Apply overrides from pruning_cfg to pointnext_cfg (in-place)
    2. Add pruning, wandb, seed fields to the config
    3. Return the modified EasyConfig

    Args:
        pointnext_cfg: Loaded PointNeXt EasyConfig
        pruning_cfg: Pruning-specific DictConfig from Hydra

    Returns:
        EasyConfig with overrides applied and additional fields:
        - openpoint namespace contains original + overridden config
        - pruning: pruning-specific settings
        - wandb: wandb settings
        - seed: random seed
    """
    # Apply overrides to PointNeXt config
    if "overrides" in pruning_cfg:
        # Convert DictConfig overrides to dict and update EasyConfig
        overrides_dict = OmegaConf.to_container(pruning_cfg.overrides, resolve=True)
        pointnext_cfg.update(overrides_dict)
        logger.info(
            f"Applied {len(pruning_cfg.overrides)} overrides to PointNeXt config"
        )

    # Create wrapper config structure
    # Convert DictConfig to dict, then to EasyConfig for attribute access
    final_cfg = EasyConfig()
    final_cfg.openpoint = pointnext_cfg

    # Convert to dict then wrap in EasyConfig for both dict and attribute access
    pruning_dict = OmegaConf.to_container(pruning_cfg.pruning, resolve=True)
    wandb_dict = OmegaConf.to_container(pruning_cfg.get("wandb", {}), resolve=True)

    final_cfg.pruning = EasyConfig()
    final_cfg.pruning.update(pruning_dict)

    final_cfg.wandb = EasyConfig()
    final_cfg.wandb.update(wandb_dict)

    final_cfg.seed = pruning_cfg.get("seed", 42)

    return final_cfg


def merge_composition_config(
    pointnext_cfg: EasyConfig, comp_cfg: DictConfig
) -> EasyConfig:
    """Merge composition config with base PointNeXt config.

    Similar to merge_pruning_config but uses 'composition' key instead of 'pruning'.

    Args:
        pointnext_cfg: Base PointNeXt configuration from load_pointnext_config()
        comp_cfg: Composition configuration (DictConfig from Hydra)

    Returns:
        Merged EasyConfig with:
            - openpoint: PointNeXt training config (with overrides applied)
            - composition: Composition experiment settings
            - wandb: WandB logging settings
            - seed: Random seed
    """
    # Apply overrides to PointNeXt config
    if "overrides" in comp_cfg:
        overrides_dict = OmegaConf.to_container(comp_cfg.overrides, resolve=True)
        pointnext_cfg.update(overrides_dict)
        logger.info(f"Applied {len(comp_cfg.overrides)} overrides to PointNeXt config")

    # Create wrapper config structure
    final_cfg = EasyConfig()
    final_cfg.openpoint = pointnext_cfg

    # Convert composition settings
    composition_dict = OmegaConf.to_container(comp_cfg.composition, resolve=True)
    wandb_dict = OmegaConf.to_container(comp_cfg.get("wandb", {}), resolve=True)

    final_cfg.composition = EasyConfig()
    final_cfg.composition.update(composition_dict)

    final_cfg.wandb = EasyConfig()
    final_cfg.wandb.update(wandb_dict)

    final_cfg.seed = comp_cfg.get("seed", 42)

    return final_cfg


def merge_class_balanced_config(
    pointnext_cfg: EasyConfig, cb_cfg: DictConfig
) -> EasyConfig:
    """Merge class-balanced config into PointNeXt config.

    Similar to merge_pruning_config but for class-balanced retraining.

    Args:
        pointnext_cfg: Loaded PointNeXt EasyConfig
        cb_cfg: Class-balanced DictConfig from Hydra

    Returns:
        EasyConfig with overrides applied and additional fields:
        - openpoint namespace contains original + overridden config
        - class_balanced: class-balanced-specific settings
        - wandb: wandb settings
        - seed: random seed
    """
    # Apply overrides to PointNeXt config
    if "overrides" in cb_cfg:
        overrides_dict = OmegaConf.to_container(cb_cfg.overrides, resolve=True)
        pointnext_cfg.update(overrides_dict)
        logger.info(f"Applied {len(cb_cfg.overrides)} overrides to PointNeXt config")

    # Create wrapper config structure
    final_cfg = EasyConfig()
    final_cfg.openpoint = pointnext_cfg

    # Convert to dict then wrap in EasyConfig
    cb_dict = OmegaConf.to_container(cb_cfg.class_balanced, resolve=True)
    wandb_dict = OmegaConf.to_container(cb_cfg.get("wandb", {}), resolve=True)

    final_cfg.class_balanced = EasyConfig()
    final_cfg.class_balanced.update(cb_dict)

    final_cfg.wandb = EasyConfig()
    final_cfg.wandb.update(wandb_dict)

    final_cfg.seed = cb_cfg.get("seed", 42)

    return final_cfg


def find_config_for_checkpoint(ckpt_path: str) -> str:
    """Find config YAML for a checkpoint.

    Simple search strategy:
    1. Look for <ckpt_dir>/<dir_name>.yaml
    2. Look for <ckpt_dir>/config.yaml
    3. Check checkpoint metadata for config_path
    4. Raise error if not found

    Example:
        checkpoints/modelnet40/pointnext-s/best.pth
        → checkpoints/modelnet40/pointnext-s.yaml
        OR checkpoints/modelnet40/pointnext-s/config.yaml

    Args:
        ckpt_path: Path to checkpoint file

    Returns:
        Path to config file

    Raises:
        FileNotFoundError: If config cannot be found with helpful error message
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_dir = ckpt_path.parent

    # Strategy 1: Look for model-specific yaml (e.g., pointnet++.yaml) in run directory
    # For structure: log/dataset/run_dir/checkpoint/ckpt.pth
    # We want: log/dataset/run_dir/{model}.yaml
    run_dir = ckpt_dir.parent if ckpt_dir.name == "checkpoint" else ckpt_dir

    yaml_candidates = []

    # Find all .yaml files in run_dir, but EXCLUDE cfg.yaml (runtime config)
    if run_dir.exists() and run_dir.is_dir():
        yaml_files = list(run_dir.glob("*.yaml"))
        # Filter: exclude cfg.yaml, only keep model configs
        yaml_candidates.extend([f for f in yaml_files if f.name != "cfg.yaml"])

    for yaml_path in yaml_candidates:
        if yaml_path.exists():
            logger.info(f"Found config for checkpoint: {yaml_path}")
            return str(yaml_path)

    # Strategy 3: Check checkpoint metadata
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config_path" in ckpt:
        config_path = ckpt["config_path"]
        if Path(config_path).exists():
            logger.info(f"Found config from checkpoint metadata: {config_path}")
            return config_path
            
    # Not found - raise clear error
    searched_paths = "\n".join([f"  - {p}" for p in yaml_candidates])
    raise FileNotFoundError(
        f"Cannot find config for checkpoint: {ckpt_path}\n"
        f"Searched locations:\n"
        f"{searched_paths}\n"
        f"  - Checkpoint metadata['config_path']\n"
        f"\nPlease:\n"
        f"  1. Place a cfg.yaml or model-specific yaml in the run directory, OR\n"
        f"  2. Specify config manually using pruning.score_config in your config"
    )


def load_model_from_checkpoint(
    ckpt_path: str, config_path: Optional[str] = None, freeze: bool = True
) -> tuple[nn.Module, DictConfig]:
    """Load model from checkpoint with optional auto-config discovery.

    Args:
        ckpt_path: Path to checkpoint file
        config_path: Optional path to config. If None, auto-discover
        freeze: If True (default), set all parameters' requires_grad to False

    Returns:
        Tuple of (model, config)
        - model: Loaded PyTorch model in eval mode
        - config: Model configuration

    Raises:
        FileNotFoundError: If checkpoint or config not found
    """
    # Auto-discover config if not specified
    if config_path is None:
        logger.info(f"Auto-discovering config for: {ckpt_path}")
        config_path = find_config_for_checkpoint(ckpt_path)

    # Use EasyConfig (same as quick_load_model.py) for proper object structure
    cfg = EasyConfig()
    cfg.load(config_path, recursive=True)

    # EasyConfig objects support attribute access (cfg.model.radius)
    # which is required by build_model_from_cfg
    model = build_model_from_cfg(cfg.model)

    # Load checkpoint weights (same logic as openpoints.utils.load_checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state dict - check multiple possible keys
    state_dict = ckpt
    for key in ["model", "net", "network", "state_dict", "base_model"]:
        if key in ckpt:
            state_dict = ckpt[key]
            break

    # Clean up "module." prefix from DDP training
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    # Freeze parameters for inference-only use cases unless explicitly disabled
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        logger.info(f"Loaded model from {ckpt_path} (parameters frozen)")
    else:
        logger.info(f"Loaded model from {ckpt_path} (parameters trainable)")

    return model, cfg
