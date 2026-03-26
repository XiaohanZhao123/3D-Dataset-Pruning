"""Training utilities shared across pruning and retraining scripts.

Extracted from prune.py, class_balanced_retrain.py, and prune_with_balanced_model.py
to eliminate code duplication while following KISS principle.
"""

import logging
from utils.data_prep import prepare_data_dict, resample_points_fps
from utils.kd import compute_kd_loss

logger = logging.getLogger(__name__)


def setup_model_config(cfg):
    """Setup criterion_args and in_channels before model building.

    Args:
        cfg: OpenPoint config object with model settings
    """
    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels


def compute_kd_loss_standard(student_logits, teacher_logits, hard_loss, alpha, temperature):
    """Compute standard knowledge distillation loss (single-head).

    Args:
        student_logits: [B, num_classes] student predictions
        teacher_logits: [B, num_classes] teacher predictions
        hard_loss: ground truth loss (cross entropy)
        alpha: distillation loss weight (1-alpha for hard loss)
        temperature: softmax temperature for KD

    Returns:
        Combined KD loss
    """
    return compute_kd_loss(
        student_logits,
        teacher_logits,
        hard_loss,
        alpha,
        temperature,
    )


def build_student_model(cfg, device):
    """Build and initialize student model.

    Args:
        cfg: OpenPoint config with model configuration
        device: torch device

    Returns:
        tuple: (model, model_size)
            model: Built model on device
            model_size: Number of parameters
    """
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import cal_model_parm_nums

    setup_model_config(cfg)
    model = build_model_from_cfg(cfg.model).to(device)

    model_size = cal_model_parm_nums(model)
    logger.info(f'Model parameters: {model_size / 1e6:.2f}M')

    return model, model_size


def setup_experiment(cfg, default_exp_name='experiment'):
    """Setup experiment directory and generate run name.

    Args:
        cfg: Config with ckpt_dir attribute
        default_exp_name: Default name if exp_name not in config

    Returns:
        run_name: Generated unique run name
    """
    import time
    import pathlib
    import shortuuid

    exp_name = cfg.get('exp_name', default_exp_name)
    if not hasattr(cfg, 'run_name'):
        expid = time.strftime('%Y%m%d-%H%M%S-') + str(shortuuid.uuid())
        cfg.run_name = f'{exp_name}-{expid}'

    pathlib.Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint directory: {cfg.ckpt_dir}")
    logger.info(f"Run name: {cfg.run_name}")

    return cfg.run_name


def build_dataloaders(cfg, splits=['train', 'val'], distributed=False):
    """Build dataloaders for specified splits.

    Args:
        cfg: Config with dataset, dataloader, datatransforms
        splits: List of splits to build loaders for
        distributed: Whether to use distributed training

    Returns:
        dict mapping split name to dataloader
    """
    from openpoints.dataset import build_dataloader_from_cfg

    loaders = {}
    for split in splits:
        batch_size = (
            cfg.get('val_batch_size', cfg.batch_size)
            if split == 'val'
            else cfg.batch_size
        )
        loaders[split] = build_dataloader_from_cfg(
            batch_size,
            cfg.dataset,
            cfg.dataloader,
            datatransforms_cfg=cfg.datatransforms,
            split=split,
            distributed=distributed
        )
    return loaders
