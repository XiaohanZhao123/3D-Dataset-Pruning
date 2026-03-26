"""Utility functions for building pruning pipelines.

This module provides factory functions for creating feature extractors and scoring functions
used in data pruning workflows.
"""

import logging
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Callable

from pruning.scorers import (
    make_loss_scorer,
    make_prototype_scorer,
    make_cluster_scorer,
    make_herding_scorer,
)

logger = logging.getLogger(__name__)


def build_logits_pipe(
    model: nn.Module, cfg, device: torch.device
) -> Callable:
    """Build pipeline that extracts logits (full forward pass).

    Used for loss-based scoring where final predictions are needed.

    Args:
        model: Pre-trained model (expects BaseCls with encoder + prediction)
        cfg: Config with model settings (score_cfg from checkpoint)
        device: Device to run model on

    Returns:
        Function that maps data batches to logits [B, num_classes]
    """
    model.to(device)
    model.eval()

    # Get inference batch size from full config if available
    inference_batch_size = getattr(cfg, 'inference_batch_size', 32)
    if hasattr(cfg, 'pruning'):
        inference_batch_size = cfg.pruning.get('inference_batch_size', inference_batch_size)

    def logits_pipe(data_batch):
        """Extract logits from data batch via full forward pass.

        Handles both dict (PointNeXt) and Batch (PyG) formats.
        """
        with torch.inference_mode():
            # PointNeXt dict format
            if isinstance(data_batch, dict):
                # Move tensors to device
                data_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in data_batch.items()
                }

                batch_size = data_batch['x'].shape[0] if 'x' in data_batch else 1

                if batch_size > inference_batch_size:
                    # Batched inference for large batches
                    logger.debug(f"Batched inference: {batch_size} samples")
                    all_logits = []

                    for start_idx in range(0, batch_size, inference_batch_size):
                        end_idx = min(start_idx + inference_batch_size, batch_size)
                        sub_batch = {
                            k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v
                            for k, v in data_batch.items()
                        }

                        # Prepare PointNeXt format
                        if 'x' in sub_batch:
                            points = sub_batch['x']
                            in_channels = cfg.model.encoder_args.in_channels
                            sub_batch['pos'] = points[:, :, :3].contiguous()
                            sub_batch['x'] = points[:, :, :in_channels].transpose(1, 2).contiguous()

                        all_logits.append(model(sub_batch))

                    return torch.cat(all_logits, dim=0)
                else:
                    # Single batch inference
                    if 'x' in data_batch:
                        points = data_batch['x']
                        in_channels = cfg.model.encoder_args.in_channels
                        data_batch['pos'] = points[:, :, :3].contiguous()
                        data_batch['x'] = points[:, :, :in_channels].transpose(1, 2).contiguous()

                    return model(data_batch)

            # PyG Batch format
            elif hasattr(data_batch, 'num_graphs') and data_batch.num_graphs > inference_batch_size:
                logger.debug(f"Batched inference: {data_batch.num_graphs} samples")
                all_logits = []
                data_list = data_batch.to_data_list()

                for start_idx in range(0, len(data_list), inference_batch_size):
                    end_idx = min(start_idx + inference_batch_size, len(data_list))
                    sub_batch = Batch.from_data_list(data_list[start_idx:end_idx])
                    sub_batch = sub_batch.to(device)
                    all_logits.append(model(sub_batch))

                return torch.cat(all_logits, dim=0)
            else:
                # PyG single batch
                data_batch = data_batch.to(device)
                return model(data_batch)

    return logits_pipe


def build_feature_pipe(
    model: nn.Module, cfg, device: torch.device
) -> Callable:
    """Build pipeline that extracts penultimate features (encoder output only).

    Used for prototype/cluster/herding scoring where embeddings are needed.

    Args:
        model: Pre-trained model (expects BaseCls with encoder.forward_cls_feat())
        cfg: Config with model settings (score_cfg from checkpoint)
        device: Device to run model on

    Returns:
        Function that maps data batches to features [B, feature_dim]
    """
    model.to(device)
    model.eval()

    # Check model has encoder
    if not hasattr(model, 'encoder'):
        raise ValueError(
            f"Model {type(model).__name__} does not have 'encoder' attribute. "
            "Feature extraction requires BaseCls or compatible model."
        )

    encoder = model.encoder

    # Get inference batch size
    inference_batch_size = getattr(cfg, 'inference_batch_size', 32)
    if hasattr(cfg, 'pruning'):
        inference_batch_size = cfg.pruning.get('inference_batch_size', inference_batch_size)

    def feature_pipe(data_batch):
        """Extract penultimate features from data batch via encoder.

        Handles both dict (PointNeXt) and Batch (PyG) formats.
        """
        with torch.inference_mode():
            # PointNeXt dict format
            if isinstance(data_batch, dict):
                # Move tensors to device
                data_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in data_batch.items()
                }

                batch_size = data_batch['x'].shape[0] if 'x' in data_batch else 1

                if batch_size > inference_batch_size:
                    # Batched inference for large batches
                    logger.debug(f"Batched feature extraction: {batch_size} samples")
                    all_features = []

                    for start_idx in range(0, batch_size, inference_batch_size):
                        end_idx = min(start_idx + inference_batch_size, batch_size)
                        sub_batch = {
                            k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v
                            for k, v in data_batch.items()
                        }

                        # Prepare PointNeXt format
                        if 'x' in sub_batch:
                            points = sub_batch['x']
                            in_channels = cfg.model.encoder_args.in_channels
                            sub_batch['pos'] = points[:, :, :3].contiguous()
                            sub_batch['x'] = points[:, :, :in_channels].transpose(1, 2).contiguous()

                        # Extract features via encoder only
                        all_features.append(encoder.forward_cls_feat(sub_batch))

                    return torch.cat(all_features, dim=0)
                else:
                    # Single batch inference
                    if 'x' in data_batch:
                        points = data_batch['x']
                        in_channels = cfg.model.encoder_args.in_channels
                        data_batch['pos'] = points[:, :, :3].contiguous()
                        data_batch['x'] = points[:, :, :in_channels].transpose(1, 2).contiguous()

                    # Extract features via encoder only
                    return encoder.forward_cls_feat(data_batch)

            # PyG Batch format
            elif hasattr(data_batch, 'num_graphs') and data_batch.num_graphs > inference_batch_size:
                logger.debug(f"Batched feature extraction: {data_batch.num_graphs} samples")
                all_features = []
                data_list = data_batch.to_data_list()

                for start_idx in range(0, len(data_list), inference_batch_size):
                    end_idx = min(start_idx + inference_batch_size, len(data_list))
                    sub_batch = Batch.from_data_list(data_list[start_idx:end_idx])
                    sub_batch = sub_batch.to(device)
                    all_features.append(encoder.forward_cls_feat(sub_batch))

                return torch.cat(all_features, dim=0)
            else:
                # PyG single batch
                data_batch = data_batch.to(device)
                return encoder.forward_cls_feat(data_batch)

    return feature_pipe


def build_score_fn(cfg, score_model: nn.Module, score_cfg, device: torch.device, num_classes: int = None):
    """Build scoring function based on config with appropriate feature extraction.

    Automatically selects the correct feature pipe:
    - Loss scorer: Uses logits_pipe (full forward pass)
    - Prototype/cluster/herding: Uses feature_pipe (encoder only)

    Args:
        cfg: Full config with pruning.score_fn settings
        score_model: Pre-trained model for scoring
        score_cfg: Config used to build the scoring model
        device: Device to run model on
        num_classes: Number of classes in the dataset (required for herding)

    Returns:
        Scoring function that maps samples to scores

    Raises:
        ValueError: If unknown score function name or if herding requires num_classes
    """
    score_fn_name = cfg.pruning.score_fn.name

    # Select appropriate feature extraction pipeline
    if score_fn_name == "loss":
        # Loss scoring needs logits (full forward pass)
        pipe = build_logits_pipe(score_model, score_cfg, device)
        score_fn = make_loss_scorer(pipe, nn.functional.cross_entropy)
        logger.info(f"Built loss scorer with logits pipe")

    elif score_fn_name == "prototype":
        # Prototype scoring needs features (encoder only)
        pipe = build_feature_pipe(score_model, score_cfg, device)
        score_fn = make_prototype_scorer(pipe)
        logger.info(f"Built prototype scorer with feature pipe")

    elif score_fn_name == "cluster":
        # Cluster scoring needs features (encoder only)
        pipe = build_feature_pipe(score_model, score_cfg, device)
        n_clusters = cfg.pruning.score_fn.n_clusters
        score_fn = make_cluster_scorer(pipe, n_clusters)
        logger.info(f"Built cluster scorer with feature pipe (n_clusters={n_clusters})")

    elif score_fn_name == "herding":
        # Herding scoring needs features (encoder only)
        pipe = build_feature_pipe(score_model, score_cfg, device)

        # Herding requires max mode
        if cfg.pruning.mode != "max":
            logger.warning("Herding scorer requires max mode, forcing mode to 'max'")
            cfg.pruning.mode = "max"

        # Herding needs per-class sample count
        if num_classes is None:
            raise ValueError("Herding scorer requires num_classes to compute per-class budget")
        
        if cfg.pruning.total_samples % num_classes != 0:
            raise ValueError(
                f"total_samples ({cfg.pruning.total_samples}) must be divisible by "
                f"num_classes ({num_classes}) for herding scorer"
            )
        
        samples_per_class = cfg.pruning.total_samples // num_classes
        score_fn = make_herding_scorer(pipe, samples_per_class)
        logger.info(f"Built herding scorer with feature pipe (samples_per_class={samples_per_class})")

    else:
        raise ValueError(f"Unknown score function: {score_fn_name}")

    return score_fn
