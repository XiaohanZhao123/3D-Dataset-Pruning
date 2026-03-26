"""Shared data preparation utilities for training and scoring."""

from typing import Tuple

import numpy as np
import torch
from openpoints.models.layers import furthest_point_sample


def move_to_device(data: dict, device: torch.device) -> dict:
    """Move tensor values in a batch dict to device (in-place)."""
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device, non_blocking=True)
    return data


def resample_points_fps(points: torch.Tensor, npoints: int) -> torch.Tensor:
    """Resample points using FPS + random selection.

    Args:
        points: [B, N, C] point cloud tensor
        npoints: target number of points (1024, 4096, 8192)

    Returns:
        Resampled points [B, npoints, C]
    """
    num_curr_pts = points.shape[1]
    if num_curr_pts <= npoints:
        return points

    # Determine FPS pool size based on target
    if npoints == 1024:
        point_all = 1200
    elif npoints == 4096:
        point_all = 4800
    elif npoints == 8192:
        point_all = 8192
    else:
        point_all = num_curr_pts

    point_all = min(point_all, points.size(1))

    # FPS + random selection
    fps_idx = furthest_point_sample(points[:, :, :3].contiguous(), point_all)
    fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
    return torch.gather(
        points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1])
    )


def prepare_data_dict(points: torch.Tensor, cfg) -> dict:
    """Prepare data dict for model forward pass.

    Args:
        points: [B, N, C] point cloud tensor
        cfg: config with model.encoder_args.in_channels

    Returns:
        dict with 'pos' [B, N, 3] and 'x' [B, C, N]

    Note:
        Uses min(in_channels, actual_channels) to handle cases where
        the data has fewer channels than the model expects (e.g., ScanObjectNN
        with 3 channels but model configured for 7).
    """
    data = {}
    data["pos"] = points[:, :, :3].contiguous()
    in_channels = (
        cfg.model.encoder_args.in_channels if hasattr(cfg.model, "encoder_args") else 3
    )
    # Handle case where data has fewer channels than model expects
    actual_channels = points.shape[-1]
    use_channels = min(in_channels, actual_channels)
    data["x"] = points[:, :, :use_channels].transpose(1, 2).contiguous()
    return data


def prepare_batch(
    data: dict,
    cfg,
    device: torch.device,
    npoints: int | None = None,
    resample: bool = True,
    truncate: bool = False,
) -> Tuple[dict, torch.Tensor]:
    """Move batch to device, adjust points, and build model input dict.

    Args:
        data: Raw batch dict from dataloader
        cfg: OpenPoint config with num_points and encoder_args
        device: Target torch device
        npoints: Number of points to use (defaults to cfg.num_points)
        resample: If True, apply FPS-based resampling
        truncate: If True and resample=False, truncate to npoints

    Returns:
        (prepared_data, target) tuple
    """
    move_to_device(data, device)

    target = data["y"]
    points = data["x"]
    if npoints is None:
        npoints = cfg.num_points

    if npoints is not None:
        if resample:
            points = resample_points_fps(points, npoints)
        elif truncate:
            points = points[:, :npoints]

    data.update(prepare_data_dict(points, cfg))
    return data, target
