"""Dedicated Point-MAE data loader with GPU FPS and caching.

This module provides a data loader specifically designed for Point-MAE models,
matching the exact preprocessing pipeline used during Point-MAE training:
1. Load raw points from .txt files
2. FPS to 8192 points (cached)
3. pc_normalize (center + scale to unit sphere)
4. Shuffle for training
5. Apply PointcloudScaleAndTranslate augmentation

Usage:
    from utils.pointmae_dataloader import PointMAEModelNet40

    train_dataset = PointMAEModelNet40(
        data_dir='data/ModelNet/modelnet40_normal_resampled',
        split='train',
        num_points=8192,
    )
"""

import os
import logging
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def pc_normalize(pc):
    """Normalize point cloud to unit sphere (Point-MAE version).

    Centers the point cloud and scales to fit in unit sphere.

    Args:
        pc: numpy array [N, 3] or [N, D]

    Returns:
        Normalized point cloud
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def torch_cluster_fps(points, npoints, device='cuda'):
    """Farthest Point Sampling using torch_cluster (GPU).

    Args:
        points: numpy array [N, D]
        npoints: number of points to sample
        device: 'cuda' or 'cpu'

    Returns:
        Sampled points [npoints, D] as numpy array
    """
    from torch_cluster import fps

    N, D = points.shape

    # Convert to tensor
    points_tensor = torch.from_numpy(points).float().to(device)

    # Compute ratio
    ratio = npoints / N

    # FPS returns indices
    idx = fps(points_tensor[:, :3], ratio=ratio, random_start=True)

    # Ensure exact number of points
    if len(idx) > npoints:
        idx = idx[:npoints]
    elif len(idx) < npoints:
        # Pad with random points if needed
        extra_idx = torch.randint(0, N, (npoints - len(idx),), device=device)
        idx = torch.cat([idx, extra_idx])

    sampled = points_tensor[idx].cpu().numpy()
    return sampled


def numpy_fps(point, npoint):
    """Farthest Point Sampling (NumPy version, for CPU fallback).

    This is the original Point-MAE implementation.

    Args:
        point: pointcloud data [N, D]
        npoint: number of samples

    Returns:
        Sampled pointcloud [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return point[centroids]


class PointMAEModelNet40(Dataset):
    """ModelNet40 dataset loader matching Point-MAE preprocessing exactly.

    Features:
    - Loads from modelnet40_normal_resampled format (.txt files)
    - GPU-accelerated FPS via torch_cluster
    - Caches FPS results to .dat files
    - pc_normalize after FPS
    - Shuffles points for training
    - Can load directly from cache if raw data not available

    Args:
        data_dir: Path to modelnet40_normal_resampled directory
        split: 'train' or 'test'
        num_points: Number of points (default 8192)
        use_normals: Whether to include normal vectors (default False)
        use_gpu_fps: Use GPU for FPS (default True)
        cache_fps: Cache FPS results (default True)
        cache_only: If True, only use cache (don't require raw data)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_points: int = 8192,
        use_normals: bool = False,
        use_gpu_fps: bool = True,
        cache_fps: bool = True,
        num_classes: int = 40,
        cache_only: bool = False,
    ):
        self.root = Path(data_dir)
        self.split = 'train' if split.lower() == 'train' else 'test'
        self.npoints = num_points
        self.use_normals = use_normals
        self.use_gpu_fps = use_gpu_fps and torch.cuda.is_available()
        self.cache_fps = cache_fps
        self.num_category = num_classes
        self.cache_only = cache_only

        # Setup caching first (may load from cache without needing raw data)
        self.list_of_points = None
        self.list_of_labels = None
        self.datapath = None

        # Check for cache file
        cache_name = f'modelnet{self.num_category}_{self.split}_{self.npoints}pts_fps.dat'
        self.cache_path = self.root / cache_name

        if self.cache_path.exists():
            logger.info(f'Loading cached FPS data from {self.cache_path}')
            with open(self.cache_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)
            logger.info(f'Loaded {len(self.list_of_points)} cached samples')
            # Create dummy datapath for length
            self.datapath = [(None, None)] * len(self.list_of_points)
        elif cache_only:
            raise FileNotFoundError(
                f"Cache file not found: {self.cache_path}. "
                f"Set cache_only=False to build from raw data."
            )
        else:
            # Need to load from raw data
            self._load_raw_data()
            if self.cache_fps:
                self._build_cache()

    def _load_raw_data(self):
        """Load raw data file list."""
        # Load class names
        if self.num_category == 10:
            catfile = self.root / 'modelnet10_shape_names.txt'
        else:
            catfile = self.root / 'modelnet40_shape_names.txt'

        if not catfile.exists():
            raise FileNotFoundError(f"Class names file not found: {catfile}")

        self.cat = [line.rstrip() for line in open(catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # Load file list
        if self.num_category == 10:
            split_file = f'modelnet10_{self.split}.txt'
        else:
            split_file = f'modelnet40_{self.split}.txt'

        split_path = self.root / split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        shape_ids = [line.rstrip() for line in open(split_path)]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]

        self.datapath = [
            (shape_names[i], self.root / shape_names[i] / f'{shape_ids[i]}.txt')
            for i in range(len(shape_ids))
        ]

        logger.info(f'PointMAEModelNet40: {self.split} split, {len(self.datapath)} samples')

    def _build_cache(self):
        """Build FPS cache for all samples."""
        self.list_of_points = [None] * len(self.datapath)
        self.list_of_labels = [None] * len(self.datapath)

        for idx in tqdm(range(len(self.datapath)), desc='Caching FPS'):
            class_name, filepath = self.datapath[idx]
            cls = self.classes[class_name]

            # Load raw points
            point_set = np.loadtxt(filepath, delimiter=',').astype(np.float32)

            # FPS to num_points
            if self.use_gpu_fps:
                point_set = torch_cluster_fps(point_set, self.npoints)
            else:
                point_set = numpy_fps(point_set, self.npoints)

            self.list_of_points[idx] = point_set
            self.list_of_labels[idx] = np.array([cls]).astype(np.int32)

        # Save cache
        logger.info(f'Saving FPS cache to {self.cache_path}')
        with open(self.cache_path, 'wb') as f:
            pickle.dump([self.list_of_points, self.list_of_labels], f)

    def __len__(self):
        return len(self.datapath)

    @property
    def targets(self):
        """Return labels for sampler compatibility."""
        if self.list_of_labels is not None:
            return np.array([l[0] for l in self.list_of_labels])
        # Fallback: extract from class names
        return np.array([self.classes[self.datapath[i][0]] for i in range(len(self))])

    @property
    def labels(self):
        """Alias for targets."""
        return self.targets

    def __getitem__(self, index):
        """Get a single sample.

        Returns:
            dict with 'pos' [N, 3] and 'y' (label)
        """
        # Get points and label
        if self.list_of_points is not None:
            point_set = self.list_of_points[index].copy()
            label = self.list_of_labels[index][0]
        else:
            # Load on the fly (no cache)
            class_name, filepath = self.datapath[index]
            label = self.classes[class_name]
            point_set = np.loadtxt(filepath, delimiter=',').astype(np.float32)

            if self.use_gpu_fps:
                point_set = torch_cluster_fps(point_set, self.npoints)
            else:
                point_set = numpy_fps(point_set, self.npoints)

        # pc_normalize (Point-MAE does this in _get_item)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # Use only xyz or include normals
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        # Shuffle for training (Point-MAE behavior)
        if self.split == 'train':
            np.random.shuffle(point_set)

        # Convert to tensor
        points = torch.from_numpy(point_set).float()

        # Return in dict format for compatibility
        return {
            'pos': points,
            'y': label,
            'x': points,  # For compatibility with some models
        }


# ============================================================================
# Point-MAE Augmentations (ported exactly from Point-MAE)
# ============================================================================

class PointcloudScaleAndTranslate:
    """Point-MAE's scale and translate augmentation.

    Ported exactly from third_party/pointmae/datasets/data_transforms.py

    Applied during BOTH training and testing in Point-MAE.
    """

    def __init__(self, scale_low=2./3., scale_high=3./2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        """Apply augmentation.

        Args:
            pc: [B, N, 3] tensor on GPU

        Returns:
            Augmented point cloud [B, N, 3]
        """
        bsize = pc.size()[0]
        device = pc.device

        for i in range(bsize):
            # Random scale per axis
            xyz1 = np.random.uniform(
                low=self.scale_low,
                high=self.scale_high,
                size=[3]
            )
            # Random translation
            xyz2 = np.random.uniform(
                low=-self.translate_range,
                high=self.translate_range,
                size=[3]
            )

            scale = torch.from_numpy(xyz1).float().to(device)
            translate = torch.from_numpy(xyz2).float().to(device)

            pc[i, :, 0:3] = pc[i, :, 0:3] * scale + translate

        return pc


class PointcloudRotate:
    """Point-MAE's rotation augmentation (around Y axis).

    Note: This is commented out in Point-MAE's finetune config.
    """

    def __call__(self, pc):
        """Apply rotation.

        Args:
            pc: [B, N, 3] tensor on GPU

        Returns:
            Rotated point cloud [B, N, 3]
        """
        bsize = pc.size()[0]
        device = pc.device

        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([
                [cosval, 0, sinval],
                [0, 1, 0],
                [-sinval, 0, cosval]
            ])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(device)
            pc[i, :, :] = torch.matmul(pc[i], R)

        return pc


class PointcloudJitter:
    """Point-MAE's jitter augmentation.

    Note: This is commented out in Point-MAE's finetune config.
    """

    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

    def __call__(self, pc):
        """Apply jitter.

        Args:
            pc: [B, N, 3] tensor on GPU

        Returns:
            Jittered point cloud [B, N, 3]
        """
        bsize = pc.size()[0]

        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data

        return pc


def get_pointmae_transforms(split='train'):
    """Get Point-MAE transforms for training or testing.

    Point-MAE uses PointcloudScaleAndTranslate for BOTH train and test.

    Args:
        split: 'train' or 'test'

    Returns:
        Compose of transforms
    """
    from torchvision import transforms

    # Point-MAE uses the same augmentation for both train and test
    # (See runner_finetune.py lines 15-34)
    return transforms.Compose([
        PointcloudScaleAndTranslate(),
    ])


# ============================================================================
# Helper functions
# ============================================================================

class PointMAEScanObjectNN(Dataset):
    """ScanObjectNN dataset loader for Point-MAE.

    Loads from h5 files in the ScanObjectNN format.
    Supports three variants:
    - hardest: training/test_objectdataset_augmentedrot_scale75.h5
    - objbg: training/test_objectdataset.h5 (main_split)
    - objonly: training/test_objectdataset.h5 (main_split_nobg)

    Args:
        data_dir: Path to ScanObjectNN h5_files directory
        split: 'train' or 'test'
        variant: 'hardest', 'objbg', or 'objonly' (default: 'hardest')
        num_points: Number of points to use (default 2048 - native resolution)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        variant: str = 'hardest',
        num_points: int = 2048,
    ):
        import h5py

        self.root = Path(data_dir)
        self.split = 'train' if split.lower() == 'train' else 'test'
        self.variant = variant.lower()
        self.npoints = num_points

        # Determine which h5 files to load based on variant
        if self.variant == 'hardest':
            subdir = 'main_split'
            if self.split == 'train':
                h5_file = 'training_objectdataset_augmentedrot_scale75.h5'
            else:
                h5_file = 'test_objectdataset_augmentedrot_scale75.h5'
        elif self.variant == 'objbg':
            subdir = 'main_split'
            if self.split == 'train':
                h5_file = 'training_objectdataset.h5'
            else:
                h5_file = 'test_objectdataset.h5'
        elif self.variant == 'objonly':
            subdir = 'main_split_nobg'
            if self.split == 'train':
                h5_file = 'training_objectdataset.h5'
            else:
                h5_file = 'test_objectdataset.h5'
        else:
            raise ValueError(f"Unknown variant: {self.variant}. Use 'hardest', 'objbg', or 'objonly'")

        h5_path = self.root / subdir / h5_file
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        logger.info(f"Loading ScanObjectNN {self.variant} from {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.list_of_labels = np.array(f['label']).astype(np.int64)

        logger.info(f"Loaded {len(self.points)} samples, shape: {self.points.shape}")

    def __len__(self):
        return len(self.points)

    @property
    def targets(self):
        """Return labels for sampler compatibility."""
        return self.list_of_labels

    @property
    def labels(self):
        """Alias for targets."""
        return self.targets

    def __getitem__(self, index):
        """Get a single sample.

        Returns:
            dict with 'pos' [N, 3] and 'y' (label)
        """
        point_set = self.points[index].copy()
        label = self.list_of_labels[index]

        # Shuffle point indices for training (Point-MAE behavior)
        pt_idxs = np.arange(0, point_set.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)

        point_set = point_set[pt_idxs]

        # Subsample if needed (ScanObjectNN is 2048 points natively)
        if self.npoints < point_set.shape[0]:
            point_set = point_set[:self.npoints]

        # Convert to tensor
        points = torch.from_numpy(point_set).float()

        # Return in dict format for compatibility
        return {
            'pos': points,
            'y': label,
            'x': points,
        }


def build_pointmae_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 32,
    num_points: int = 8192,
    num_workers: int = 4,
    shuffle: bool = None,
    sampler = None,
):
    """Build a DataLoader for Point-MAE dataset.

    Args:
        data_dir: Path to ModelNet data
        split: 'train' or 'test'
        batch_size: Batch size
        num_points: Number of points per sample
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (default: True for train, False for test)
        sampler: Optional sampler (overrides shuffle)

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader

    dataset = PointMAEModelNet40(
        data_dir=data_dir,
        split=split,
        num_points=num_points,
    )

    if shuffle is None:
        shuffle = (split == 'train') and (sampler is None)

    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
