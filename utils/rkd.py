"""
Relational Knowledge Distillation (RKD) implementation.

Based on: "Relational Knowledge Distillation" (CVPR 2019)

RKD transfers structural knowledge from teacher to student via:
1. Distance-wise loss: Pairwise distance relations between samples
2. Angle-wise loss: Angular relations between sample triplets
"""

import torch
import torch.nn.functional as F


def pdist(
    embeddings: torch.Tensor, squared: bool = False, eps: float = 1e-12
) -> torch.Tensor:
    """Compute pairwise Euclidean distances between embeddings.

    Args:
        embeddings: [B, D] tensor of embeddings
        squared: If True, return squared distances
        eps: Small value for numerical stability

    Returns:
        [B, B] tensor of pairwise distances
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
    dot_product = torch.mm(embeddings, embeddings.t())
    sq_norms = torch.diag(dot_product)

    # D[i,j] = ||a_i||^2 + ||a_j||^2 - 2 * a_i.a_j
    distances_sq = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2.0 * dot_product
    distances_sq = torch.clamp(distances_sq, min=0.0)  # Numerical stability

    if squared:
        return distances_sq

    # Add eps only to non-zero elements to preserve exact zeros on diagonal
    distances = torch.sqrt(distances_sq + eps)
    # Zero out diagonal explicitly for numerical cleanliness
    distances = distances - torch.diag(torch.diag(distances))
    return distances


def rkd_distance_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute RKD distance-wise distillation loss.

    Transfers pairwise distance relations from teacher to student.
    ψ_dist(x_i, x_j) = d(x_i, x_j) / μ  where μ is mean distance
    L_dist = Σ_{i≠j} Huber(ψ_dist^T - ψ_dist^S)

    Args:
        student_embeddings: [B, D] student feature embeddings
        teacher_embeddings: [B, D] teacher feature embeddings
        eps: Small value for numerical stability

    Returns:
        Scalar distance loss
    """
    # Compute pairwise distances
    t_dist = pdist(teacher_embeddings, squared=False, eps=eps)
    s_dist = pdist(student_embeddings, squared=False, eps=eps)

    # Normalize by mean distance (excluding diagonal zeros)
    # Mean over non-diagonal elements
    batch_size = teacher_embeddings.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=student_embeddings.device)

    # Create mask for non-diagonal elements
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=teacher_embeddings.device)

    # Compute mean distances for normalization
    t_mean = t_dist[mask].mean()
    s_mean = s_dist[mask].mean()

    # Normalize distances
    t_dist_norm = t_dist / (t_mean + eps)
    s_dist_norm = s_dist / (s_mean + eps)

    # Huber loss (smooth L1) between normalized distances
    # Only consider non-diagonal elements (i ≠ j)
    loss = F.smooth_l1_loss(s_dist_norm[mask], t_dist_norm[mask], reduction="mean")

    return loss


def rkd_angle_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute RKD angle-wise distillation loss.

    Transfers angular relations between triplets from teacher to student.
    For triplet (i, j, k): e_ji = x_i - x_j, e_jk = x_k - x_j
    cos_ijk = (e_ji · e_jk) / (||e_ji|| * ||e_jk||)
    L_angle = Σ_{i,j,k distinct} Huber(cos^T_ijk - cos^S_ijk)

    Args:
        student_embeddings: [B, D] student feature embeddings
        teacher_embeddings: [B, D] teacher feature embeddings
        eps: Small value for numerical stability

    Returns:
        Scalar angle loss
    """
    batch_size = student_embeddings.size(0)
    if batch_size < 3:
        return torch.tensor(0.0, device=student_embeddings.device)

    def _angle_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        """Compute angle matrix for all triplets with middle vertex j.

        For efficiency, compute cos(angle_ijk) for all valid (i, j, k) triplets
        where j is the vertex of the angle.

        Returns: [B, B, B] tensor where [i, j, k] = cos(angle at j from i to k)
        """
        # Normalize embeddings
        emb_norm = F.normalize(embeddings, p=2, dim=1)  # [B, D]

        # Compute all pairwise difference vectors
        # e_ji = x_i - x_j for all (j, i) pairs
        # Shape: [B, B, D] where [j, i, :] = x_i - x_j
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # [B, B, D]

        # Normalize difference vectors
        diff_norm = F.normalize(diff, p=2, dim=2, eps=eps)  # [B, B, D]

        # Compute cosine of angles: cos_ijk = (e_ji · e_jk) / (||e_ji|| * ||e_jk||)
        # This is dot product of normalized vectors
        # For each j, compute dot product between diff_norm[j, i, :] and diff_norm[j, k, :]
        # Result shape: [B, B, B] where [i, j, k] = cos(angle at j)
        cos_angles = torch.bmm(diff_norm, diff_norm.transpose(1, 2))  # [B, B, B]

        return cos_angles

    # Compute angle matrices
    t_angles = _angle_matrix(teacher_embeddings)  # [B, B, B]
    s_angles = _angle_matrix(student_embeddings)  # [B, B, B]

    # Create mask for valid triplets (i ≠ j, j ≠ k, i ≠ k)
    # Actually for angle at j, we need i ≠ j and k ≠ j (i can equal k but gives cos=1)
    idx = torch.arange(batch_size, device=student_embeddings.device)
    # Mask where i != j and k != j
    mask_ij = idx.unsqueeze(0) != idx.unsqueeze(1)  # [B, B]
    mask_jk = idx.unsqueeze(0) != idx.unsqueeze(1)  # [B, B]
    # Combine: for triplet (i, j, k), need mask[i, j] && mask[j, k]
    # In our [B, B, B] tensor indexed as [j, i, k], we need i != j and k != j
    valid_mask = mask_ij.unsqueeze(2) & mask_jk.unsqueeze(1)  # [B, B, B]

    # Huber loss between teacher and student angles
    loss = F.smooth_l1_loss(
        s_angles[valid_mask], t_angles[valid_mask], reduction="mean"
    )

    return loss


def rkd_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    distance_weight: float = 1.0,
    angle_weight: float = 2.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined RKD loss.

    L_RKD = β * L_dist + γ * L_angle

    Default weights from paper:
    - Metric learning: β=25, γ=50
    - Image classification: β=1, γ=2

    Args:
        student_embeddings: [B, D] student feature embeddings
        teacher_embeddings: [B, D] teacher feature embeddings
        distance_weight: Weight for distance loss (β)
        angle_weight: Weight for angle loss (γ)
        eps: Small value for numerical stability

    Returns:
        Tuple of (total_rkd_loss, distance_loss, angle_loss)
    """
    dist_loss = rkd_distance_loss(student_embeddings, teacher_embeddings, eps)
    angle_loss = rkd_angle_loss(student_embeddings, teacher_embeddings, eps)

    total_loss = distance_weight * dist_loss + angle_weight * angle_loss

    return total_loss, dist_loss, angle_loss


def extract_embeddings_from_model(
    model, data_dict: dict, use_module: bool = False
) -> torch.Tensor:
    """Extract feature embeddings from model encoder.

    Args:
        model: Model with encoder attribute
        data_dict: Input data dictionary with 'pos' and 'x' keys
        use_module: If True, access model.module (for DataParallel)

    Returns:
        [B, D] feature embeddings
    """
    if use_module:
        encoder = model.module.encoder
    else:
        encoder = model.encoder

    # Get encoder output
    features = encoder.forward_cls_feat(data_dict)  # [B, D]

    return features


# =============================================================================
# Memory-Augmented RKD (MoCo-style)
# =============================================================================


def _compute_live_pair_mask(
    n_live: int, n_total: int, device: torch.device
) -> torch.Tensor:
    """Compute mask for pairs where at least one sample is live (from current batch).

    For memory-augmented RKD, we only compute loss on pairs that involve at least
    one "live" sample (current batch), since memory-memory pairs have no gradient.

    Args:
        n_live: Number of live samples (current batch size)
        n_total: Total samples (live + memory)
        device: Torch device

    Returns:
        [N, N] bool tensor where True = valid pair for loss computation
    """
    idx = torch.arange(n_total, device=device)
    is_live = idx < n_live  # [N]
    # Pair (i,j) valid if is_live[i] OR is_live[j]
    pair_has_live = is_live.unsqueeze(1) | is_live.unsqueeze(0)  # [N, N]
    # Also exclude diagonal (i != j)
    not_diagonal = ~torch.eye(n_total, dtype=torch.bool, device=device)
    return pair_has_live & not_diagonal


def _compute_live_triplet_mask(
    n_live: int, n_total: int, device: torch.device
) -> torch.Tensor:
    """Compute mask for triplets where at least one sample is live.

    For angle loss, we need triplets (i, j, k) where at least one index
    is from the current batch (live), since memory-only triplets have no gradient.

    Args:
        n_live: Number of live samples (current batch size)
        n_total: Total samples (live + memory)
        device: Torch device

    Returns:
        [N, N, N] bool tensor indexed as [j, i, k] for angle at vertex j
    """
    idx = torch.arange(n_total, device=device)
    is_live = idx < n_live  # [N]
    # Triplet (i,j,k) valid if is_live[i] OR is_live[j] OR is_live[k]
    # In [j, i, k] indexing:
    live_j = is_live.view(-1, 1, 1)  # [N, 1, 1]
    live_i = is_live.view(1, -1, 1)  # [1, N, 1]
    live_k = is_live.view(1, 1, -1)  # [1, 1, N]
    triplet_has_live = live_j | live_i | live_k  # [N, N, N]
    # Also need i != j and k != j for valid angle computation
    mask_neq = idx.unsqueeze(0) != idx.unsqueeze(1)  # [N, N]
    valid_ij_jk = mask_neq.unsqueeze(2) & mask_neq.unsqueeze(1)  # [N, N, N]
    return triplet_has_live & valid_ij_jk


def rkd_distance_loss_masked(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute RKD distance loss with custom pair mask.

    Args:
        student_embeddings: [N, D] student features (live + memory)
        teacher_embeddings: [N, D] teacher features (live + memory)
        mask: [N, N] bool mask for valid pairs
        eps: Numerical stability

    Returns:
        Scalar distance loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_embeddings.device)

    # Compute pairwise distances
    t_dist = pdist(teacher_embeddings, squared=False, eps=eps)
    s_dist = pdist(student_embeddings, squared=False, eps=eps)

    # Normalize by mean distance over valid pairs
    t_mean = t_dist[mask].mean()
    s_mean = s_dist[mask].mean()

    t_dist_norm = t_dist / (t_mean + eps)
    s_dist_norm = s_dist / (s_mean + eps)

    # Huber loss over valid pairs only
    loss = F.smooth_l1_loss(s_dist_norm[mask], t_dist_norm[mask], reduction="mean")
    return loss


def rkd_angle_loss_masked(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute RKD angle loss with custom triplet mask.

    Args:
        student_embeddings: [N, D] student features (live + memory)
        teacher_embeddings: [N, D] teacher features (live + memory)
        mask: [N, N, N] bool mask for valid triplets (indexed as [j, i, k])
        eps: Numerical stability

    Returns:
        Scalar angle loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_embeddings.device)

    n_total = student_embeddings.size(0)
    if n_total < 3:
        return torch.tensor(0.0, device=student_embeddings.device)

    def _angle_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        """Compute [N, N, N] angle matrix where [j, i, k] = cos(angle at j)."""
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # [N, N, D]
        diff_norm = F.normalize(diff, p=2, dim=2, eps=eps)  # [N, N, D]
        cos_angles = torch.bmm(diff_norm, diff_norm.transpose(1, 2))  # [N, N, N]
        return cos_angles

    t_angles = _angle_matrix(teacher_embeddings)
    s_angles = _angle_matrix(student_embeddings)

    loss = F.smooth_l1_loss(s_angles[mask], t_angles[mask], reduction="mean")
    return loss


def rkd_loss_with_anchor_mask(
    student_all: torch.Tensor,
    teacher_all: torch.Tensor,
    n_active: int,
    distance_weight: float = 1.0,
    angle_weight: float = 2.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RKD loss with anchor augmentation.

    Only computes loss on pairs/triplets involving at least one "active" sample
    (first n_active samples). Anchor samples (remaining) provide relational context
    but don't receive gradients.

    Args:
        student_all: [N, D] student embeddings (first n_active have grad, rest detached)
        teacher_all: [N, D] teacher embeddings (all detached)
        n_active: Number of active (gradient-bearing) samples
        distance_weight: Weight for distance loss (β)
        angle_weight: Weight for angle loss (γ)
        eps: Numerical stability

    Returns:
        (total_loss, distance_loss, angle_loss)
    """
    device = student_all.device
    n_total = student_all.size(0)

    # Compute masks for pairs/triplets with at least one active sample
    pair_mask = _compute_live_pair_mask(n_active, n_total, device)
    triplet_mask = _compute_live_triplet_mask(n_active, n_total, device)

    # Compute masked losses
    dist_loss = rkd_distance_loss_masked(student_all, teacher_all, pair_mask, eps)
    angle_loss = rkd_angle_loss_masked(student_all, teacher_all, triplet_mask, eps)

    total_loss = distance_weight * dist_loss + angle_weight * angle_loss

    return total_loss, dist_loss, angle_loss


# =============================================================================
# Prototype-Augmented RKD (Proto-RKD)
# =============================================================================


def proto_kd_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    prototypes: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Prototype distribution matching loss for Proto-RKD.

    Computes KL divergence between teacher and student distributions over
    class prototypes. This transfers global geometry knowledge from the full
    dataset (encoded in prototypes) to the student trained on a subset.

    For each sample i:
        p_T(k|i) = softmax(cos(t_i, c_k) / tau)  # teacher dist over prototypes
        p_S(k|i) = softmax(cos(s_i, c_k) / tau)  # student dist over prototypes
        L = KL(p_T || p_S)

    Args:
        student_emb: [B, D] student embeddings
        teacher_emb: [B, D] teacher embeddings
        prototypes: [K, D] class prototypes (L2 normalized)
        tau: Temperature for softmax (lower = sharper distribution)

    Returns:
        Scalar KL divergence loss
    """
    # Normalize embeddings for cosine similarity
    student_norm = F.normalize(student_emb, p=2, dim=1)  # [B, D]
    teacher_norm = F.normalize(teacher_emb, p=2, dim=1)  # [B, D]
    # prototypes should already be normalized

    # Compute cosine similarities to prototypes
    # [B, D] @ [D, K] -> [B, K]
    t_sim = torch.mm(teacher_norm, prototypes.t()) / tau
    s_sim = torch.mm(student_norm, prototypes.t()) / tau

    # Teacher distribution (target)
    p_teacher = F.softmax(t_sim, dim=1)  # [B, K]

    # Student log distribution
    log_p_student = F.log_softmax(s_sim, dim=1)  # [B, K]

    # KL divergence: KL(p_T || p_S) = sum(p_T * (log(p_T) - log(p_S)))
    loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean")

    return loss


class MemoryAugmentedRKD:
    """Memory-augmented RKD with MoCo-style queue.

    Maintains a FIFO queue of (teacher, student) embedding pairs from previous batches.
    Computes RKD loss over "virtual large batch" = current batch + memory samples.
    Gradients flow only through current batch's student embeddings.

    This enables effective relational learning even with small batch sizes (e.g., 16)
    by computing relations against a large memory bank (e.g., 384 samples).

    Example:
        >>> memory_rkd = MemoryAugmentedRKD(queue_size=384, embedding_dim=256)
        >>> for batch in dataloader:
        ...     t_emb = teacher.encoder(batch)  # [B, D]
        ...     s_emb = student.encoder(batch)  # [B, D]
        ...     loss, dist_loss, angle_loss = memory_rkd.compute_loss(t_emb, s_emb)
        ...     loss.backward()  # Gradients only through s_emb
        ...     memory_rkd.push(t_emb, s_emb)  # Update queue after backward

    Args:
        queue_size: Max entries in memory queue (should be < dataset size)
        embedding_dim: Feature dimension from encoder
        sample_size: How many memory samples to use per step
        distance_weight: RKD distance loss weight β
        angle_weight: RKD angle loss weight γ
    """

    def __init__(
        self,
        queue_size: int = 384,
        embedding_dim: int = 256,
        sample_size: int = 72,
        distance_weight: float = 1.0,
        angle_weight: float = 2.0,
    ):
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.sample_size = sample_size
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

        # Circular buffer storage (lazy init on first push)
        self.teacher_queue: torch.Tensor | None = None  # [K, D]
        self.student_queue: torch.Tensor | None = None  # [K, D]
        self.queue_ptr = 0  # Next write position
        self.queue_len = 0  # Current valid entries (0 to queue_size)
        self._device: torch.device | None = None

    @torch.no_grad()
    def push(self, teacher_emb: torch.Tensor, student_emb: torch.Tensor) -> None:
        """Push batch embeddings into queue (FIFO).

        Both embeddings are detached and stored without gradients.
        Call this AFTER backward pass to avoid including current batch in its own loss.

        Args:
            teacher_emb: [B, D] teacher features
            student_emb: [B, D] student features
        """
        batch_size = teacher_emb.size(0)
        embed_dim = teacher_emb.size(1)

        # Lazy initialization
        if self.teacher_queue is None:
            self._device = teacher_emb.device
            self.teacher_queue = torch.zeros(
                self.queue_size, embed_dim, device=self._device
            )
            self.student_queue = torch.zeros(
                self.queue_size, embed_dim, device=self._device
            )

        # Detach embeddings
        teacher_emb = teacher_emb.detach()
        student_emb = student_emb.detach()

        # Handle case where batch is larger than remaining space
        # Write in chunks if needed (circular buffer)
        ptr = self.queue_ptr
        for i in range(batch_size):
            self.teacher_queue[ptr] = teacher_emb[i]
            self.student_queue[ptr] = student_emb[i]
            ptr = (ptr + 1) % self.queue_size

        # Update pointer and length
        self.queue_ptr = ptr
        self.queue_len = min(self.queue_len + batch_size, self.queue_size)

    def sample(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample subset from memory queue.

        Args:
            device: Target device for returned tensors (default: queue device)

        Returns:
            (teacher_samples, student_samples) each [K', D], or None if queue empty
        """
        if self.queue_len == 0 or self.teacher_queue is None:
            return None

        device = device or self._device

        # Sample min(sample_size, queue_len) indices
        n_samples = min(self.sample_size, self.queue_len)
        indices = torch.randperm(self.queue_len, device=self._device)[:n_samples]

        teacher_samples = self.teacher_queue[indices].to(device)
        student_samples = self.student_queue[indices].to(device)

        return teacher_samples, student_samples

    def compute_loss(
        self,
        teacher_batch: torch.Tensor,
        student_batch: torch.Tensor,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute memory-augmented RKD loss.

        Gradients only flow through student_batch (live samples).
        Memory samples are constants (like MoCo keys).

        Args:
            teacher_batch: [B, D] current batch teacher embeddings
            student_batch: [B, D] current batch student embeddings (requires grad)
            eps: Numerical stability

        Returns:
            (total_loss, distance_loss, angle_loss)
        """
        device = student_batch.device
        n_live = student_batch.size(0)

        # Sample from memory (may be None if queue empty)
        memory_samples = self.sample(device)

        if memory_samples is None:
            # No memory yet - fall back to standard RKD on batch only
            return rkd_loss(
                student_batch,
                teacher_batch,
                self.distance_weight,
                self.angle_weight,
                eps,
            )

        teacher_mem, student_mem = memory_samples
        n_mem = teacher_mem.size(0)
        n_total = n_live + n_mem

        # Concatenate: live samples first, then memory
        # Live samples have gradients, memory samples are detached
        teacher_all = torch.cat([teacher_batch, teacher_mem], dim=0)  # [N, D]
        student_all = torch.cat([student_batch, student_mem.detach()], dim=0)  # [N, D]

        # Compute masks for pairs/triplets with at least one live sample
        pair_mask = _compute_live_pair_mask(n_live, n_total, device)
        triplet_mask = _compute_live_triplet_mask(n_live, n_total, device)

        # Compute masked losses
        dist_loss = rkd_distance_loss_masked(student_all, teacher_all, pair_mask, eps)
        angle_loss = rkd_angle_loss_masked(student_all, teacher_all, triplet_mask, eps)

        total_loss = self.distance_weight * dist_loss + self.angle_weight * angle_loss

        return total_loss, dist_loss, angle_loss

    def reset(self) -> None:
        """Clear the queue (e.g., between experiments)."""
        self.teacher_queue = None
        self.student_queue = None
        self.queue_ptr = 0
        self.queue_len = 0
        self._device = None

    def __len__(self) -> int:
        """Return current number of entries in queue."""
        return self.queue_len

    def is_full(self) -> bool:
        """Check if queue has reached capacity."""
        return self.queue_len >= self.queue_size

    def __repr__(self) -> str:
        return (
            f"MemoryAugmentedRKD(queue_size={self.queue_size}, "
            f"embedding_dim={self.embedding_dim}, sample_size={self.sample_size}, "
            f"current_len={self.queue_len}, distance_weight={self.distance_weight}, "
            f"angle_weight={self.angle_weight})"
        )
