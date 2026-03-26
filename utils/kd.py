"""Knowledge distillation helpers."""

import torch
import torch.nn.functional as F


def distill_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute KL divergence distillation loss."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_loss: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Combine hard loss with distillation loss."""
    distill_loss = distill_kl_loss(student_logits, teacher_logits, temperature)
    return alpha * distill_loss + (1.0 - alpha) * hard_loss
