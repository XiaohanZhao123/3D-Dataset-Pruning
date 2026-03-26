"""DRoP: Distributionally Robust Data Pruning baseline.

Paper: "DRoP: Distributionally Robust Data Pruning" (Vysogorets et al., ICLR 2025)
"""

from pruning.baseline.drop.quota import (
    compute_drop_quota,
    compute_per_class_metrics,
    get_class_sizes,
    select_with_drop_quota,
)

__all__ = [
    "compute_drop_quota",
    "compute_per_class_metrics",
    "get_class_sizes",
    "select_with_drop_quota",
]
