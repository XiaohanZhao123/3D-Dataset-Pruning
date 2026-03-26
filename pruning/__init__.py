"""Data pruning module for selecting representative samples from datasets.

Architecture:
- functional.py: Pure selection functions (shared utilities)
- scorers.py: Closure-based scorers (loss, prototype, cluster, herding)
- pruner.py: ScoreBasedDataPruner wrapper class
- utils.py: Utilities for building scoring pipelines
- balanced_scorers.py: OOP-based scorers with registry pattern (recommended)
- baseline/: Baseline methods (DRoP, NUCS)

Usage:
    # Modern OOP approach (recommended)
    from pruning.balanced_scorers import get_scorer, SCORER_REGISTRY
    scorer = get_scorer("loss", model, cfg, device="cuda")
    scores, labels, indices = scorer.compute(dataloader)

    # Closure-based scorers
    from pruning import make_loss_scorer, ScoreBasedDataPruner

    # NUCS budget allocation
    from pruning.baseline import compute_nucs_budgets, NUCSBudgetAllocator
"""

# Core functional utilities (shared)
from pruning.functional import (
    select_by_score,
    _random_selection,
    get_all_samples,
    get_label_indices_map,
    get_label_samples_map,
    compute_class_budgets,
    extract_label,
)

# Closure-based scorers
from pruning.scorers import (
    make_cluster_scorer,
    make_loss_scorer,
    make_prototype_scorer,
    make_herding_scorer,
)

# Pruner wrapper
from pruning.pruner import ScoreBasedDataPruner

# Utilities
from pruning.utils import (
    build_score_fn,
    build_feature_pipe,
    build_logits_pipe,
)

# Modern OOP scorers (recommended)
from pruning.balanced_scorers import (
    BaseScorer,
    SCORER_REGISTRY,
    get_scorer,
    list_scorers,
    register_scorer,
    # Concrete scorers
    LossScorer,
    HerdingScorer,
    KCenterGreedyScorer,
    KMeansPrototypeScorer,
    SubmodularCosineScorer,
    SubmodularRBFScorer,
    EntropyScorer,
    EL2NScorer,
    GradNormScorer,
    GradHerdingScorer,
    # Mixins
    FeatureExtractorMixin,
    ClassNamesMixin,
)

__all__ = [
    # Core functional utilities
    "select_by_score",
    "_random_selection",
    "get_all_samples",
    "get_label_indices_map",
    "get_label_samples_map",
    "compute_class_budgets",
    "extract_label",
    # Closure-based scorers
    "make_cluster_scorer",
    "make_loss_scorer",
    "make_prototype_scorer",
    "make_herding_scorer",
    # Pruner wrapper
    "ScoreBasedDataPruner",
    # Utilities
    "build_score_fn",
    "build_feature_pipe",
    "build_logits_pipe",
    # Modern OOP interface
    "BaseScorer",
    "SCORER_REGISTRY",
    "get_scorer",
    "list_scorers",
    "register_scorer",
    "LossScorer",
    "HerdingScorer",
    "KCenterGreedyScorer",
    "KMeansPrototypeScorer",
    "SubmodularCosineScorer",
    "SubmodularRBFScorer",
    "EntropyScorer",
    "EL2NScorer",
    "GradNormScorer",
    "GradHerdingScorer",
    "FeatureExtractorMixin",
    "ClassNamesMixin",
]
