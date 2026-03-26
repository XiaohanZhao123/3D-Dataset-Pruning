"""Shared visualization utilities for class-wise loss analysis."""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.data_prep import prepare_batch

logger = logging.getLogger(__name__)

# Publication-quality matplotlib settings
PLOT_CONFIG = {
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
}


def apply_plot_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(PLOT_CONFIG)


def compute_class_losses(model, dataloader, cfg, device, num_classes):
    """
    Compute mean loss per class.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for dataset
        cfg: Config with model settings
        device: Device for computation
        num_classes: Number of classes

    Returns:
        Dictionary with class statistics: {class_id: {'mean': float, 'std': float, 'count': int}}
    """
    logger.info("Computing class-wise losses...")

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    model.to(device)

    # Collect losses per class
    class_losses = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing losses"):
            batch, target = prepare_batch(
                batch,
                cfg,
                device,
                resample=False,
                truncate=True,
            )

            # Forward pass
            logits = model(batch)

            # Compute per-sample loss
            losses = loss_fn(logits, target)

            # Group by class
            for loss, label in zip(losses.cpu().numpy(), target.cpu().numpy()):
                class_losses[label].append(loss)

    # Compute statistics per class
    class_stats = {}
    for class_idx in range(num_classes):
        if class_idx in class_losses:
            losses = np.array(class_losses[class_idx])
            class_stats[class_idx] = {
                "mean": float(np.mean(losses)),
                "std": float(np.std(losses)),
                "count": len(losses),
            }
        else:
            logger.warning(f"No samples found for class {class_idx}")
            class_stats[class_idx] = {"mean": 0.0, "std": 0.0, "count": 0}

    logger.info(f"Computed losses for {len(class_stats)} classes")
    return class_stats


def compute_multi_head_class_losses(model, dataloader, cfg, device, num_classes, num_heads):
    """
    Compute mean loss per class for multi-head classifier (averaged across heads).

    Args:
        model: Model with multi-head classifier (ModuleList in model.prediction.head)
        dataloader: DataLoader for dataset
        cfg: Config with model settings
        device: Device for computation
        num_classes: Number of classes
        num_heads: Number of classifier heads

    Returns:
        Tuple of (per_head_stats, averaged_stats):
            - per_head_stats: List[Dict] - statistics for each head individually
            - averaged_stats: Dict - averaged statistics across all heads
    """
    logger.info(f"Computing class-wise losses for {num_heads} heads...")

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    model.to(device)

    # Collect losses per class per head
    per_head_class_losses = [defaultdict(list) for _ in range(num_heads)]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing multi-head losses"):
            batch, target = prepare_batch(
                batch,
                cfg,
                device,
                resample=False,
                truncate=True,
            )

            # Forward through encoder to get global features
            global_feat = model.encoder.forward_cls_feat(batch)

            # Forward through each head
            for head_idx in range(num_heads):
                logits = model.prediction.head[head_idx](global_feat)
                losses = loss_fn(logits, target)

                # Group by class for this head
                for loss, label in zip(losses.cpu().numpy(), target.cpu().numpy()):
                    per_head_class_losses[head_idx][label].append(loss)

    # Compute statistics per head
    per_head_stats = []
    for head_idx in range(num_heads):
        head_stats = {}
        for class_idx in range(num_classes):
            if class_idx in per_head_class_losses[head_idx]:
                losses = np.array(per_head_class_losses[head_idx][class_idx])
                head_stats[class_idx] = {
                    "mean": float(np.mean(losses)),
                    "std": float(np.std(losses)),
                    "count": len(losses),
                }
            else:
                head_stats[class_idx] = {"mean": 0.0, "std": 0.0, "count": 0}
        per_head_stats.append(head_stats)

    # Compute averaged statistics across heads
    averaged_stats = {}
    for class_idx in range(num_classes):
        # Extract means and stds from all heads
        means = [per_head_stats[h][class_idx]["mean"] for h in range(num_heads)]
        stds = [per_head_stats[h][class_idx]["std"] for h in range(num_heads)]
        count = per_head_stats[0][class_idx]["count"]  # All heads see same samples

        averaged_stats[class_idx] = {
            "mean": float(np.mean(means)),  # Average of means
            "std": float(np.mean(stds)),    # Average of within-head stds
            "std_across_heads": float(np.std(means)),  # Variance across heads (initialization sensitivity)
            "count": count,
        }

    # Log variance statistics
    init_variance = np.mean([averaged_stats[i]["std_across_heads"] for i in range(num_classes)])
    logger.info(f"Mean initialization variance across heads: {init_variance:.4f}")
    logger.info(f"Computed losses for {num_classes} classes across {num_heads} heads")

    return per_head_stats, averaged_stats


def plot_single_class_loss(
    class_stats,
    class_names,
    output_path,
    model_name="PointNeXt",
    title_suffix="Training Set"
):
    """
    Create horizontal bar plot of class-wise mean loss (single model).

    Args:
        class_stats: Dictionary of class statistics from compute_class_losses
        class_names: List of class names (ordered by class index)
        output_path: Path to save output figures
        model_name: Name of the model for plot title
        title_suffix: Additional text for title (e.g., "Training Set", "Test Set")
    """
    logger.info("Creating class loss visualization...")

    apply_plot_style()

    num_classes = len(class_names)
    mean_losses = np.array([class_stats[i]["mean"] for i in range(num_classes)])
    std_losses = np.array([class_stats[i]["std"] for i in range(num_classes)])
    sample_counts = np.array([class_stats[i]["count"] for i in range(num_classes)])

    # Sort by mean loss
    sorted_indices = np.argsort(mean_losses)
    classes_sorted = [class_names[i] for i in sorted_indices]
    losses_sorted = mean_losses[sorted_indices]
    stds_sorted = std_losses[sorted_indices]
    counts_sorted = sample_counts[sorted_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 12))

    y_pos = np.arange(len(classes_sorted))
    height = 0.75

    # Color gradient (green to red)
    from matplotlib.colors import LinearSegmentedColormap
    colors_gradient = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#C0392B"]
    cmap = LinearSegmentedColormap.from_list("loss_gradient", colors_gradient, N=len(classes_sorted))
    norm = plt.Normalize(vmin=losses_sorted.min(), vmax=losses_sorted.max())
    colors = cmap(norm(losses_sorted))

    # Horizontal bars
    bars = ax.barh(y_pos, losses_sorted, height, color=colors, edgecolor='white', linewidth=0.8)

    # Add loss values as text
    for i, (bar, loss, std, count) in enumerate(zip(bars, losses_sorted, stds_sorted, counts_sorted)):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{loss:.3f}±{std:.3f} (n={count})",
            ha="left", va="center", fontsize=8,
        )

    # Customize plot
    ax.set_xlabel("Mean Cross-Entropy Loss", fontweight="bold", fontsize=13)
    ax.set_ylabel("Class Name", fontweight="bold", fontsize=13)
    ax.set_title(
        f"Class-Wise Average Loss - {model_name}\n{title_suffix}",
        fontweight="bold", pad=20, fontsize=14,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes_sorted, fontsize=10)
    ax.xaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, losses_sorted.max() * 1.15)

    # Statistics text box
    total_samples = sum(sample_counts)
    overall_mean = np.mean(mean_losses)
    overall_std = np.std(mean_losses)
    min_loss = min(mean_losses)
    max_loss = max(mean_losses)

    stats_text = f"""Statistics:
Total Samples: {total_samples:,}
Overall Mean: {overall_mean:.4f} ± {overall_std:.4f}
Range: [{min_loss:.4f} - {max_loss:.4f}]"""

    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        family="monospace",
    )

    plt.tight_layout()

    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_obj.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(output_path_obj.with_suffix(".png"), format="png", bbox_inches="tight")

    logger.info(f"Plots saved: {output_path_obj.with_suffix('.pdf')}, {output_path_obj.with_suffix('.png')}")
    plt.close()


def plot_class_loss_comparison(
    before_stats,
    after_stats,
    class_names,
    output_path,
    true_class_counts=None,
    model_name="PointNeXt",
    before_label="Before Retraining",
    after_label="After Retraining"
):
    """
    Create three subplots: before, after, and improvement. Color represents TRUE class sample count.

    Args:
        before_stats: Class statistics before retraining
        after_stats: Class statistics after retraining
        class_names: List of class names
        output_path: Path to save output figures
        true_class_counts: TRUE class distribution from dataset (not from sampler)
        model_name: Model name for title
        before_label: Label for before condition
        after_label: Label for after condition
    """
    logger.info("Creating comparison visualization...")

    apply_plot_style()
    from matplotlib.colors import LinearSegmentedColormap

    num_classes = len(class_names)

    # Extract data for both conditions
    before_losses = np.array([before_stats[i]["mean"] for i in range(num_classes)])
    before_stds = np.array([before_stats[i]["std"] for i in range(num_classes)])

    after_losses = np.array([after_stats[i]["mean"] for i in range(num_classes)])
    after_stds = np.array([after_stats[i]["std"] for i in range(num_classes)])

    improvements = before_losses - after_losses  # Positive = improvement

    # Use TRUE class counts (from dataset, not sampler)
    if true_class_counts is None:
        # Fallback to stats counts if not provided
        true_class_counts = np.array([before_stats[i]["count"] for i in range(num_classes)])

    # Independent sorting for each subplot
    before_sorted_indices = np.argsort(before_losses)
    after_sorted_indices = np.argsort(after_losses)
    improve_sorted_indices = np.argsort(improvements)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 12))

    # REVERSED color gradient (RED=few samples, BLUE=many samples)
    colors_gradient = ["#E74C3C", "#E67E22", "#F1C40F", "#1ABC9C", "#3498DB"]
    cmap = LinearSegmentedColormap.from_list("count_gradient", colors_gradient, N=256)

    # Use RANK-based coloring for even color distribution
    # Assign colors based on percentile rank, not absolute values
    from scipy.stats import rankdata

    min_count = true_class_counts.min()
    max_count = true_class_counts.max()

    # Compute percentile ranks (0-100) for each class
    # Higher rank = more samples
    class_ranks = rankdata(true_class_counts, method='average')
    class_percentiles = (class_ranks - 1) / (num_classes - 1) * 100  # 0-100 scale

    logger.info(f"Sample count range: {min_count}-{max_count}")
    logger.info(f"Rank-based percentiles: {class_percentiles.min():.1f}-{class_percentiles.max():.1f}")

    # Normalize to 0-1 for colormap (based on rank, not value!)
    norm = plt.Normalize(vmin=0, vmax=100)

    y_pos = np.arange(num_classes)
    height = 0.75

    # ===== LEFT SUBPLOT: BEFORE =====
    before_classes_sorted = [class_names[i] for i in before_sorted_indices]
    before_losses_sorted = before_losses[before_sorted_indices]
    before_stds_sorted = before_stds[before_sorted_indices]
    before_counts_sorted = true_class_counts[before_sorted_indices]
    before_percentiles_sorted = class_percentiles[before_sorted_indices]

    before_colors = cmap(norm(before_percentiles_sorted))
    bars1 = ax1.barh(y_pos, before_losses_sorted, height, color=before_colors,
                     edgecolor='white', linewidth=0.8)

    # Add loss and count as text
    for i, (bar, loss, std, count) in enumerate(zip(bars1, before_losses_sorted,
                                                     before_stds_sorted, before_counts_sorted)):
        ax1.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{loss:.3f}±{std:.3f} | n={count}",
            ha="left", va="center", fontsize=8,
        )

    ax1.set_xlabel("Mean Cross-Entropy Loss", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Class Name", fontweight="bold", fontsize=13)
    ax1.set_title(f"{before_label}\n{model_name}", fontweight="bold", pad=20, fontsize=14)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(before_classes_sorted, fontsize=10)
    ax1.xaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(0, before_losses_sorted.max() * 1.2)

    # ===== MIDDLE SUBPLOT: AFTER =====
    after_classes_sorted = [class_names[i] for i in after_sorted_indices]
    after_losses_sorted = after_losses[after_sorted_indices]
    after_stds_sorted = after_stds[after_sorted_indices]
    after_counts_sorted = true_class_counts[after_sorted_indices]
    after_percentiles_sorted = class_percentiles[after_sorted_indices]

    after_colors = cmap(norm(after_percentiles_sorted))
    bars2 = ax2.barh(y_pos, after_losses_sorted, height, color=after_colors,
                     edgecolor='white', linewidth=0.8)

    for i, (bar, loss, std, count) in enumerate(zip(bars2, after_losses_sorted,
                                                     after_stds_sorted, after_counts_sorted)):
        ax2.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{loss:.3f}±{std:.3f} | n={count}",
            ha="left", va="center", fontsize=8,
        )

    ax2.set_xlabel("Mean Cross-Entropy Loss", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Class Name", fontweight="bold", fontsize=13)
    ax2.set_title(f"{after_label}\n{model_name}", fontweight="bold", pad=20, fontsize=14)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(after_classes_sorted, fontsize=10)
    ax2.xaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(0, after_losses_sorted.max() * 1.2)

    # ===== RIGHT SUBPLOT: IMPROVEMENT =====
    improve_classes_sorted = [class_names[i] for i in improve_sorted_indices]
    improvements_sorted = improvements[improve_sorted_indices]
    improve_counts_sorted = true_class_counts[improve_sorted_indices]
    improve_percentiles_sorted = class_percentiles[improve_sorted_indices]

    improve_colors = cmap(norm(improve_percentiles_sorted))
    bars3 = ax3.barh(y_pos, improvements_sorted, height, color=improve_colors,
                     edgecolor='white', linewidth=0.8)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    for i, (bar, improve, count) in enumerate(zip(bars3, improvements_sorted, improve_counts_sorted)):
        ax3.text(
            bar.get_width() + 0.005 if improve >= 0 else bar.get_width() - 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{improve:+.3f} | n={count}",
            ha="left" if improve >= 0 else "right",
            va="center", fontsize=8,
        )

    ax3.set_xlabel("Loss Reduction (Before - After)", fontweight="bold", fontsize=13)
    ax3.set_ylabel("Class Name", fontweight="bold", fontsize=13)
    ax3.set_title("Improvement per Class", fontweight="bold", pad=20, fontsize=14)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(improve_classes_sorted, fontsize=10)
    ax3.xaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Adjust layout to leave space for colorbar
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Add colorbar below the plots showing rank-based mapping
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    # Show actual count values at key percentiles
    percentile_positions = [0, 25, 50, 75, 100]
    tick_percentiles = percentile_positions
    tick_counts = [int(np.percentile(true_class_counts, p)) for p in percentile_positions]

    cbar.set_ticks(tick_percentiles)
    cbar.set_ticklabels([f'{c}' for c in tick_counts], fontsize=10)
    cbar.set_label(
        f'Sample Count per Class (Rank-based coloring: {min_count}→{max_count})',
        fontweight='bold', fontsize=12
    )

    # Compute statistics for logging
    total_samples = sum(true_class_counts)
    mean_before = np.mean(before_losses)
    mean_after = np.mean(after_losses)
    mean_improve = mean_before - mean_after
    improve_pct = (mean_improve / mean_before) * 100 if mean_before > 0 else 0
    improved_classes = sum(1 for imp in improvements if imp > 0)
    worse_classes = sum(1 for imp in improvements if imp < 0)

    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_obj.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(output_path_obj.with_suffix(".png"), format="png", bbox_inches="tight", dpi=150)

    logger.info(f"Comparison plots saved: {output_path_obj.with_suffix('.pdf')}, {output_path_obj.with_suffix('.png')}")
    plt.close()

    # Print improvement table
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT METRICS TABLE (sorted by improvement)")
    logger.info("="*80)
    logger.info(f"Overall: {mean_before:.4f} → {mean_after:.4f} = {mean_improve:.4f} improvement ({improve_pct:+.1f}%)")
    logger.info(f"Classes Improved: {improved_classes}/{num_classes} | Worse: {worse_classes}/{num_classes}")
    logger.info("="*80)
    logger.info(f"{'Class':<20} {'Samples':<10} {'Before':<10} {'After':<10} {'Improve':<12} {'Change %':<10}")
    logger.info("-"*80)

    improvement_sorted_indices = np.argsort(improvements)[::-1]
    for i in improvement_sorted_indices:
        class_name = class_names[i]
        count = true_class_counts[i]
        before = before_losses[i]
        after = after_losses[i]
        improve = improvements[i]
        change_pct = (improve / before) * 100 if before > 0 else 0

        logger.info(
            f"{class_name:<20} {count:<10} {before:<10.4f} {after:<10.4f} "
            f"{improve:+<12.4f} {change_pct:+<10.1f}%"
        )

    logger.info("="*80)


def auto_generate_comparison(
    before_ckpt_path,
    after_model,
    train_dataset,
    full_cfg,
    device,
    num_heads=1,
    model_name="PointNeXt"
):
    """Auto-generate before/after comparison visualization.

    Complete pipeline that:
    1. Extracts true class distribution from dataset
    2. Builds evaluation dataloader
    3. Loads before model from checkpoint
    4. Computes losses for both models
    5. Generates comparison plots
    6. Prints summary statistics

    Args:
        before_ckpt_path: Path to checkpoint before retraining
        after_model: Current model after retraining
        train_dataset: Training dataset (for label extraction)
        full_cfg: Full config with openpoint settings
        device: torch device
        num_heads: Number of classifier heads in after_model (default 1)
        model_name: Name for plot title (default "PointNeXt")
    """
    import os
    from collections import Counter
    import numpy as np
    import torch
    from openpoints.dataset import build_dataloader_from_cfg
    from openpoints.models import build_model_from_cfg
    from utils.constants import get_class_names

    logger.info("\n" + "=" * 80)
    logger.info("Generating before/after comparison visualization...")
    logger.info("=" * 80)

    # Get class names
    class_names = get_class_names(full_cfg.openpoint.num_classes)

    # Get TRUE class distribution from dataset (not from balanced sampler!)
    logger.info("Extracting true class distribution from dataset...")
    if hasattr(train_dataset, "label"):
        dataset_labels = train_dataset.label
    elif hasattr(train_dataset, "labels"):
        dataset_labels = train_dataset.labels
    elif hasattr(train_dataset, "targets"):
        dataset_labels = train_dataset.targets
    else:
        raise ValueError("Cannot find labels in dataset")

    label_counts = Counter(dataset_labels)
    true_class_counts = np.array(
        [label_counts[i] for i in range(full_cfg.openpoint.num_classes)]
    )
    logger.info(
        f"True class distribution: min={true_class_counts.min()}, max={true_class_counts.max()}"
    )

    # Build standard (non-balanced) dataloader for evaluation
    logger.info("Building standard dataloader for evaluation...")
    eval_loader = build_dataloader_from_cfg(
        full_cfg.openpoint.batch_size,
        full_cfg.openpoint.dataset,
        full_cfg.openpoint.dataloader,
        datatransforms_cfg=full_cfg.openpoint.datatransforms,
        split="train",
        distributed=False,
    )

    # Load before model (reload fresh to avoid multi-head structure conflict)
    logger.info("Computing BEFORE losses...")
    before_checkpoint = torch.load(
        before_ckpt_path, map_location=device, weights_only=False
    )

    # Build model from saved config
    before_model = build_model_from_cfg(before_checkpoint["config"]["model"])
    before_model.load_state_dict(before_checkpoint["model"])
    before_model.to(device)
    before_model.eval()

    before_stats = compute_class_losses(
        before_model,
        eval_loader,
        full_cfg.openpoint,
        device,
        full_cfg.openpoint.num_classes,
    )

    # Compute after losses (current model)
    if num_heads > 1:
        logger.info(f"Computing AFTER losses ({num_heads} heads)...")
        per_head_stats, after_stats = compute_multi_head_class_losses(
            after_model,
            eval_loader,
            full_cfg.openpoint,
            device,
            full_cfg.openpoint.num_classes,
            num_heads,
        )

        # Report initialization variance across heads
        variance_across_heads = [
            after_stats[i]["std_across_heads"]
            for i in range(full_cfg.openpoint.num_classes)
        ]
        logger.info(f"Initialization variance across {num_heads} heads:")
        logger.info(f"  Mean: {np.mean(variance_across_heads):.4f}")
        logger.info(f"  Std:  {np.std(variance_across_heads):.4f}")
        logger.info(f"  Max:  {np.max(variance_across_heads):.4f}")
    else:
        logger.info("Computing AFTER losses...")
        after_stats = compute_class_losses(
            after_model,
            eval_loader,
            full_cfg.openpoint,
            device,
            full_cfg.openpoint.num_classes,
        )

    # Generate comparison plot
    output_path = os.path.join(
        full_cfg.openpoint.ckpt_dir, "class_loss_comparison.pdf"
    )
    after_label = (
        f"After Retraining (avg of {num_heads} heads)"
        if num_heads > 1
        else "After Retraining"
    )
    plot_class_loss_comparison(
        before_stats,
        after_stats,
        class_names,
        output_path,
        true_class_counts=true_class_counts,
        model_name=model_name,
        before_label="Before Retraining",
        after_label=after_label,
    )

    logger.info(f"Comparison visualization saved: {output_path}")

    # Print summary statistics
    logger.info("=" * 80)
    logger.info("LOSS REDUCTION SUMMARY")
    logger.info("=" * 80)

    improvements = []
    for class_idx in range(full_cfg.openpoint.num_classes):
        before_loss = before_stats[class_idx]["mean"]
        after_loss = after_stats[class_idx]["mean"]
        reduction = (before_loss - after_loss) / before_loss * 100
        improvements.append(reduction)
        logger.info(
            f"Class {class_idx:2d} ({class_names[class_idx]:15s}): {before_loss:.4f} → {after_loss:.4f} ({reduction:+.1f}%)"
        )

    logger.info(f"\nOverall average reduction: {np.mean(improvements):.2f}%")
    logger.info(
        f"Classes with >10% reduction: {sum(1 for r in improvements if r > 10)}/{full_cfg.openpoint.num_classes}"
    )
    logger.info(
        f"Classes with <0% (worse): {sum(1 for r in improvements if r < 0)}/{full_cfg.openpoint.num_classes}"
    )
    logger.info("=" * 80)
