"""Publication-quality plotting for Proof-of-Time benchmark analysis.

Design goals for ACL submission:
- Professional typography suitable for conference papers
- Colorblind-accessible palettes that work in grayscale
- Figure sizes optimized for single/double column ACL format
- Clear, information-dense visualizations
- Consistent styling across all figures
- No titles (use captions instead)
- Legends at top with bold text
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import colorsys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Try to import seaborn, use matplotlib fallback if unavailable
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ seaborn not available, using matplotlib fallback")

# ---------------------------------------------------------------------------
# ACL Publication Style Configuration
# ---------------------------------------------------------------------------

# Toggle to include/exclude titles in plots (default: False for publication)
INCLUDE_TITLES = False

# Toggle to include/exclude axis labels (default: True for clarity)
INCLUDE_AXIS_LABELS = True

# Toggle to exclude incomplete tasks (all samples hit limit) from simple_vs_agentic plot
EXCLUDE_INCOMPLETE_TASKS = False

# High DPI for publication quality
DEFAULT_DPI = 300

# ACL column widths: single column ≈ 3.3", double column ≈ 7"
FIGURE_SIZE_SINGLE = (3.3, 2.8)   # Single column figure
FIGURE_SIZE_DOUBLE = (7.0, 4.0)   # Double column figure
FIGURE_SIZE_TALL = (7.0, 6.0)     # Double column, tall
FIGURE_SIZE_STANDARD = (8.0, 5.0) # Standard presentation size
FIGURE_SIZE_WIDE = (10.0, 5.0)    # Wide format

# Professional, colorblind-friendly palette
# Based on Paul Tol's qualitative palette
FAMILY_COLORS: Dict[str, str] = {
    "Anthropic": "#EE7733",  # Orange - warm
    "OpenAI": "#0077BB",     # Blue - cool
    "Google": "#009988",     # Teal - neutral
    "Other": "#BBBBBB",      # Gray - muted
}

VARIANT_COLORS: Dict[str, str] = {
    "offline_prompt": "#4477AA",      # Blue
    "no_offline_prompt": "#EE6677",   # Red
    "simple_task": "#CCBB44",         # Yellow
}

# Color palette for task families (colorblind-safe)
cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C0","C1","C2","C3"])
TASK_FAMILY_COLORS = [cycle[0], cycle[3], cycle[2], cycle[1], cycle[1]]
# TASK_FAMILY_COLORS = [
#     "#4477AA",  # Blue
#     "#EE6677",  # Red
#     "#228833",  # Green
#     "#CCBB44",  # Yellow
#     "#66CCEE",  # Cyan
#     "#AA3377",  # Purple
#     "#BBBBBB",  # Gray
# ]

# Stratum colors for agent behavior analysis (using Paul Tol palette for consistency)
STRATUM_COLORS = {
    'correct': '#228833',      # Green (from Paul Tol palette) - success/correct
    'wrong': '#EE6677',        # Red (from Paul Tol palette) - error/wrong
    'incomplete': '#4477AA',   # Blue (from Paul Tol palette) - incomplete/in-progress
}

# UI colors
UI_COLORS = {
    'bg': '#FAFAFA',           # Off-white background
    'text': '#2B2D42',         # Dark slate text
    'grid': '#E5E5E5',         # Subtle grid
}

# Task family display order for consistency
TASK_FAMILY_ORDER = [
    "citation",
    "emnlp_awards",
    "emnlp_awards_acl2025",
    "emnlp_awards_emnlp2025",
    "emnlp_historical",
    "faculty",
    "sota",
]

# Professional font configuration
plt.rcParams.update({
    # Figure settings
    "figure.figsize": FIGURE_SIZE_STANDARD,
    "figure.dpi": DEFAULT_DPI,
    "figure.facecolor": "white",
    "figure.edgecolor": "white",

    # Font settings - use serif for publication
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",  # Math font compatible with Times
    "font.size": 10,

    # Axis settings
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Tick settings
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    # Legend settings - bold text, at top
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",

    # Grid settings
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.color": "#CCCCCC",

    # Line settings
    "lines.linewidth": 1.5,
    "lines.markersize": 6,

    # Save settings
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Set seaborn theme if available
if HAS_SEABORN:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        rc={
            "figure.figsize": FIGURE_SIZE_STANDARD,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        },
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _pretty_task_name(task: str) -> str:
    """Convert task name to human-readable format for figures."""
    t = str(task).replace("_", " ").strip()

    # Specific replacements for clarity
    replacements = {
        "mcq": "MCQ",
        "sota": "SOTA",
        "emnlp": "EMNLP",
        "acl2025": "ACL'25",
        "emnlp2025": "EMNLP'25",
        # Combined task names (new)
        "emnlp awards historical combined": "Awards (Historical)",
        "emnlp awards post cutoff combined": "Awards (Post-Cutoff)",
        # Legacy task names
        "citation multiple choice": "Citations MCQ",
        "citation bucket prediction": "Citations Bucket",
        "citation ranking": "Citations Rank",
        "emnlp awards mcq": "Awards MCQ",
        "emnlp awards mcq acl2025": "ACL'25 Award",
        "emnlp awards mcq emnlp2025": "EMNLP'25 Award",
        "emnlp historical mcq": "Historical MCQ",
        "faculty professor article": "Prof. Article",
        "faculty professor field": "Prof. Field",
        "faculty field focus": "Field Focus",
        "sota bucket": "SOTA Bucket",
    }

    t_lower = t.lower()
    for old, new in replacements.items():
        if t_lower == old:
            return new
        t = t.replace(old, new)

    return t.title()


def _pretty_variant_name(variant: str) -> str:
    """Convert variant name to human-readable format."""
    mapping = {
        "offline_prompt": "Agent + Prompt",
        "no_offline_prompt": "Agent Only",
        "simple_task": "Zero-shot",
    }
    return mapping.get(variant, variant)


def _pretty_model_name(model: str) -> str:
    """Convert model names to standardized display format.

    Maps internal model names to publication-ready display names.
    """
    # Standardized model name mapping
    mapping = {
        # Anthropic models
        "claude-opus-4-5": "Claude Opus 4.5",
        "claude-sonnet-4-5": "Claude Sonnet 4.5",
        "claude-haiku-4-5": "Claude Haiku 4.5",

        # Google models
        "gemini-3-pro-preview": "Gemini 3 Pro Preview",
        "gemini-3-flash-preview": "Gemini 3 Flash Preview",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "gemini-2.5-flash": "Gemini 2.5 Flash",

        # OpenAI models
        "gpt-5.2-2025-12-11": "GPT-5.2",
        "gpt-5.1-2025-11-13": "GPT-5.1",
        "gpt-5-mini-2025-08-07": "GPT-5 Mini",
        "gpt-5-nano-2025-08-07": "GPT-5 Nano",
    }

    return mapping.get(model, model)


def _weighted_mean_accuracy(group: pd.DataFrame) -> float:
    """Compute accuracy weighted by total_samples."""
    if "total_samples" not in group.columns:
        return float(group["accuracy"].mean())
    denom = float(group["total_samples"].sum())
    if denom <= 0:
        return float(group["accuracy"].mean())
    return float((group["accuracy"] * group["total_samples"]).sum() / denom)


def get_gradient_colors(base_hex: str, n: int, lightness_range: Tuple[float, float] = (0.35, 1.0)) -> List[str]:
    """Generate gradient from light to dark based on a base color."""
    # Convert hex to RGB then to HLS
    base_rgb = tuple(int(base_hex.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(*base_rgb)

    colors = []
    for i, intensity in enumerate(np.linspace(lightness_range[0], lightness_range[1], n)):
        # Adjust lightness while keeping hue and saturation
        new_l = l * intensity + (1 - intensity) * 0.92
        new_s = s * (0.6 + 0.4 * intensity)  # Slightly desaturate lighter colors
        r, g, b = colorsys.hls_to_rgb(h, new_l, new_s)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return colors


def save_plot(fig: plt.Figure, filename: str, output_dir: Path) -> None:
    """Save plot to output directory (PNG + PDF for publication)."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename

    fig.tight_layout()

    # Save PNG for quick viewing
    fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")

    # Save PDF for publication (vector format)
    if filename.lower().endswith(".png"):
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
            format="pdf"
        )

    plt.close(fig)
    print(f"  → Saved: {path.name} (+ PDF)")


def style_axis(ax, xlabel=None, ylabel=None):
    """Apply consistent styling to axis."""
    ax.set_facecolor('white')
    if xlabel and INCLUDE_AXIS_LABELS:
        ax.set_xlabel(xlabel, color=UI_COLORS['text'])
    if ylabel and INCLUDE_AXIS_LABELS:
        ax.set_ylabel(ylabel, color=UI_COLORS['text'])
    ax.tick_params(colors=UI_COLORS['text'], length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_color(UI_COLORS['text'])
        spine.set_alpha(0.7)


# ---------------------------------------------------------------------------
# Core plots - Benchmark Performance Analysis
# ---------------------------------------------------------------------------

def plot_overall_performance(df: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart: overall accuracy by model (sample-weighted)."""
    print("📊 Generating overall performance plots...")

    for limit in [30, 50]:
        subset = df[(df["message_limit"] == limit) & (df["variant"] == "offline_prompt")]
        if subset.empty:
            continue

        # Compute weighted mean accuracy per model
        avg_perf = (
            subset.groupby(["short_model", "model_family"], as_index=False)
            .apply(_weighted_mean_accuracy)
            .rename(columns={None: "accuracy"})
            .sort_values("accuracy", ascending=True)
            .reset_index(drop=True)
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        # Color bars by model family
        colors = [FAMILY_COLORS.get(fam, "#999999") for fam in avg_perf["model_family"]]

        y_pos = range(len(avg_perf))
        bars = ax.barh(
            y=y_pos,
            width=avg_perf["accuracy"],
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            height=0.7,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([_pretty_model_name(m) for m in avg_perf["short_model"]])

        if INCLUDE_AXIS_LABELS:
            ax.set_xlabel("Accuracy (sample-weighted)")
        ax.set_xlim(0.0, 1.0)

        # Add value labels
        for bar, acc in zip(bars, avg_perf["accuracy"]):
            ax.text(
                acc + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}",
                va="center", ha="left", fontsize=8
            )

        # Add legend for model families - at top, bold text
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", label=family)
            for family, color in FAMILY_COLORS.items()
            if family in avg_perf["model_family"].values
        ]
        legend = ax.legend(handles=handles, title="Provider", loc="upper center",
                          bbox_to_anchor=(0.5, 1.15), ncol=len(handles), fontsize=9, frameon=False)
        plt.setp(legend.get_texts(), fontweight='bold')

        # Add random baseline
        ax.axvline(x=0.25, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        # Position label at top of chart, slightly right of the line, below legend
        ax.text(0.27, len(avg_perf) - 0.2, "Random (4-way)", fontsize=7, alpha=0.6, ha='left', va='top')

        save_plot(fig, f"overall_performance_msg{limit}.png", output_dir)


def plot_task_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: model × task accuracy."""
    print("📊 Generating task breakdown heatmap...")

    # Use combined evaluation dataframe for award tasks
    from data_loader import get_combined_evaluation_df
    df_combined = get_combined_evaluation_df(df)

    for limit in [50, 30]:
        subset = df_combined[(df_combined["message_limit"] == limit) & (df_combined["variant"] == "offline_prompt")]
        if not subset.empty:
            break
    else:
        print("  ⚠️ No data for heatmap")
        return

    subset = subset.copy()

    # For award tasks, use combined family names; for others, use individual task names
    def get_task_label(row):
        if row["task_family"] == "emnlp_awards":
            return "Awards Pre-cutoff"
        elif row["task_family"] == "emnlp_awards_post_cutoff":
            return "Awards Post-cutoff"
        else:
            # Use individual task names for citation, faculty, SOTA
            return _pretty_task_name(row["base_task"])

    # Filter to main task families only (exclude 'emnlp' historical MCQ)
    main_families = {"citation", "faculty", "sota", "emnlp_awards", "emnlp_awards_post_cutoff"}
    subset = subset[subset["task_family"].isin(main_families)].copy()
    subset["task_pretty"] = subset.apply(get_task_label, axis=1)

    # Create pivot table
    pivot = subset.pivot_table(
        index="short_model",
        columns="task_pretty",
        values="accuracy",
        aggfunc="mean"
    )

    # Apply pretty model names to index
    pivot.index = pivot.index.map(_pretty_model_name)

    # Sort by average performance
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.mean(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 6))

    if HAS_SEABORN:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Accuracy", "shrink": 0.8},
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 8},
        )
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.3 or val > 0.7 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           color=color, fontsize=7)

        plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)

    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Task")
        ax.set_ylabel("Model")
    plt.xticks(rotation=45, ha="right")

    save_plot(fig, f"model_task_heatmap_msg{limit}.png", output_dir)


def plot_ablation_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: ReAct+Prompt vs ReAct-only (ablation study)."""
    print("📊 Generating ablation scatter plots...")

    for target_limit in [30, 50]:
        subset = df[df["message_limit"] == target_limit].copy()
        if subset.empty:
            continue

        pivot = (
            subset.pivot_table(
                index=["short_model", "base_task"],
                columns="variant",
                values="accuracy",
                aggfunc="mean",
            )
            .reset_index()
        )

        if "offline_prompt" not in pivot.columns or "no_offline_prompt" not in pivot.columns:
            continue

        pivot["model_family"] = pivot["short_model"].apply(
            lambda x: "Anthropic" if "claude" in x.lower()
            else "OpenAI" if "gpt" in x.lower()
            else "Google" if "gemini" in x.lower()
            else "Other"
        )

        fig, ax = plt.subplots(figsize=(5.5, 5))

        for family, color in FAMILY_COLORS.items():
            mask = pivot["model_family"] == family
            if mask.any():
                ax.scatter(
                    pivot.loc[mask, "no_offline_prompt"],
                    pivot.loc[mask, "offline_prompt"],
                    c=color,
                    s=50,
                    alpha=0.7,
                    label=family,
                    edgecolors="white",
                    linewidth=0.5,
                )

        # Diagonal line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, zorder=0)

        # Shade regions
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="green", zorder=0)
        ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", zorder=0)

        # Labels
        ax.text(0.15, 0.85, "Prompt\nHelps", fontsize=9, alpha=0.5, ha="center", style="italic")
        ax.text(0.85, 0.15, "Plain Agent\nBetter", fontsize=9, alpha=0.5, ha="center", style="italic")

        if INCLUDE_AXIS_LABELS:
            ax.set_xlabel("Accuracy — Agent Only")
            ax.set_ylabel("Accuracy — Agent + Structured Prompt")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")

        legend = ax.legend(title="Provider", loc="upper center", bbox_to_anchor=(0.5, 1.12),
                          ncol=len(FAMILY_COLORS), fontsize=8, frameon=False)
        plt.setp(legend.get_texts(), fontweight='bold')

        save_plot(fig, f"ablation_scatter_msg{target_limit}.png", output_dir)


def plot_scaling_lines(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: test-time scaling curves by model."""
    print("📊 Generating scaling line plots...")

    subset = df[df["variant"] == "offline_prompt"].copy()
    if subset.empty:
        return

    agg = (
        subset.groupby(["short_model", "model_family", "message_limit"], as_index=False)
        .apply(_weighted_mean_accuracy)
        .rename(columns={None: "accuracy"})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('white')

    # Define markers for each model within a family (different markers)
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

    # Define line styles for each family (same line style within family)
    family_linestyles = {
        "Anthropic": '-',      # Solid
        "Google": '-',        # Dashed
        "OpenAI": '--',        # Dash-dot
    }

    # Group models by family for cleaner visualization
    for family in ["Anthropic", "Google", "OpenAI"]:
        family_models = agg[agg["model_family"] == family]["short_model"].unique()
        base_color = FAMILY_COLORS.get(family, "#999999")
        linestyle = family_linestyles.get(family, '-')

        for i, model in enumerate(sorted(family_models)):
            model_data = agg[agg["short_model"] == model].sort_values("message_limit")

            # Use different markers for models within the same family
            marker = markers[i % len(markers)]

            ax.plot(
                model_data["message_limit"],
                model_data["accuracy"],
                marker=marker,
                linestyle=linestyle,
                linewidth=0.5,
                markersize=8,
                markeredgewidth=0.2,
                markeredgecolor='white',
                label=_pretty_model_name(model),
                color=base_color,
                alpha=0.85,
                zorder=3,
            )

    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Message Limit", fontsize=11, fontweight='medium')
        ax.set_ylabel("Accuracy (sample-weighted)", fontsize=11, fontweight='medium')
    ax.set_xticks(sorted(agg["message_limit"].unique()))
    ax.set_ylim(0.0, 0.85)
    ax.tick_params(axis='both', which='major', labelsize=10)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                      ncol=4, fontsize=8, frameon=False, title="Model",
                      columnspacing=1.2, handlelength=2.5)
    plt.setp(legend.get_texts(), fontweight='medium')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=9)

    # Grid styling
    ax.grid(True, alpha=0.25, linewidth=0.8, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # Spines styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

    save_plot(fig, "scaling_by_model.png", output_dir)


def plot_scaling_by_family(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: test-time scaling by task family."""
    print("📊 Generating task family scaling plot...")

    # Use combined evaluation dataframe for award tasks
    from data_loader import get_combined_evaluation_df
    df_combined = get_combined_evaluation_df(df)

    subset = df_combined[df_combined["variant"] == "offline_prompt"].copy()
    if subset.empty:
        return

    agg = (
        subset.groupby(["task_family", "message_limit"], as_index=False)
        .apply(_weighted_mean_accuracy)
        .rename(columns={None: "accuracy"})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('white')

    # Define the 5 task families we want to show
    family_mapping = {
        "citation": "Citations",
        "faculty": "Faculty",
        "sota": "SOTA",
        "emnlp_awards": "Awards Pre-cutoff",  # Historical awards
        "emnlp_awards_post_cutoff": "Awards Post-cutoff",  # ACL'25 + EMNLP'25
    }

    # Define distinct markers for each task family
    family_markers = ['o', 's', '^', 'D', 'v']

    # Filter to only these families and order them
    families = [f for f in family_mapping.keys() if f in agg["task_family"].unique()]

    for i, family in enumerate(families):
        family_data = agg[agg["task_family"] == family].sort_values("message_limit")
        color = TASK_FAMILY_COLORS[i % len(TASK_FAMILY_COLORS)]
        marker = family_markers[i % len(family_markers)]
        pretty_name = family_mapping[family]

        ax.plot(
            family_data["message_limit"],
            family_data["accuracy"],
            marker=marker,
            linewidth=0.5,
            markersize=10,
            markeredgewidth=0.2,
            markeredgecolor='white',
            label=pretty_name,
            color=color,
            alpha=0.9,
            zorder=3,
        )

    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Message Limit", fontsize=11, fontweight='medium')
        ax.set_ylabel("Accuracy (sample-weighted)", fontsize=11, fontweight='medium')
    ax.set_xticks(sorted(agg["message_limit"].unique()))
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis='both', which='major', labelsize=10)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                      ncol=3, fontsize=9, frameon=False, title="Task Family",
                      columnspacing=1.5, handlelength=2.5)
    plt.setp(legend.get_texts(), fontweight='medium')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=10)

    # Grid styling
    ax.grid(True, alpha=0.25, linewidth=0.8, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # Spines styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

    save_plot(fig, "scaling_by_family.png", output_dir)


def plot_model_family_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: model family comparison across message limits."""
    print("📊 Generating model family comparison plot...")

    subset = df[df["variant"] == "offline_prompt"].copy()
    if subset.empty:
        return

    agg = (
        subset.groupby(["model_family", "message_limit"], as_index=False)
        .apply(_weighted_mean_accuracy)
        .rename(columns={None: "accuracy"})
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    families = sorted(agg["model_family"].unique())
    limits = sorted(agg["message_limit"].unique())
    x = np.arange(len(families))
    width = 0.25

    # Color gradient for message limits
    colors = ["#9ECAE1", "#4292C6", "#084594"]  # Light to dark blue

    for i, limit in enumerate(limits):
        limit_data = agg[agg["message_limit"] == limit]
        heights = [
            limit_data[limit_data["model_family"] == fam]["accuracy"].values[0]
            if fam in limit_data["model_family"].values else 0
            for fam in families
        ]
        bars = ax.bar(
            x + i * width - width,
            heights,
            width,
            label=f"Limit {limit}",
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
        )

        # Add value labels
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=7
                )

    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Model Provider")
        ax.set_ylabel("Accuracy (sample-weighted)")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylim(0, 0.75)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                      ncol=len(limits), fontsize=8, frameon=False, title="Limit")
    plt.setp(legend.get_texts(), fontweight='bold')
    ax.grid(True, alpha=0.3, axis="y")

    save_plot(fig, "model_family_comparison.png", output_dir)


def plot_hit_limit_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: hit-limit rate by model at each message limit."""
    print("📊 Generating hit-limit analysis plot...")

    subset = df[df["variant"] == "offline_prompt"].copy()
    if subset.empty:
        return

    agg = (
        subset.groupby(["short_model", "model_family", "message_limit"], as_index=False)
        .agg({"hit_limit_rate": "mean"})
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    for ax, limit in zip(axes, [15, 30, 50]):
        limit_data = agg[agg["message_limit"] == limit].sort_values("hit_limit_rate", ascending=True)
        if limit_data.empty:
            continue

        colors = [FAMILY_COLORS.get(fam, "#999999") for fam in limit_data["model_family"]]

        ax.barh(
            range(len(limit_data)),
            limit_data["hit_limit_rate"],
            color=colors,
            edgecolor="white",
            height=0.7,
        )
        ax.set_yticks(range(len(limit_data)))
        ax.set_yticklabels([_pretty_model_name(m) for m in limit_data["short_model"]], fontsize=8)
        if INCLUDE_AXIS_LABELS:
            ax.set_xlabel("Hit Limit Rate")
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color="#CC0000", linestyle="--", alpha=0.5, linewidth=1)

    if INCLUDE_AXIS_LABELS:
        axes[0].set_ylabel("Model")

    save_plot(fig, "hit_limit_analysis.png", output_dir)


def plot_simple_vs_agentic(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: zero-shot vs agentic performance comparison."""
    print("📊 Generating zero-shot vs agentic comparison plot...")

    subset = df[df["message_limit"] == 50].copy()
    if subset.empty:
        subset = df[df["message_limit"] == 30].copy()
    if subset.empty:
        return

    # Filter out tasks where all samples hit the limit (incomplete tasks)
    if EXCLUDE_INCOMPLETE_TASKS:
        # For agentic tasks (offline_prompt variant), exclude where total_samples == samples_hit_limit
        agentic_subset = subset[subset["variant"] == "offline_prompt"].copy()
        if "total_samples" in agentic_subset.columns and "samples_hit_limit" in agentic_subset.columns:
            # Mark incomplete tasks
            agentic_subset["all_hit_limit"] = (
                agentic_subset["total_samples"] == agentic_subset["samples_hit_limit"]
            ) & (agentic_subset["total_samples"] > 0)

            # Get tasks to exclude
            incomplete_tasks = agentic_subset[agentic_subset["all_hit_limit"]]["base_task"].unique()

            if len(incomplete_tasks) > 0:
                print(f"  ℹ️ Excluding {len(incomplete_tasks)} incomplete tasks (all samples hit limit)")
                subset = subset[~subset["base_task"].isin(incomplete_tasks)]

    pivot = (
        subset.pivot_table(
            index=["short_model", "base_task"],
            columns="variant",
            values="accuracy",
            aggfunc="mean",
        )
        .reset_index()
    )

    if "offline_prompt" not in pivot.columns or "simple_task" not in pivot.columns:
        print("  ⚠️ Missing variants for zero-shot vs agentic comparison")
        return

    pivot["model_family"] = pivot["short_model"].apply(
        lambda x: "Anthropic" if "claude" in x.lower()
        else "OpenAI" if "gpt" in x.lower()
        else "Google" if "gemini" in x.lower()
        else "Other"
    )

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for family, color in FAMILY_COLORS.items():
        mask = pivot["model_family"] == family
        if mask.any():
            ax.scatter(
                pivot.loc[mask, "simple_task"],
                pivot.loc[mask, "offline_prompt"],
                c=color,
                s=50,
                alpha=0.7,
                label=family,
                edgecolors="white",
                linewidth=0.5,
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, zorder=0)

    # Shade regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="green", zorder=0)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", zorder=0)

    ax.text(0.15, 0.85, "Agentic\nBetter", fontsize=9, alpha=0.5, ha="center", style="italic")
    ax.text(0.85, 0.15, "Zero-shot\nBetter", fontsize=9, alpha=0.5, ha="center", style="italic")

    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Accuracy — Zero-shot")
        ax.set_ylabel("Accuracy — Agentic")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                      ncol=len(FAMILY_COLORS), fontsize=8, frameon=False, title="Provider")
    plt.setp(legend.get_texts(), fontweight='bold')

    save_plot(fig, "simple_vs_agentic.png", output_dir)


def plot_scaling_gain_waterfall(df: pd.DataFrame, output_dir: Path) -> None:
    """Waterfall chart showing scaling gains from 15 to 50 messages."""
    print("📊 Generating scaling gain waterfall...")

    subset = df[df["variant"] == "offline_prompt"]
    pivot = subset.pivot_table(
        index=["model", "short_model", "model_family"],
        columns="message_limit",
        values="accuracy",
        aggfunc="mean"
    ).reset_index()

    if 15 not in pivot.columns or 50 not in pivot.columns:
        return

    pivot["gain"] = pivot[50] - pivot[15]
    pivot = pivot.dropna(subset=["gain"]).sort_values("gain", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = [FAMILY_COLORS.get(fam, "#999999") for fam in pivot["model_family"]]

    bars = ax.barh(
        range(len(pivot)),
        pivot["gain"],
        color=colors,
        edgecolor="white",
        height=0.7,
    )

    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([_pretty_model_name(m) for m in pivot["short_model"]], fontsize=9)
    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Accuracy Gain (Acc@50 − Acc@15)")
    ax.axvline(x=0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (gain, _) in enumerate(zip(pivot["gain"], pivot["short_model"])):
        offset = 0.01 if gain >= 0 else -0.01
        ha = "left" if gain >= 0 else "right"
        ax.text(gain + offset, i, f"{gain:+.0%}", va="center", ha=ha, fontsize=8)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", label=family)
        for family, color in FAMILY_COLORS.items()
        if family in pivot["model_family"].values
    ]
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                      ncol=len(handles), fontsize=8, frameon=False, title="Provider")
    plt.setp(legend.get_texts(), fontweight='bold')

    save_plot(fig, "scaling_gain_waterfall.png", output_dir)


def plot_task_difficulty_ranking(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart showing task difficulty (inverse of average accuracy)."""
    print("📊 Generating task difficulty ranking...")

    # Use combined evaluation dataframe for award tasks
    from data_loader import get_combined_evaluation_df
    df_combined = get_combined_evaluation_df(df)

    subset = df_combined[(df_combined["variant"] == "offline_prompt") & (df_combined["message_limit"] == 50)]
    if subset.empty:
        subset = df_combined[(df_combined["variant"] == "offline_prompt") & (df_combined["message_limit"] == 30)]
    if subset.empty:
        return

    # Filter to main task families only (exclude 'emnlp' historical MCQ)
    main_families = {"citation", "faculty", "sota", "emnlp_awards", "emnlp_awards_post_cutoff"}
    subset = subset[subset["task_family"].isin(main_families)].copy()

    # For award tasks, use combined family names; for others, use individual task names
    def get_task_label(row):
        if row["task_family"] == "emnlp_awards":
            return "Awards Pre-cutoff"
        elif row["task_family"] == "emnlp_awards_post_cutoff":
            return "Awards Post-cutoff"
        else:
            # Use individual task names for citation, faculty, SOTA
            return _pretty_task_name(row["base_task"])

    subset["task_label"] = subset.apply(get_task_label, axis=1)

    # Group by task_label and calculate mean accuracy
    task_acc = (
        subset.groupby("task_label", as_index=False)
        .agg({"accuracy": "mean"})
        .sort_values("accuracy", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    # Color by difficulty tier
    colors = [
        "#D73027" if acc < 0.25 else  # Hard (red)
        "#FC8D59" if acc < 0.40 else  # Medium-hard (orange)
        "#FEE08B" if acc < 0.55 else  # Medium (yellow)
        "#91CF60" if acc < 0.70 else  # Medium-easy (light green)
        "#1A9850"                       # Easy (green)
        for acc in task_acc["accuracy"]
    ]

    ax.barh(
        range(len(task_acc)),
        task_acc["accuracy"],
        color=colors,
        edgecolor="white",
        height=0.7,
    )

    ax.set_yticks(range(len(task_acc)))
    ax.set_yticklabels(task_acc["task_label"].tolist(), fontsize=9)
    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Average Accuracy Across Models")
    ax.set_xlim(0, 1)

    # Random baseline
    ax.axvline(x=0.25, color="#666666", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(0.26, len(task_acc) - 0.5, "Random\n(4-way)", fontsize=7, alpha=0.6)

    ax.grid(True, alpha=0.3, axis="x")

    save_plot(fig, "task_difficulty_ranking.png", output_dir)


def plot_post_cutoff_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart comparing historical vs post-cutoff performance."""
    print("📊 Generating post-cutoff comparison plot...")

    subset = df[(df["message_limit"] == 50) & (df["variant"] == "offline_prompt")]
    if subset.empty:
        return

    # Filter to award-related tasks
    award_tasks = subset[subset["task_family"].str.contains("emnlp_awards", na=False)]
    if award_tasks.empty:
        print("  ⚠️ No award task data for post-cutoff comparison")
        return

    # Aggregate by model and task family
    agg = award_tasks.groupby(["short_model", "task_family"])["accuracy"].mean().reset_index()

    # Pivot for comparison
    pivot = agg.pivot(index="short_model", columns="task_family", values="accuracy")

    # Only proceed if we have both historical and post-cutoff
    if "emnlp_awards" not in pivot.columns:
        return

    has_acl = "emnlp_awards_acl2025" in pivot.columns
    has_emnlp = "emnlp_awards_emnlp2025" in pivot.columns

    if not has_acl and not has_emnlp:
        return

    # Sort by historical performance
    pivot = pivot.sort_values("emnlp_awards", ascending=True)

    # Apply pretty model names to index
    pivot.index = pivot.index.map(_pretty_model_name)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(pivot))
    width = 0.25

    # Historical
    ax.barh(x, pivot["emnlp_awards"], width, label="Historical (2021-24)",
            color="#4477AA", edgecolor="white")

    # ACL 2025
    if has_acl:
        ax.barh(x + width, pivot["emnlp_awards_acl2025"], width, label="ACL 2025",
                color="#EE6677", edgecolor="white")

    # EMNLP 2025
    if has_emnlp:
        offset = 2 * width if has_acl else width
        ax.barh(x + offset, pivot["emnlp_awards_emnlp2025"], width, label="EMNLP 2025",
                color="#228833", edgecolor="white")

    ax.set_yticks(x + width / 2)
    ax.set_yticklabels(pivot.index, fontsize=9)
    if INCLUDE_AXIS_LABELS:
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Model")
    ax.set_xlim(0, 1)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                      ncol=3, fontsize=8, frameon=False)
    plt.setp(legend.get_texts(), fontweight='bold')
    ax.grid(True, alpha=0.3, axis="x")

    # Random baseline
    ax.axvline(x=0.25, color="#666666", linestyle="--", alpha=0.6, linewidth=1)

    save_plot(fig, "post_cutoff_comparison.png", output_dir)


def plot_combined_three_panel(df: pd.DataFrame, output_dir: Path) -> None:
    """Combined 1x3 panel figure: Scaling by Model + Simple vs Agentic + Ablation (msg=50).

    Designed for two-column publication layout.
    """
    print("📊 Generating combined three-panel figure...")

    # Create figure with 1x3 subplots - increase height to accommodate top legends
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # ========== Panel A: Scaling by Model ==========
    ax_scaling = axes[0]

    subset = df[df["variant"] == "offline_prompt"].copy()
    if not subset.empty:
        agg = (
            subset.groupby(["short_model", "model_family", "message_limit"], as_index=False)
            .apply(_weighted_mean_accuracy)
            .rename(columns={None: "accuracy"})
        )

        ax_scaling.set_facecolor('white')

        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
        family_linestyles = {
            "Anthropic": '-',
            "Google": '-',
            "OpenAI": '--',
        }

        for family in ["Anthropic", "Google", "OpenAI"]:
            family_models = agg[agg["model_family"] == family]["short_model"].unique()
            base_color = FAMILY_COLORS.get(family, "#999999")
            linestyle = family_linestyles.get(family, '-')

            for i, model in enumerate(sorted(family_models)):
                model_data = agg[agg["short_model"] == model].sort_values("message_limit")
                marker = markers[i % len(markers)]

                ax_scaling.plot(
                    model_data["message_limit"],
                    model_data["accuracy"],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=0.5,
                    markersize=6,
                    markeredgewidth=0.2,
                    markeredgecolor='white',
                    label=_pretty_model_name(model),
                    color=base_color,
                    alpha=0.85,
                    zorder=3,
                )

        if INCLUDE_AXIS_LABELS:
            ax_scaling.set_xlabel("Message Limit", fontsize=9)
            ax_scaling.set_ylabel("Accuracy", fontsize=9)
        ax_scaling.set_xticks(sorted(agg["message_limit"].unique()))
        ax_scaling.set_ylim(0.0, 0.75)
        ax_scaling.tick_params(axis='both', which='major', labelsize=8)

        legend = ax_scaling.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                                   ncol=3, fontsize=6, frameon=False,
                                   columnspacing=0.8, handlelength=2.0)
        plt.setp(legend.get_texts(), fontweight='medium')

        ax_scaling.grid(True, alpha=0.25, linewidth=0.6, linestyle='--', zorder=0)
        ax_scaling.set_axisbelow(True)

        # Remove top and right spines - only show left and bottom axes
        ax_scaling.spines['top'].set_visible(False)
        ax_scaling.spines['right'].set_visible(False)
        ax_scaling.spines['left'].set_linewidth(1.0)
        ax_scaling.spines['left'].set_color('#333333')
        ax_scaling.spines['bottom'].set_linewidth(1.0)
        ax_scaling.spines['bottom'].set_color('#333333')

        ax_scaling.text(0.02, 0.98, '(A)', transform=ax_scaling.transAxes,
                       fontsize=10, fontweight='bold', va='top')

    # ========== Panel B: Simple vs Agentic ==========
    ax_agentic = axes[1]

    target_limit = 50
    subset_agentic = df[df["message_limit"] == target_limit]
    if subset_agentic.empty:
        target_limit = 30
        subset_agentic = df[df["message_limit"] == target_limit]

    if not subset_agentic.empty:
        # Filter out tasks where all samples hit the limit (incomplete tasks)
        if EXCLUDE_INCOMPLETE_TASKS:
            agentic_check = subset_agentic[subset_agentic["variant"] == "offline_prompt"].copy()
            if "total_samples" in agentic_check.columns and "samples_hit_limit" in agentic_check.columns:
                agentic_check["all_hit_limit"] = (
                    agentic_check["total_samples"] == agentic_check["samples_hit_limit"]
                ) & (agentic_check["total_samples"] > 0)
                incomplete_tasks = agentic_check[agentic_check["all_hit_limit"]]["base_task"].unique()
                if len(incomplete_tasks) > 0:
                    subset_agentic = subset_agentic[~subset_agentic["base_task"].isin(incomplete_tasks)]

        # Get data for each variant WITHOUT aggregation - show all model-task combinations
        simple_data = subset_agentic[subset_agentic["variant"] == "simple_task"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "simple_acc"})

        agentic_data = subset_agentic[subset_agentic["variant"] == "offline_prompt"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "agentic_acc"})

        # Merge to get paired points for each model-task combination
        import pandas as pd
        paired_data = pd.merge(
            simple_data,
            agentic_data,
            on=["short_model", "model_family", "base_task"],
            how="inner"
        )

        if not paired_data.empty:
            ax_agentic.set_facecolor('white')

            # Define markers for each family
            family_markers = {
                "Anthropic": "o",  # Circle
                "Google": "s",     # Square
                "OpenAI": "^",     # Triangle
            }

            for family in ["Anthropic", "Google", "OpenAI"]:
                family_data = paired_data[paired_data["model_family"] == family]
                color = FAMILY_COLORS.get(family, "#999999")
                marker = family_markers.get(family, "o")

                ax_agentic.scatter(
                    family_data["simple_acc"],
                    family_data["agentic_acc"],
                    s=60,
                    marker=marker,
                    color=color,
                    alpha=0.6,
                    edgecolor="white",
                    linewidth=0.2,
                    label=family,
                    zorder=3,
                )

            ax_agentic.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, zorder=0)
            ax_agentic.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="green", zorder=0)
            ax_agentic.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", zorder=0)

            ax_agentic.text(0.15, 0.85, "Agentic\nBetter", fontsize=7, alpha=0.5, ha="center", style="italic")
            ax_agentic.text(0.85, 0.15, "Zero-shot\nBetter", fontsize=7, alpha=0.5, ha="center", style="italic")

            if INCLUDE_AXIS_LABELS:
                ax_agentic.set_xlabel("Zero-shot Accuracy", fontsize=9)
                ax_agentic.set_ylabel("Agentic Accuracy", fontsize=9)
            ax_agentic.set_xlim(0.0, 1.0)
            ax_agentic.set_ylim(0.0, 1.0)
            ax_agentic.set_aspect("equal")

            # Add grid for better visibility
            ax_agentic.grid(True, alpha=0.2, linewidth=0.5, linestyle='--', zorder=0)
            ax_agentic.set_axisbelow(True)
            ax_agentic.tick_params(axis='both', which='major', labelsize=8)

            # Remove top and right spines - only show left and bottom axes
            ax_agentic.spines['top'].set_visible(False)
            ax_agentic.spines['right'].set_visible(False)
            ax_agentic.spines['left'].set_linewidth(1.0)
            ax_agentic.spines['left'].set_color('#333333')
            ax_agentic.spines['bottom'].set_linewidth(1.0)
            ax_agentic.spines['bottom'].set_color('#333333')

            legend = ax_agentic.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                                      ncol=3, fontsize=7, frameon=False)
            plt.setp(legend.get_texts(), fontweight='bold')

            ax_agentic.text(0.02, 0.98, '(B)', transform=ax_agentic.transAxes,
                           fontsize=10, fontweight='bold', va='top')

    # ========== Panel C: Ablation Scatter (msg=50) ==========
    ax_ablation = axes[2]

    target_limit = 50
    subset_ablation = df[df["message_limit"] == target_limit]
    if subset_ablation.empty:
        target_limit = 30
        subset_ablation = df[df["message_limit"] == target_limit]

    if not subset_ablation.empty:
        # Filter out tasks where all samples hit the limit (incomplete tasks)
        if EXCLUDE_INCOMPLETE_TASKS:
            # Check both with_prompt and no_prompt variants for incomplete tasks
            with_prompt_check = subset_ablation[subset_ablation["variant"] == "offline_prompt"].copy()
            no_prompt_check = subset_ablation[subset_ablation["variant"] == "no_offline_prompt"].copy()

            incomplete_tasks = set()
            for check_df in [with_prompt_check, no_prompt_check]:
                if "total_samples" in check_df.columns and "samples_hit_limit" in check_df.columns:
                    check_df["all_hit_limit"] = (
                        check_df["total_samples"] == check_df["samples_hit_limit"]
                    ) & (check_df["total_samples"] > 0)
                    incomplete_tasks.update(check_df[check_df["all_hit_limit"]]["base_task"].unique())

            if len(incomplete_tasks) > 0:
                subset_ablation = subset_ablation[~subset_ablation["base_task"].isin(incomplete_tasks)]

        # Get data for each variant WITHOUT aggregation - show all model-task combinations
        no_prompt_data = subset_ablation[subset_ablation["variant"] == "no_offline_prompt"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "no_prompt_acc"})

        with_prompt_data = subset_ablation[subset_ablation["variant"] == "offline_prompt"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "with_prompt_acc"})

        # Merge to get paired points for each model-task combination
        paired_ablation = pd.merge(
            no_prompt_data,
            with_prompt_data,
            on=["short_model", "model_family", "base_task"],
            how="inner"
        )

        if not paired_ablation.empty:
            ax_ablation.set_facecolor('white')

            # Define markers for each family
            family_markers = {
                "Anthropic": "o",  # Circle
                "Google": "s",     # Square
                "OpenAI": "^",     # Triangle
            }

            for family in ["Anthropic", "Google", "OpenAI"]:
                family_data = paired_ablation[paired_ablation["model_family"] == family]
                color = FAMILY_COLORS.get(family, "#999999")
                marker = family_markers.get(family, "o")

                ax_ablation.scatter(
                    family_data["no_prompt_acc"],
                    family_data["with_prompt_acc"],
                    s=60,
                    marker=marker,
                    color=color,
                    alpha=0.6,
                    edgecolor="white",
                    linewidth=0.2,
                    label=family,
                    zorder=3,
                )

            ax_ablation.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, zorder=0)
            ax_ablation.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="green", zorder=0)
            ax_ablation.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", zorder=0)

            ax_ablation.text(0.15, 0.85, "Prompt\nHelps", fontsize=7, alpha=0.5, ha="center", style="italic")
            ax_ablation.text(0.85, 0.15, "Plain Agent\nBetter", fontsize=7, alpha=0.5, ha="center", style="italic")

            if INCLUDE_AXIS_LABELS:
                ax_ablation.set_xlabel("Agent Only Accuracy", fontsize=9)
                ax_ablation.set_ylabel("Agent + Prompt Accuracy", fontsize=9)
            ax_ablation.set_xlim(0.0, 1.0)
            ax_ablation.set_ylim(0.0, 1.0)
            ax_ablation.set_aspect("equal")

            # Add grid for better visibility
            ax_ablation.grid(True, alpha=0.2, linewidth=0.5, linestyle='--', zorder=0)
            ax_ablation.set_axisbelow(True)
            ax_ablation.tick_params(axis='both', which='major', labelsize=8)

            # Remove top and right spines - only show left and bottom axes
            ax_ablation.spines['top'].set_visible(False)
            ax_ablation.spines['right'].set_visible(False)
            ax_ablation.spines['left'].set_linewidth(1.0)
            ax_ablation.spines['left'].set_color('#333333')
            ax_ablation.spines['bottom'].set_linewidth(1.0)
            ax_ablation.spines['bottom'].set_color('#333333')

            legend = ax_ablation.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                                       ncol=3, fontsize=7, frameon=False)
            plt.setp(legend.get_texts(), fontweight='bold')

            ax_ablation.text(0.02, 0.98, '(C)', transform=ax_ablation.transAxes,
                           fontsize=10, fontweight='bold', va='top')

    # Adjust layout manually to control individual panel gaps
    # Parameters: [left, bottom, width, height]
    # Equal spacing: gap1 (A-B) = gap2 (B-C) = 0.04
    panel_width = 0.28  # Width of each panel
    gap = 0.02          # Gap between panels
    left_margin = 0.06

    # Position each panel manually
    axes[0].set_position([left_margin, 0.15, panel_width, 0.67])  # Panel A
    axes[1].set_position([left_margin + panel_width + gap, 0.15, panel_width, 0.67])  # Panel B
    axes[2].set_position([left_margin + 2*(panel_width + gap), 0.15, panel_width, 0.67])  # Panel C

    save_plot(fig, "combined_three_panel.png", output_dir)


def plot_combined_three_panel_with_violin(df: pd.DataFrame, output_dir: Path) -> None:
    """Combined 1x3 panel figure: Scaling by Model + Delta by Family (Violin) + Ablation (msg=50).

    Designed for two-column publication layout.
    Middle panel shows delta accuracy by task family using violin + box plots.
    """
    print("📊 Generating combined three-panel figure with violin plot...")

    # Create figure with 1x3 subplots - increase height to accommodate top legends
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # ========== Panel A: Scaling by Model (same as before) ==========
    ax_scaling = axes[0]

    subset = df[df["variant"] == "offline_prompt"].copy()
    if not subset.empty:
        agg = (
            subset.groupby(["short_model", "model_family", "message_limit"], as_index=False)
            .apply(_weighted_mean_accuracy)
            .rename(columns={None: "accuracy"})
        )

        ax_scaling.set_facecolor('white')

        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
        family_linestyles = {
            "Anthropic": '-',
            "Google": '-',
            "OpenAI": '--',
        }

        for family in ["Anthropic", "Google", "OpenAI"]:
            family_models = agg[agg["model_family"] == family]["short_model"].unique()
            base_color = FAMILY_COLORS.get(family, "#999999")
            linestyle = family_linestyles.get(family, '-')

            for i, model in enumerate(sorted(family_models)):
                model_data = agg[agg["short_model"] == model].sort_values("message_limit")
                marker = markers[i % len(markers)]

                ax_scaling.plot(
                    model_data["message_limit"],
                    model_data["accuracy"],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=0.5,
                    markersize=6,
                    markeredgewidth=0.2,
                    markeredgecolor='white',
                    label=_pretty_model_name(model),
                    color=base_color,
                    alpha=0.85,
                    zorder=3,
                )

        if INCLUDE_AXIS_LABELS:
            ax_scaling.set_xlabel("Message Limit", fontsize=9)
            ax_scaling.set_ylabel("Accuracy", fontsize=9)
        ax_scaling.set_xticks(sorted(agg["message_limit"].unique()))
        ax_scaling.set_ylim(0.0, 0.75)
        ax_scaling.tick_params(axis='both', which='major', labelsize=8)

        legend = ax_scaling.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                                   ncol=3, fontsize=6, frameon=False,
                                   columnspacing=0.8, handlelength=2.0)
        plt.setp(legend.get_texts(), fontweight='medium')

        ax_scaling.grid(True, alpha=0.25, linewidth=0.6, linestyle='--', zorder=0)
        ax_scaling.set_axisbelow(True)

        # Remove top and right spines - only show left and bottom axes
        ax_scaling.spines['top'].set_visible(False)
        ax_scaling.spines['right'].set_visible(False)
        ax_scaling.spines['left'].set_linewidth(1.0)
        ax_scaling.spines['left'].set_color('#333333')
        ax_scaling.spines['bottom'].set_linewidth(1.0)
        ax_scaling.spines['bottom'].set_color('#333333')

        ax_scaling.text(0.02, 0.98, '(A)', transform=ax_scaling.transAxes,
                       fontsize=10, fontweight='bold', va='top')

    # ========== Panel B: Delta by Family (Violin + Box) ==========
    ax_violin = axes[1]

    target_limit = 50
    subset_violin = df[df["message_limit"] == target_limit]
    if subset_violin.empty:
        target_limit = 30
        subset_violin = df[df["message_limit"] == target_limit]

    if not subset_violin.empty:
        # Filter out incomplete tasks if toggle is enabled
        if EXCLUDE_INCOMPLETE_TASKS:
            agentic_check = subset_violin[subset_violin["variant"] == "offline_prompt"].copy()
            if "total_samples" in agentic_check.columns and "samples_hit_limit" in agentic_check.columns:
                agentic_check["all_hit_limit"] = (
                    agentic_check["total_samples"] == agentic_check["samples_hit_limit"]
                ) & (agentic_check["total_samples"] > 0)
                incomplete_pairs = agentic_check[agentic_check["all_hit_limit"]][["base_task", "model"]].copy()
                if len(incomplete_pairs) > 0:
                    subset_violin["is_incomplete"] = subset_violin.apply(
                        lambda row: ((incomplete_pairs["base_task"] == row["base_task"]) &
                                    (incomplete_pairs["model"] == row["model"])).any(),
                        axis=1
                    )
                    subset_violin = subset_violin[~subset_violin["is_incomplete"]].copy()

        # Get data for each variant
        simple_data = subset_violin[subset_violin["variant"] == "simple_task"][
            ["short_model", "base_task", "accuracy", "task_family"]
        ].rename(columns={"accuracy": "simple_acc"})

        agentic_data = subset_violin[subset_violin["variant"] == "offline_prompt"][
            ["short_model", "base_task", "accuracy", "task_family"]
        ].rename(columns={"accuracy": "agentic_acc"})

        # Merge to get paired points
        import pandas as pd
        paired_data = pd.merge(
            simple_data,
            agentic_data,
            on=["short_model", "base_task", "task_family"],
            how="inner"
        )
        paired_data["delta_accuracy"] = paired_data["agentic_acc"] - paired_data["simple_acc"]

        if not paired_data.empty:
            ax_violin.set_facecolor('white')

            # Task families and colors (using default cycle)
            cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C0","C1","C2","C3"])

            # Map data task_family values to display names
            family_mapping = {
                "citation": "Citations",
                "faculty": "Faculty",
                "emnlp_awards": "Awards",
                "emnlp_awards_acl2025": "Awards",
                "emnlp_awards_emnlp2025": "Awards",
                "emnlp_historical": "Awards",
                "sota": "SOTA"
            }

            # Create normalized family column for grouping
            paired_data["family_display"] = paired_data["task_family"].map(family_mapping)

            # Define display order and colors
            families = ["Citations", "Faculty", "Awards", "SOTA"]
            data = [paired_data[paired_data["family_display"] == f]["delta_accuracy"].to_numpy() for f in families]
            colors = [cycle[0], cycle[3], cycle[1], cycle[2]]

            # Filter out empty families
            non_empty_indices = [i for i, d in enumerate(data) if len(d) > 0]
            if non_empty_indices:
                filtered_families = [families[i] for i in non_empty_indices]
                filtered_data = [data[i] for i in non_empty_indices]
                filtered_colors = [colors[i] for i in non_empty_indices]
                filtered_positions = list(range(1, len(filtered_families) + 1))

                # Violin plots for distribution
                vp = ax_violin.violinplot(filtered_data, positions=filtered_positions, widths=0.5,
                                         showmeans=False, showmedians=False, showextrema=False)
                for i, (pc, color) in enumerate(zip(vp['bodies'], filtered_colors)):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.3)
                    pc.set_edgecolor(color)
                    pc.set_linewidth(0.5)

                # Box plots on top for quartile information
                bp = ax_violin.boxplot(filtered_data, positions=filtered_positions, tick_labels=filtered_families,
                                      widths=0.25, patch_artist=True, showfliers=False, showcaps=True,
                                      boxprops=dict(linewidth=1.0), whiskerprops=dict(linewidth=0.5),
                                      medianprops=dict(linewidth=0.5, color='black'))
                for patch, color in zip(bp["boxes"], filtered_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(1.0)
                    patch.set_linewidth(0.5)
                    patch.set_edgecolor('black')

                ax_violin.set_ylim(-1, 1)
                ax_violin.axhline(0, linestyle="--", linewidth=0.5, color="gray", alpha=0.9, zorder=0)

                if INCLUDE_AXIS_LABELS:
                    ax_violin.set_ylabel(r"$\Delta$ Accuracy", fontsize=9)
                ax_violin.tick_params(axis='both', which='major', labelsize=8)

                # Remove top and right spines
                ax_violin.spines['top'].set_visible(False)
                ax_violin.spines['right'].set_visible(False)
                ax_violin.spines['left'].set_linewidth(1.0)
                ax_violin.spines['left'].set_color('#333333')
                ax_violin.spines['bottom'].set_linewidth(1.0)
                ax_violin.spines['bottom'].set_color('#333333')

                ax_violin.text(0.02, 0.98, '(B)', transform=ax_violin.transAxes,
                              fontsize=10, fontweight='bold', va='top')

    # ========== Panel C: Ablation Scatter (same as before) ==========
    ax_ablation = axes[2]

    target_limit = 50
    subset_ablation = df[df["message_limit"] == target_limit]
    if subset_ablation.empty:
        target_limit = 30
        subset_ablation = df[df["message_limit"] == target_limit]

    if not subset_ablation.empty:
        # Filter out incomplete tasks if toggle is enabled
        if EXCLUDE_INCOMPLETE_TASKS:
            with_prompt_check = subset_ablation[subset_ablation["variant"] == "offline_prompt"].copy()
            no_prompt_check = subset_ablation[subset_ablation["variant"] == "no_offline_prompt"].copy()

            incomplete_tasks = set()
            for check_df in [with_prompt_check, no_prompt_check]:
                if "total_samples" in check_df.columns and "samples_hit_limit" in check_df.columns:
                    check_df["all_hit_limit"] = (
                        check_df["total_samples"] == check_df["samples_hit_limit"]
                    ) & (check_df["total_samples"] > 0)
                    incomplete_tasks.update(check_df[check_df["all_hit_limit"]]["base_task"].unique())

            if len(incomplete_tasks) > 0:
                subset_ablation = subset_ablation[~subset_ablation["base_task"].isin(incomplete_tasks)]

        # Get data for each variant
        no_prompt_data = subset_ablation[subset_ablation["variant"] == "no_offline_prompt"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "no_prompt_acc"})

        with_prompt_data = subset_ablation[subset_ablation["variant"] == "offline_prompt"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "with_prompt_acc"})

        # Merge to get paired points
        paired_ablation = pd.merge(
            no_prompt_data,
            with_prompt_data,
            on=["short_model", "model_family", "base_task"],
            how="inner"
        )

        if not paired_ablation.empty:
            ax_ablation.set_facecolor('white')

            # Define markers for each family
            family_markers = {
                "Anthropic": "o",  # Circle
                "Google": "s",     # Square
                "OpenAI": "^",     # Triangle
            }

            for family in ["Anthropic", "Google", "OpenAI"]:
                family_data = paired_ablation[paired_ablation["model_family"] == family]
                color = FAMILY_COLORS.get(family, "#999999")
                marker = family_markers.get(family, "o")

                ax_ablation.scatter(
                    family_data["no_prompt_acc"],
                    family_data["with_prompt_acc"],
                    s=60,
                    marker=marker,
                    color=color,
                    alpha=0.6,
                    edgecolor="white",
                    linewidth=0.2,
                    label=family,
                    zorder=3,
                )

            ax_ablation.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, zorder=0)
            ax_ablation.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="green", zorder=0)
            ax_ablation.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", zorder=0)

            ax_ablation.text(0.15, 0.85, "Prompt\nHelps", fontsize=7, alpha=0.5, ha="center", style="italic")
            ax_ablation.text(0.85, 0.15, "Plain Agent\nBetter", fontsize=7, alpha=0.5, ha="center", style="italic")

            if INCLUDE_AXIS_LABELS:
                ax_ablation.set_xlabel("Agent Only Accuracy", fontsize=9)
                ax_ablation.set_ylabel("Agent + Prompt Accuracy", fontsize=9)
            ax_ablation.set_xlim(0.0, 1.0)
            ax_ablation.set_ylim(0.0, 1.0)
            ax_ablation.set_aspect("equal")

            # Add grid for better visibility
            ax_ablation.grid(True, alpha=0.2, linewidth=0.5, linestyle='--', zorder=0)
            ax_ablation.set_axisbelow(True)
            ax_ablation.tick_params(axis='both', which='major', labelsize=8)

            # Remove top and right spines
            ax_ablation.spines['top'].set_visible(False)
            ax_ablation.spines['right'].set_visible(False)
            ax_ablation.spines['left'].set_linewidth(1.0)
            ax_ablation.spines['left'].set_color('#333333')
            ax_ablation.spines['bottom'].set_linewidth(1.0)
            ax_ablation.spines['bottom'].set_color('#333333')

            legend = ax_ablation.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                                       ncol=3, fontsize=7, frameon=False)
            plt.setp(legend.get_texts(), fontweight='bold')

            ax_ablation.text(0.02, 0.98, '(C)', transform=ax_ablation.transAxes,
                           fontsize=10, fontweight='bold', va='top')

    # Adjust layout manually to control individual panel gaps
    panel_width = 0.28  # Width of each panel
    gap = 0.02          # Gap between panels
    left_margin = 0.06

    # Position each panel manually
    axes[0].set_position([left_margin, 0.15, panel_width, 0.67])  # Panel A
    axes[1].set_position([left_margin + panel_width + gap, 0.15, panel_width, 0.67])  # Panel B
    axes[2].set_position([left_margin + 2*(panel_width + gap), 0.15, panel_width, 0.67])  # Panel C

    save_plot(fig, "combined_three_panel_violin.png", output_dir)


def plot_combined_three_panel_with_waterfall(df: pd.DataFrame, output_dir: Path) -> None:
    """Combined 1x3 panel figure: Scaling by Model + Delta Waterfall + Ablation (msg=50).

    Designed for two-column publication layout.
    Middle panel shows simple and agentic accuracy side by side with delta indicator.
    """
    print("📊 Generating combined three-panel figure with waterfall plot...")

    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # ========== Panel A: Scaling by Model (same as before) ==========
    ax_scaling = axes[0]

    subset = df[df["variant"] == "offline_prompt"].copy()
    if not subset.empty:
        agg = (
            subset.groupby(["short_model", "model_family", "message_limit"], as_index=False)
            .apply(_weighted_mean_accuracy)
            .rename(columns={None: "accuracy"})
        )

        ax_scaling.set_facecolor('white')

        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
        family_linestyles = {
            "Anthropic": '-',
            "Google": '-',
            "OpenAI": '--',
        }

        for family in ["Anthropic", "Google", "OpenAI"]:
            family_models = agg[agg["model_family"] == family]["short_model"].unique()
            base_color = FAMILY_COLORS.get(family, "#999999")
            linestyle = family_linestyles.get(family, '-')

            for i, model in enumerate(sorted(family_models)):
                model_data = agg[agg["short_model"] == model].sort_values("message_limit")
                marker = markers[i % len(markers)]

                ax_scaling.plot(
                    model_data["message_limit"],
                    model_data["accuracy"],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=0.5,
                    markersize=6,
                    markeredgewidth=0.2,
                    markeredgecolor='white',
                    label=_pretty_model_name(model),
                    color=base_color,
                    alpha=0.85,
                    zorder=3,
                )

        if INCLUDE_AXIS_LABELS:
            ax_scaling.set_xlabel("Message Limit", fontsize=9)
            ax_scaling.set_ylabel("Accuracy", fontsize=9)
        ax_scaling.set_xticks(sorted(agg["message_limit"].unique()))
        ax_scaling.set_ylim(0.0, 0.75)
        ax_scaling.tick_params(axis='both', which='major', labelsize=8)

        legend = ax_scaling.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                                   ncol=3, fontsize=6, frameon=False,
                                   columnspacing=0.8, handlelength=2.0)
        plt.setp(legend.get_texts(), fontweight='medium')

        ax_scaling.grid(True, alpha=0.25, linewidth=0.6, linestyle='--', zorder=0)
        ax_scaling.set_axisbelow(True)

        ax_scaling.spines['top'].set_visible(False)
        ax_scaling.spines['right'].set_visible(False)
        ax_scaling.spines['left'].set_linewidth(1.0)
        ax_scaling.spines['left'].set_color('#333333')
        ax_scaling.spines['bottom'].set_linewidth(1.0)
        ax_scaling.spines['bottom'].set_color('#333333')

        ax_scaling.text(0.02, 1.00, '(a)', transform=ax_scaling.transAxes,
                       fontsize=10, fontweight='bold', va='top')

    # ========== Panel B: Waterfall (Simple vs Agentic by Family) ==========
    ax_waterfall = axes[1]

    target_limit = 50
    subset_waterfall = df[df["message_limit"] == target_limit]
    if subset_waterfall.empty:
        target_limit = 30
        subset_waterfall = df[df["message_limit"] == target_limit]

    if not subset_waterfall.empty:
        # Filter out incomplete tasks if toggle is enabled
        if EXCLUDE_INCOMPLETE_TASKS:
            agentic_check = subset_waterfall[subset_waterfall["variant"] == "offline_prompt"].copy()
            if "total_samples" in agentic_check.columns and "samples_hit_limit" in agentic_check.columns:
                agentic_check["all_hit_limit"] = (
                    agentic_check["total_samples"] == agentic_check["samples_hit_limit"]
                ) & (agentic_check["total_samples"] > 0)
                incomplete_pairs = agentic_check[agentic_check["all_hit_limit"]][["base_task", "model"]].copy()
                if len(incomplete_pairs) > 0:
                    subset_waterfall["is_incomplete"] = subset_waterfall.apply(
                        lambda row: ((incomplete_pairs["base_task"] == row["base_task"]) &
                                    (incomplete_pairs["model"] == row["model"])).any(),
                        axis=1
                    )
                    subset_waterfall = subset_waterfall[~subset_waterfall["is_incomplete"]].copy()

        # Get average accuracy by family and variant
        family_mapping = {
            "citation": "Citations",
            "faculty": "Faculty",
            "emnlp_awards": "Awards",
            "emnlp_awards_acl2025": "Awards",
            "emnlp_awards_emnlp2025": "Awards",
            "emnlp_historical": "Awards",
            "sota": "SOTA"
        }
        subset_waterfall["family_display"] = subset_waterfall["task_family"].map(family_mapping)

        # Calculate mean accuracy and bootstrap CI by family and variant
        def bootstrap_ci(data, n_bootstrap=1000, ci=95):
            """Calculate bootstrap confidence interval"""
            if len(data) == 0:
                return 0, 0
            if len(data) == 1:
                return 0, 0

            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(sample))

            lower = np.percentile(bootstrap_means, (100 - ci) / 2)
            upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
            return lower, upper

        family_stats = subset_waterfall.groupby(["family_display", "variant"])["accuracy"].mean().unstack(fill_value=0)

        if "simple_task" in family_stats.columns and "offline_prompt" in family_stats.columns:
            families_order = ["SOTA", "Faculty", "Awards", "Citations"]
            available_families = [f for f in families_order if f in family_stats.index]

            if available_families:
                ax_waterfall.set_facecolor('white')

                # Setup positions for grouped bars
                y_pos = np.arange(len(available_families))
                bar_height = 0.35

                # Colors
                cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C0","C1","C2","C3"])
                family_colors = {
                    "Citations": cycle[0],
                    "Faculty": cycle[3],
                    "Awards": cycle[1],
                    "SOTA": cycle[2]
                }

                # Calculate means and CIs for each family
                simple_vals = []
                simple_cis = []
                agentic_vals = []
                agentic_cis = []

                for fam in available_families:
                    # Simple task data
                    simple_data = subset_waterfall[
                        (subset_waterfall["family_display"] == fam) &
                        (subset_waterfall["variant"] == "simple_task")
                    ]["accuracy"].values
                    simple_mean = np.mean(simple_data) if len(simple_data) > 0 else 0
                    simple_lower, simple_upper = bootstrap_ci(simple_data)
                    simple_vals.append(simple_mean)
                    simple_cis.append((simple_mean - simple_lower, simple_upper - simple_mean))

                    # Agentic task data
                    agentic_data = subset_waterfall[
                        (subset_waterfall["family_display"] == fam) &
                        (subset_waterfall["variant"] == "offline_prompt")
                    ]["accuracy"].values
                    agentic_mean = np.mean(agentic_data) if len(agentic_data) > 0 else 0
                    agentic_lower, agentic_upper = bootstrap_ci(agentic_data)
                    agentic_vals.append(agentic_mean)
                    agentic_cis.append((agentic_mean - agentic_lower, agentic_upper - agentic_mean))

                colors = [family_colors.get(f, "#999999") for f in available_families]

                # Minimum visible bar width (2% of axis)
                MIN_VISIBLE_WIDTH = 0.02

                # Draw bars for simple tasks with striped pattern
                # For very small values, show minimum width
                simple_vals_display = []
                for val in simple_vals:
                    if 0 < val < MIN_VISIBLE_WIDTH:
                        simple_vals_display.append(MIN_VISIBLE_WIDTH)
                    else:
                        simple_vals_display.append(val)

                # Draw zero-shot bars with striped pattern (for all families)
                bars1 = ax_waterfall.barh(y_pos - bar_height/2, simple_vals_display, bar_height,
                                         label='Zero-shot', color='white',
                                         edgecolor='black', linewidth=0.5)

                # Add stripe pattern to all zero-shot bars
                for i, (val, color) in enumerate(zip(simple_vals_display, colors)):
                    if val > 0:
                        ax_waterfall.barh(y_pos[i] - bar_height/2, val, bar_height,
                                        color='none', edgecolor=color,
                                        linewidth=1.0, hatch='///', alpha=1.0)

                # Draw bars for agentic tasks (solid color, no pattern)
                bars2 = ax_waterfall.barh(y_pos + bar_height/2, agentic_vals, bar_height,
                                         label='Agentic', color=colors, alpha=1.0,
                                         edgecolor='black', linewidth=0.5)

                # Add error bars (95% bootstrap CI)
                # Error bars for simple tasks
                simple_ci_array = np.array(simple_cis).T  # Shape: (2, n_families)
                for i in range(len(available_families)):
                    if simple_cis[i][0] > 0 or simple_cis[i][1] > 0:  # Only if CI exists
                        ax_waterfall.errorbar(simple_vals[i], y_pos[i] - bar_height/2,
                                            xerr=[[simple_cis[i][0]], [simple_cis[i][1]]],
                                            fmt='none', ecolor='black', elinewidth=0.8,
                                            capsize=3, capthick=0.8, alpha=0.8)

                # Error bars for agentic tasks
                for i in range(len(available_families)):
                    if agentic_cis[i][0] > 0 or agentic_cis[i][1] > 0:  # Only if CI exists
                        ax_waterfall.errorbar(agentic_vals[i], y_pos[i] + bar_height/2,
                                            xerr=[[agentic_cis[i][0]], [agentic_cis[i][1]]],
                                            fmt='none', ecolor='black', elinewidth=0.8,
                                            capsize=3, capthick=0.8, alpha=0.8)

                # Delta is shown by the difference between agentic and simple bars
                # No need for additional visual indicators

                # Add accuracy text labels
                for i, (simple, simple_display, agentic, fam) in enumerate(zip(simple_vals, simple_vals_display, agentic_vals, available_families)):
                    # Label for zero-shot bar
                    if simple_display > 0:
                        # Use actual value for label, not display value
                        label_text = f'{simple:.1%}' if simple >= 0.01 else f'{simple:.2%}'
                        # For very small values (< 1%), position text to the right to avoid y-axis overlap
                        if simple < 0.01:
                            ax_waterfall.text(simple_display + 0.01, y_pos[i] - bar_height/2,
                                            label_text, ha='left', va='center',
                                            fontsize=7, fontweight='normal', color='black')
                        else:
                            ax_waterfall.text(simple_display / 2, y_pos[i] - bar_height/2,
                                            label_text, ha='center', va='center',
                                            fontsize=7, fontweight='normal', color='black')

                    # Label for agentic bar
                    ax_waterfall.text(agentic / 2, y_pos[i] + bar_height/2,
                                    f'{agentic:.1%}', ha='center', va='center',
                                    fontsize=7, fontweight='normal', color='white')

                ax_waterfall.set_yticks(y_pos)
                ax_waterfall.set_yticklabels(available_families, fontsize=8)
                if INCLUDE_AXIS_LABELS:
                    ax_waterfall.set_xlabel("Accuracy", fontsize=9)
                ax_waterfall.set_xlim(0, 1)

                # Add custom legend with correct styling
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='white', edgecolor='gray', hatch='///', label='Zero-shot'),
                    Patch(facecolor='gray', edgecolor='black', label='Agentic')
                ]
                legend = ax_waterfall.legend(handles=legend_elements, loc="upper center",
                                            bbox_to_anchor=(0.5, 1.15),
                                            ncol=2, fontsize=7, frameon=False)
                plt.setp(legend.get_texts(), fontweight='bold')

                ax_waterfall.grid(True, alpha=0.2, axis='x', linewidth=0.5, linestyle='--', zorder=0)
                ax_waterfall.set_axisbelow(True)
                ax_waterfall.tick_params(axis='both', which='major', labelsize=8)

                ax_waterfall.spines['top'].set_visible(False)
                ax_waterfall.spines['right'].set_visible(False)
                ax_waterfall.spines['left'].set_linewidth(1.0)
                ax_waterfall.spines['left'].set_color('#333333')
                ax_waterfall.spines['bottom'].set_linewidth(1.0)
                ax_waterfall.spines['bottom'].set_color('#333333')

                ax_waterfall.text(0.02, 1.00, '(b)', transform=ax_waterfall.transAxes,
                                 fontsize=10, fontweight='bold', va='top')

    # ========== Panel C: Ablation Scatter (same as before) ==========
    ax_ablation = axes[2]

    target_limit = 50
    subset_ablation = df[df["message_limit"] == target_limit]
    if subset_ablation.empty:
        target_limit = 30
        subset_ablation = df[df["message_limit"] == target_limit]

    if not subset_ablation.empty:
        # Filter out incomplete tasks if toggle is enabled
        if EXCLUDE_INCOMPLETE_TASKS:
            with_prompt_check = subset_ablation[subset_ablation["variant"] == "offline_prompt"].copy()
            no_prompt_check = subset_ablation[subset_ablation["variant"] == "no_offline_prompt"].copy()

            incomplete_tasks = set()
            for check_df in [with_prompt_check, no_prompt_check]:
                if "total_samples" in check_df.columns and "samples_hit_limit" in check_df.columns:
                    check_df["all_hit_limit"] = (
                        check_df["total_samples"] == check_df["samples_hit_limit"]
                    ) & (check_df["total_samples"] > 0)
                    incomplete_tasks.update(check_df[check_df["all_hit_limit"]]["base_task"].unique())

            if len(incomplete_tasks) > 0:
                subset_ablation = subset_ablation[~subset_ablation["base_task"].isin(incomplete_tasks)]

        # Get data for each variant
        no_prompt_data = subset_ablation[subset_ablation["variant"] == "no_offline_prompt"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "no_prompt_acc"})

        with_prompt_data = subset_ablation[subset_ablation["variant"] == "offline_prompt"][
            ["short_model", "model_family", "base_task", "accuracy"]
        ].rename(columns={"accuracy": "with_prompt_acc"})

        # Merge to get paired points
        paired_ablation = pd.merge(
            no_prompt_data,
            with_prompt_data,
            on=["short_model", "model_family", "base_task"],
            how="inner"
        )

        if not paired_ablation.empty:
            ax_ablation.set_facecolor('white')

            family_markers = {
                "Anthropic": "o",
                "Google": "s",
                "OpenAI": "^",
            }

            for family in ["Anthropic", "Google", "OpenAI"]:
                family_data = paired_ablation[paired_ablation["model_family"] == family]
                color = FAMILY_COLORS.get(family, "#999999")
                marker = family_markers.get(family, "o")

                ax_ablation.scatter(
                    family_data["no_prompt_acc"],
                    family_data["with_prompt_acc"],
                    s=60,
                    marker=marker,
                    color=color,
                    alpha=0.6,
                    edgecolor="white",
                    linewidth=0.2,
                    label=family,
                    zorder=3,
                )

            ax_ablation.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, zorder=0)
            ax_ablation.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="green", zorder=0)
            ax_ablation.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", zorder=0)

            ax_ablation.text(0.15, 0.85, "Prompt\nHelps", fontsize=7, alpha=0.5, ha="center", style="italic")
            ax_ablation.text(0.85, 0.15, "Plain Agent\nBetter", fontsize=7, alpha=0.5, ha="center", style="italic")

            if INCLUDE_AXIS_LABELS:
                ax_ablation.set_xlabel("Agent Only Accuracy", fontsize=9)
                ax_ablation.set_ylabel("Agent + Prompt Accuracy", fontsize=9)
            ax_ablation.set_xlim(0.0, 1.0)
            ax_ablation.set_ylim(0.0, 1.0)
            ax_ablation.set_aspect("equal")

            ax_ablation.grid(True, alpha=0.2, linewidth=0.5, linestyle='--', zorder=0)
            ax_ablation.set_axisbelow(True)
            ax_ablation.tick_params(axis='both', which='major', labelsize=8)

            ax_ablation.spines['top'].set_visible(False)
            ax_ablation.spines['right'].set_visible(False)
            ax_ablation.spines['left'].set_linewidth(1.0)
            ax_ablation.spines['left'].set_color('#333333')
            ax_ablation.spines['bottom'].set_linewidth(1.0)
            ax_ablation.spines['bottom'].set_color('#333333')

            legend = ax_ablation.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                                       ncol=3, fontsize=7, frameon=False)
            plt.setp(legend.get_texts(), fontweight='bold')

            ax_ablation.text(0.02, 1.00, '(c)', transform=ax_ablation.transAxes,
                           fontsize=10, fontweight='bold', va='top')

    # Adjust layout
    panel_width = 0.28
    gap = 0.02
    left_margin = 0.06

    axes[0].set_position([left_margin, 0.15, panel_width, 0.67])
    axes[1].set_position([left_margin + panel_width + gap, 0.15, panel_width, 0.67])
    axes[2].set_position([left_margin + 2*(panel_width + gap), 0.15, panel_width, 0.67])

    save_plot(fig, "combined_three_panel_horizontal_bar.png", output_dir)


# ---------------------------------------------------------------------------
# Agent Behavior Analysis Plots (from generate_plots.py)
# ---------------------------------------------------------------------------

def plot_strata_distribution(output_dir: Path, csv_path: str = 'aba_stratum_counts.csv') -> None:
    """Create an elegant donut chart showing outcome distribution.
    
    Style: Uses strict STRATUM_COLORS (Green/Red/Blue) to match other figures.
    """
    print("📊 Generating strata distribution donut chart...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {csv_path}")
        return

    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor='white')

    # Strict ordering to match legend/logic
    labels = ['Correct', 'Wrong', 'Incomplete']
    
    # Map counts to labels safely
    counts = []
    for label in labels:
        row = df[df['stratum'].str.lower() == label.lower()]
        counts.append(row['count'].values[0] if not row.empty else 0)
    
    sizes = np.array(counts)
    colors = [STRATUM_COLORS['correct'], STRATUM_COLORS['wrong'], STRATUM_COLORS['incomplete']]
    explode = (0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct='',
        startangle=90,
        explode=explode,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white', 'width': 0.60}, # Slightly thinner ring
        pctdistance=0.75
    )

    # Add custom annotations with count and percentage
    total = sum(sizes)
    for i, (wedge, size) in enumerate(zip(wedges, sizes)):
        if size == 0: continue
        
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))

        # Position for label
        ax.annotate(
            f'{labels[i]}\n{size:,}\n({size/total*100:.1f}%)',
            xy=(x * 0.75, y * 0.75),
            ha='center', va='center',
            fontsize=9,
            fontweight='medium',
            color='white' if i != 2 else UI_COLORS['text'] # Dark text for lighter segments if needed
        )

    # Center annotation
    ax.text(0, 0, f'N = {total:,}', ha='center', va='center',
            fontsize=12, fontweight='bold', color=UI_COLORS['text'])

    # Legend at top
    legend_elements = [
        mpatches.Patch(facecolor=STRATUM_COLORS['correct'], edgecolor='white', label='Correct'),
        mpatches.Patch(facecolor=STRATUM_COLORS['wrong'], edgecolor='white', label='Wrong'),
        mpatches.Patch(facecolor=STRATUM_COLORS['incomplete'], edgecolor='white', label='Incomplete'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12),
                      ncol=3, frameon=False, fontsize=9)
    plt.setp(legend.get_texts(), fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'aba_strata_distribution.png', output_dir)


def plot_outcomes_by_limit(output_dir: Path) -> None:
    """Create refined stacked bar chart for budget analysis.
    
    Style: Stacked bars using STRATUM_COLORS. Matches scaling plots.
    """
    print("📊 Generating outcomes by message limit stacked bar chart...")

    # Data from the paper text
    limits = [15, 30, 50]
    correct = [29.2, 37.9, 39.9]
    wrong = [25.7, 36.1, 44.2]
    incomplete = [45.1, 26.0, 15.9]

    fig, ax = plt.subplots(figsize=(5, 3.8), facecolor='white')
    style_axis(ax, ylabel='Proportion of Runs (%)', xlabel='Message Limit')

    x = np.arange(len(limits))
    width = 0.6

    # Stacked bars with solid colors
    bars1 = ax.bar(x, correct, width, label='Correct', color=STRATUM_COLORS['correct'],
                   edgecolor='white', linewidth=1.0)
    bars2 = ax.bar(x, wrong, width, bottom=correct, label='Wrong', color=STRATUM_COLORS['wrong'],
                   edgecolor='white', linewidth=1.0)
    bars3 = ax.bar(x, incomplete, width, bottom=np.array(correct)+np.array(wrong),
                   label='Incomplete', color=STRATUM_COLORS['incomplete'], 
                   edgecolor='white', linewidth=1.0)

    # Add percentage labels inside bars
    def add_bar_labels(bars, values, bottoms):
        for bar, val, bottom in zip(bars, values, bottoms):
            height = bar.get_height()
            if height > 8:  # Only label if segment is big enough
                ax.text(bar.get_x() + bar.get_width()/2, bottom + height/2,
                       f'{val:.1f}%', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')

    add_bar_labels(bars1, correct, [0]*3)
    add_bar_labels(bars2, wrong, correct)
    add_bar_labels(bars3, incomplete, np.array(correct)+np.array(wrong))

    ax.set_xticks(x)
    ax.set_xticklabels(limits)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])

    # Legend at top
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
                      frameon=False, fontsize=9, columnspacing=1.5)
    plt.setp(legend.get_texts(), fontweight='bold')

    plt.tight_layout()
    save_plot(fig, 'aba_outcomes_by_message_limit.png', output_dir)


def plot_failure_taxonomy(output_dir: Path, csv_path: str = 'aba_failure_taxonomy.csv') -> None:
    """Create refined horizontal bar chart for failure taxonomy.
    
    Style: Uses solid Red (STRATUM_COLORS['wrong']) to visually link to 'Wrong' outcomes.
    Replaces gradients with clean, flat bars.
    """
    print("📊 Generating failure taxonomy horizontal bar chart...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {csv_path}")
        return

    df = df.sort_values('count', ascending=True)

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='white')
    style_axis(ax, xlabel='Count')

    y = np.arange(len(df))

    # Use solid 'wrong' color (Red) for all bars to indicate these are failures
    bars = ax.barh(y, df['count'], height=0.7, color=STRATUM_COLORS['wrong'],
                   edgecolor='white', linewidth=1, alpha=0.9)

    # Add value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + max(df['count'])*0.02, bar.get_y() + bar.get_height()/2,
               f'{int(row["count"])} ({row["percent"]:.1f}%)',
               va='center', ha='left', fontsize=8, color=UI_COLORS['text'])

    ax.set_yticks(y)
    ax.set_yticklabels(df['failure_type'], fontsize=9)
    ax.set_xlim(0, max(df['count']) * 1.35)

    # Remove y-axis spine for cleaner look
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    save_plot(fig, 'aba_failure_taxonomy_wrong.png', output_dir)


def plot_nonconvergence(output_dir: Path, csv_path: str = 'aba_nonconvergence.csv') -> None:
    """Create refined horizontal bar chart for non-convergence categories.
    
    Style: Uses solid Blue (STRATUM_COLORS['incomplete']) to visually link to 'Incomplete' outcomes.
    """
    print("📊 Generating non-convergence categories horizontal bar chart...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {csv_path}")
        return

    df = df.sort_values('count', ascending=True)

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='white')
    style_axis(ax, xlabel='Count')

    y = np.arange(len(df))

    # Use solid 'incomplete' color (Blue)
    bars = ax.barh(y, df['count'], height=0.7, color=STRATUM_COLORS['incomplete'],
                   edgecolor='white', linewidth=1, alpha=0.9)

    # Add value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + max(df['count'])*0.02, bar.get_y() + bar.get_height()/2,
               f'{int(row["count"])} ({row["percent"]:.1f}%)',
               va='center', ha='left', fontsize=8, color=UI_COLORS['text'])

    ax.set_yticks(y)
    ax.set_yticklabels(df['nonconv_type'], fontsize=9)
    ax.set_xlim(0, max(df['count']) * 1.35)

    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    save_plot(fig, 'aba_nonconvergence_categories.png', output_dir)


def plot_bottlenecks(output_dir: Path, csv_path: str = 'aba_bottlenecks.csv') -> None:
    """Create refined horizontal bar chart for bottleneck categories.
    
    Style: Uses Purple (#AA3377) from TASK_FAMILY_COLORS for a distinct, neutral category
    that isn't explicitly 'wrong' or 'incomplete' but related to process.
    """
    print("📊 Generating bottleneck categories horizontal bar chart...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {csv_path}")
        return

    df = df.sort_values('count', ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    style_axis(ax, xlabel='Count')

    y = np.arange(len(df))

    # Use a distinct color from the palette (Purple)
    # This comes from TASK_FAMILY_COLORS[5]
    bottleneck_color = "#AA3377" 

    bars = ax.barh(y, df['count'], height=0.7, color=bottleneck_color,
                   edgecolor='white', linewidth=1, alpha=0.9)

    # Add value labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + max(df['count'])*0.02, bar.get_y() + bar.get_height()/2,
               f'{int(row["count"])} ({row["percent"]:.1f}%)',
               va='center', ha='left', fontsize=8, color=UI_COLORS['text'])

    ax.set_yticks(y)
    ax.set_yticklabels(df['bottleneck'], fontsize=9)
    ax.set_xlim(0, max(df['count']) * 1.35)

    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    save_plot(fig, 'aba_bottleneck_categories.png', output_dir)


def plot_correct_flags(output_dir: Path) -> None:
    """Create refined horizontal bar chart for judge flags in correct runs.
    
    Style: Uses solid Green (STRATUM_COLORS['correct']) to visually link to 'Correct' outcomes.
    """
    print("📊 Generating correct run flags horizontal bar chart...")

    # Data derived from the plot (fraction of correct runs)
    data = {
        'flag': ['Verification noted', 'Unsupported claims noted',
                 'Lucky/guessing noted', 'Redundancy noted'],
        'fraction': [0.16, 0.27, 0.43, 0.79]
    }
    df = pd.DataFrame(data)
    df = df.sort_values('fraction', ascending=True)

    fig, ax = plt.subplots(figsize=(6, 3), facecolor='white')
    style_axis(ax, xlabel='Fraction of Correct Runs')

    y = np.arange(len(df))

    # Use solid 'correct' color (Green)
    bars = ax.barh(y, df['fraction'], height=0.65, color=STRATUM_COLORS['correct'],
                   edgecolor='white', linewidth=1, alpha=0.9)

    # Add percentage labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{row["fraction"]*100:.0f}%',
               va='center', ha='left', fontsize=9, fontweight='medium',
               color=UI_COLORS['text'])

    ax.set_yticks(y)
    ax.set_yticklabels(df['flag'], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    save_plot(fig, 'aba_correct_run_flags.png', output_dir)


# ---------------------------------------------------------------------------
# Efficiency Frontier Plots
# ---------------------------------------------------------------------------

def plot_agent_vs_zeroshot_efficiency_frontier(
    csv_in: Path = Path("analysis/agent_query_analysis_msg_limits.csv"),
    out_points: Path = Path("analysis/comprehensive/tables/agent_vs_zeroshot_efficiency_frontier_points.csv"),
    out_fig: Path = Path("analysis/comprehensive/plots/fig_agent_vs_zeroshot_efficiency_frontier.png"),
    exclude_incomplete: bool = False
):
    """
    Agentic-vs-zeroshot efficiency frontier.

    - Zeroshot: tasks with "simple" in name
    - Agentic: tasks without "simple"
    - Uses effective tokens including cache_read_tokens + cache_write_tokens
    - X: token overhead vs zeroshot (log scale)
    - Y: accuracy gain vs zeroshot
    - Color: provider; Marker: message limit; thin connecting line per model
    """
    df = pd.read_csv(csv_in)
    required = {"task","model","message_limit","num_samples","total_tokens","accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()
    df = df[~df["task"].astype(str).str.contains("no_offline_prompt", regex=False)].copy()

    # Filter out incomplete tasks if toggle is enabled
    if exclude_incomplete:
        try:
            summary_dfs = []
            for msg_limit in [15, 30, 50]:
                summary_path = Path(f"logs_msg{msg_limit}_summary.csv")
                if summary_path.exists():
                    summary_df = pd.read_csv(summary_path)
                    if "samples_hit_limit" in summary_df.columns and "total_samples" in summary_df.columns:
                        summary_df["all_hit_limit"] = (
                            summary_df["total_samples"] == summary_df["samples_hit_limit"]
                        ) & (summary_df["total_samples"] > 0)
                        incomplete = summary_df[summary_df["all_hit_limit"]][["task_name", "model"]].copy()
                        incomplete.rename(columns={"task_name": "task"}, inplace=True)
                        summary_dfs.append(incomplete)

            if summary_dfs:
                all_incomplete = pd.concat(summary_dfs, ignore_index=True).drop_duplicates()
                df["is_incomplete"] = df.apply(
                    lambda row: ((all_incomplete["task"] == row["task"]) &
                                (all_incomplete["model"] == row["model"])).any(),
                    axis=1
                )
                n_excluded = df["is_incomplete"].sum()
                if n_excluded > 0:
                    print(f"ℹ️ Excluding {n_excluded} incomplete task-model combinations (all samples hit limit)")
                    df = df[~df["is_incomplete"]].copy()
        except Exception as e:
            print(f"⚠️ Could not load hit_limit information: {e}")

    df["effective_tokens"] = df["total_tokens"] + df.get("cache_read_tokens", 0) + df.get("cache_write_tokens", 0)
    df["setting"] = np.where(df["task"].astype(str).str.contains("simple", regex=False), "zeroshot", "agentic")

    def provider_from_model(m: str) -> str:
        if m.startswith("openai/"): return "OpenAI"
        if m.startswith("google/"): return "Google"
        if m.startswith("anthropic/"): return "Anthropic"
        return "Other"

    def eff_tokens_per_sample(g: pd.DataFrame) -> float:
        return float(g["effective_tokens"].sum() / g["num_samples"].sum())

    def wacc(g: pd.DataFrame) -> float:
        return float(np.average(g["accuracy"], weights=g["num_samples"]))

    zs = (df[df["setting"]=="zeroshot"]
          .groupby("model", as_index=False)
          .apply(lambda g: pd.Series({
              "zs_tokens_per_sample": eff_tokens_per_sample(g),
              "zs_accuracy": wacc(g),
          }))
          .reset_index(drop=True))

    ag = (df[df["setting"]=="agentic"]
          .groupby(["model","message_limit"], as_index=False)
          .apply(lambda g: pd.Series({
              "ag_tokens_per_sample": eff_tokens_per_sample(g),
              "ag_accuracy": wacc(g),
          }))
          .reset_index(drop=True))

    pts = ag.merge(zs, on="model", how="inner")
    pts["token_overhead_x"] = pts["ag_tokens_per_sample"] / pts["zs_tokens_per_sample"]
    pts["accuracy_gain"] = pts["ag_accuracy"] - pts["zs_accuracy"]
    pts["provider"] = pts["model"].apply(provider_from_model)

    def pretty_model(m: str) -> str:
        """Convert model names to standardized display format"""
        s = m.split("/")[-1]
        s = s.replace("@20251001","").replace("@20250929","").replace("@20251101","")
        mapping = {
            "claude-opus-4-5": "Claude Opus 4.5",
            "claude-sonnet-4-5": "Claude Sonnet 4.5",
            "claude-haiku-4-5": "Claude Haiku 4.5",
            "gemini-3-pro-preview": "Gemini 3 Pro Preview",
            "gemini-3-flash-preview": "Gemini 3 Flash Preview",
            "gemini-2.5-pro": "Gemini 2.5 Pro",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gpt-5.2-2025-12-11": "GPT-5.2",
            "gpt-5.1-2025-11-13": "GPT-5.1",
            "gpt-5-mini-2025-08-07": "GPT-5 Mini",
            "gpt-5-nano-2025-08-07": "GPT-5 Nano",
        }
        return mapping.get(s, s)

    pts["model_label"] = pts["model"].apply(pretty_model)

    # Ensure output directories exist
    out_points.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    pts.sort_values(["provider","model_label","message_limit"]).to_csv(out_points, index=False)

    marker_map = {15:"o", 30:"s", 50:"^"}
    providers = ["OpenAI","Google","Anthropic","Other"]
    prov_color = FAMILY_COLORS
    linestyles = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    # Group models by provider to assign line styles
    provider_models = {}
    for prov in providers:
        provider_models[prov] = sorted(pts[pts["provider"] == prov]["model_label"].unique())

    # Connect each model across budgets with different line styles
    for (prov, model), g in pts.groupby(["provider","model_label"]):
        g = g.sort_values("message_limit")
        model_idx = provider_models[prov].index(model) if model in provider_models[prov] else 0
        linestyle = linestyles[model_idx % len(linestyles)]
        ax.plot(g["token_overhead_x"], g["accuracy_gain"],
                linewidth=0.8, alpha=0.7,
                color=prov_color.get(prov, "#BBBBBB"),
                linestyle=linestyle)

    # Points
    for ml, mk in marker_map.items():
        g = pts[pts["message_limit"] == ml]
        ax.scatter(
            g["token_overhead_x"],
            g["accuracy_gain"],
            marker=mk,
            s=38,
            alpha=0.9,
            c=[prov_color.get(p, "C3") for p in g["provider"]],
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_xlim(left=9)
    x_ticks = [10, 20, 50, 100, 200]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['10', '20', '50', '100', '200'])

    ax.axhline(0.0, linewidth=0.5, linestyle="--", color="black", alpha=0.5)
    ax.axvline(1.0, linewidth=0.8, linestyle="--")

    ax.set_xlabel("Token overhead vs zeroshot (×, log scale)")
    ax.set_ylabel("Accuracy gain vs zeroshot")

    # Create legend with individual model names and their line styles
    from matplotlib.lines import Line2D
    model_handles = []
    for prov in providers:
        if prov in provider_models and len(provider_models[prov]) > 0:
            for idx, model_label in enumerate(provider_models[prov]):
                linestyle = linestyles[idx % len(linestyles)]
                handle = Line2D([0], [0],
                              color=prov_color[prov],
                              linestyle=linestyle,
                              linewidth=0.5,
                              label=model_label)
                model_handles.append(handle)

    fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, fontsize=8)

    ml_handles = [
        Line2D([0],[0], marker="o", linestyle="None", markersize=6, label="15"),
        Line2D([0],[0], marker="s", linestyle="None", markersize=6, label="30"),
        Line2D([0],[0], marker="^", linestyle="None", markersize=6, label="50"),
    ]
    ax.legend(handles=ml_handles, title="Message limit", frameon=False, fontsize=8, title_fontsize=8,
              ncol=1, loc="lower right", bbox_to_anchor=(1.05, 0.00))

    fig.subplots_adjust(top=0.82, bottom=0.15)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_fig}")


def plot_agent_vs_zeroshot_performance(
    csv_in: Path = Path("logs_msg50_summary.csv"),
    out_pairs: Path = Path("analysis/comprehensive/tables/pairs_agentic_vs_zeroshot_msg50.csv"),
    out_tab_family: Path = Path("analysis/comprehensive/tables/table_delta_by_family_msg50.csv"),
    out_tab_regime: Path = Path("analysis/comprehensive/tables/table_delta_by_regime_msg50.csv"),
    out_scatter: Path = Path("analysis/comprehensive/plots/fig_agentic_vs_zeroshot_scatter_msg50.png"),
    out_family_box: Path = Path("analysis/comprehensive/plots/fig_delta_by_family_boxplot_msg50.png"),
    out_regime_box: Path = Path("analysis/comprehensive/plots/fig_delta_by_regime_boxplot_msg50.png"),
    exclude_incomplete: bool = True
):
    """
    Performance-only analysis: agentic vs zeroshot (ignoring cost/budget).

    Outputs:
    - pairs CSV
    - tables by family and regime
    - scatter plot
    - boxplots by family and regime
    """
    import re

    def is_no_offline(t: str) -> bool:
        t = str(t)
        return ("no_offline" in t) or ("no-offline" in t) or ("nooffline" in t)

    def is_simple(t: str) -> bool:
        return "simple" in str(t)

    def base_task_name(t: str) -> str:
        t = str(t)
        t = re.sub(r"_no_offline_prompt(_task)?", "", t)
        t = re.sub(r"_task_no_offline_prompt", "", t)
        t = re.sub(r"_no_offline_prompt_task", "", t)
        t = re.sub(r"_simple_task", "", t)
        t = re.sub(r"_task$", "", t)
        return t

    def task_family(base: str) -> str:
        b = base.lower()
        if b.startswith("citation") or "citation_" in b:
            return "Citations"
        if b.startswith("faculty") or "professor" in b:
            return "Faculty"
        if "award" in b and "2025" in b:
            return "Awards"
        if b.startswith("sota") or "leaderboard" in b or "benchmark" in b:
            return "SOTA"
        return "Other"

    def regime_from_family(fam: str) -> str:
        if fam in ("Citations", "Faculty"):
            return "Evidence-intensive"
        if fam in ("Awards", "SOTA"):
            return "Structured prediction"
        return "Other"

    def provider(model: str) -> str:
        m = str(model)
        if m.startswith("openai/"):
            return "OpenAI"
        if m.startswith("google/"):
            return "Google"
        if m.startswith("anthropic/"):
            return "Anthropic"
        return "Other"

    def summarize(group: pd.DataFrame) -> pd.Series:
        d = group["delta_accuracy"].to_numpy()
        return pd.Series({
            "n_pairs": int(len(d)),
            "mean_delta": float(np.mean(d)) if len(d) else np.nan,
            "median_delta": float(np.median(d)) if len(d) else np.nan,
            "std_delta": float(np.std(d, ddof=0)) if len(d) else np.nan,
            "iqr_delta": float(np.percentile(d, 75) - np.percentile(d, 25)) if len(d) else np.nan,
            "frac_positive": float(np.mean(d > 0)) if len(d) else np.nan,
            "frac_negative": float(np.mean(d < 0)) if len(d) else np.nan,
        })

    df0 = pd.read_csv(csv_in)
    required = {"task_name", "model", "accuracy", "total_samples"}
    missing = required - set(df0.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Filter ablations
    df = df0[~df0["task_name"].apply(is_no_offline)].copy()
    df["base_task"] = df["task_name"].apply(base_task_name)
    df["setting"] = np.where(df["task_name"].apply(is_simple), "zeroshot", "agentic")
    df["family"] = df["base_task"].apply(task_family)
    df["regime"] = df["family"].apply(regime_from_family)
    df["provider"] = df["model"].apply(provider)

    # Filter out incomplete tasks if toggle is enabled
    if exclude_incomplete and "samples_hit_limit" in df.columns:
        agentic_df = df[df["setting"] == "agentic"].copy()
        agentic_df["all_hit_limit"] = (
            agentic_df["total_samples"] == agentic_df["samples_hit_limit"]
        ) & (agentic_df["total_samples"] > 0)

        incomplete_pairs = agentic_df[agentic_df["all_hit_limit"]][["base_task", "model"]].copy()

        if len(incomplete_pairs) > 0:
            print(f"ℹ️ Excluding {len(incomplete_pairs)} incomplete (task, model) pairs (all samples hit limit)")
            df["is_incomplete"] = df.apply(
                lambda row: ((incomplete_pairs["base_task"] == row["base_task"]) &
                            (incomplete_pairs["model"] == row["model"])).any(),
                axis=1
            )
            df = df[~df["is_incomplete"]].drop(columns=["is_incomplete"])

    zs = (df[df["setting"] == "zeroshot"]
          .rename(columns={"accuracy": "zs_accuracy", "total_samples": "zs_total_samples"})
          [["base_task", "model", "zs_accuracy", "zs_total_samples"]])

    ag = (df[df["setting"] == "agentic"]
          .rename(columns={"accuracy": "ag_accuracy", "total_samples": "ag_total_samples"})
          [["base_task", "model", "ag_accuracy", "ag_total_samples"]])

    pairs = ag.merge(zs, on=["base_task", "model"], how="inner")
    pairs = pairs[pairs["ag_accuracy"] > 0].copy()
    pairs["delta_accuracy"] = pairs["ag_accuracy"] - pairs["zs_accuracy"]
    pairs["family"] = pairs["base_task"].apply(task_family)
    pairs["regime"] = pairs["family"].apply(regime_from_family)
    pairs["provider"] = pairs["model"].apply(provider)

    # Ensure output directories exist
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    out_scatter.parent.mkdir(parents=True, exist_ok=True)

    pairs.to_csv(out_pairs, index=False)

    # Tables
    tab_family = pairs.groupby("family").apply(summarize).reset_index()
    tab_regime = pairs.groupby("regime").apply(summarize).reset_index()
    tab_family.to_csv(out_tab_family, index=False)
    tab_regime.to_csv(out_tab_regime, index=False)

    # Style
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C0","C1","C2","C3"])
    prov_color = {"Anthropic": cycle[1], "Google": cycle[2], "OpenAI": cycle[0], "Other": cycle[3]}

    # Figure 1: Scatter (Agentic vs Zeroshot)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", alpha=0.9)
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.06)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.03)

    for prov in ["Anthropic", "Google", "OpenAI"]:
        g = pairs[pairs["provider"] == prov]
        ax.scatter(
            g["zs_accuracy"], g["ag_accuracy"],
            s=75, alpha=0.8, label=prov,
            color=prov_color[prov], edgecolors="white", linewidths=0.7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Zeroshot accuracy")
    ax.set_ylabel("Agentic accuracy")
    ax.text(0.14, 0.84, "Agentic\nbetter", fontsize=11, color="gray", style="italic")
    ax.text(0.72, 0.14, "Zeroshot\nbetter", fontsize=11, color="gray", style="italic")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_scatter, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_scatter}")

    # Figure 2: Delta by family (violin + box)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    families = ["Citations", "Faculty", "Awards", "SOTA"]
    data = [pairs[pairs["family"] == f]["delta_accuracy"].to_numpy() for f in families]
    colors = [cycle[0], cycle[3], cycle[1], cycle[2]]

    non_empty_indices = [i for i, d in enumerate(data) if len(d) > 0]
    if not non_empty_indices:
        print("⚠️ No data for family boxplot after filtering")
        plt.close(fig)
    else:
        filtered_families = [families[i] for i in non_empty_indices]
        filtered_data = [data[i] for i in non_empty_indices]
        filtered_colors = [colors[i] for i in non_empty_indices]
        filtered_positions = list(range(1, len(filtered_families) + 1))

        vp = ax.violinplot(filtered_data, positions=filtered_positions, widths=0.5, showmeans=False, showmedians=False, showextrema=False)
        for i, (pc, color) in enumerate(zip(vp['bodies'], filtered_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.5)

        bp = ax.boxplot(filtered_data, positions=filtered_positions, tick_labels=filtered_families, widths=0.25, patch_artist=True,
                        showfliers=False, showcaps=True, boxprops=dict(linewidth=1.2),
                        whiskerprops=dict(linewidth=0.5), medianprops=dict(linewidth=0.5, color='black'))
        for patch, color in zip(bp["boxes"], filtered_colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)
            patch.set_linewidth(0.5)
            patch.set_edgecolor('black')

        ax.set_ylim(-1, 1)
        ax.axhline(0, linestyle="--", linewidth=0.5, color="gray", alpha=0.9)
        ax.set_ylabel(r"$\Delta$ Accuracy (Agentic $-$ Zeroshot)")
        fig.tight_layout()
        fig.savefig(out_family_box, dpi=300)
        plt.close(fig)
        print(f"✓ Saved: {out_family_box}")

    # Figure 3: Delta by regime (violin + box)
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    regimes = ["Evidence-intensive", "Structured prediction"]
    data = [pairs[pairs["regime"] == r]["delta_accuracy"].to_numpy() for r in regimes]
    colors = [cycle[2], cycle[3]]

    non_empty_indices = [i for i, d in enumerate(data) if len(d) > 0]
    if not non_empty_indices:
        print("⚠️ No data for regime boxplot after filtering")
        plt.close(fig)
    else:
        filtered_regimes = [regimes[i] for i in non_empty_indices]
        filtered_data = [data[i] for i in non_empty_indices]
        filtered_colors = [colors[i] for i in non_empty_indices]
        filtered_positions = list(range(1, len(filtered_regimes) + 1))

        vp = ax.violinplot(filtered_data, positions=filtered_positions, widths=0.6, showmeans=False, showmedians=False, showextrema=False)
        for i, (pc, color) in enumerate(zip(vp['bodies'], filtered_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.0)

        bp = ax.boxplot(filtered_data, positions=filtered_positions, tick_labels=filtered_regimes, widths=0.25, patch_artist=True,
                        showfliers=False, showcaps=True, boxprops=dict(linewidth=1.2),
                        whiskerprops=dict(linewidth=1.0), medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp["boxes"], filtered_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')

        ax.axhline(0, linestyle="--", linewidth=1.0, color="gray", alpha=0.9)
        ax.set_ylabel(r"$\Delta$ accuracy (agentic $-$ zeroshot)")
        fig.tight_layout()
        fig.savefig(out_regime_box, dpi=300)
        plt.close(fig)
        print(f"✓ Saved: {out_regime_box}")


def plot_cost_frontier(
    csv_in: Path = Path("analysis/agent_query_analysis_msg_limits.csv"),
    out_points: Path = Path("analysis/comprehensive/tables/cost_frontier_points_with_cached_tokens.csv"),
    out_fig: Path = Path("analysis/comprehensive/plots/fig1_frontier_with_cached_tokens.png")
):
    """
    Accuracy–Cost/Compute Frontier using effective tokens (including cached tokens).

    X-axis: effective_tokens_per_sample (log scale)
    Y-axis: accuracy
    """
    df = pd.read_csv(csv_in)

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()

    # Filter to exclude simple tasks (keep regular and no-offline tasks)
    if "task" in df.columns:
        df = df[~df["task"].str.contains("simple", case=False, na=False)].copy()
        print(f"Filtered to exclude simple tasks: {len(df)} rows remaining")

    # Add cached tokens
    df["effective_tokens"] = (
        df["total_tokens"]
        + df.get("cache_read_tokens", 0)
        + df.get("cache_write_tokens", 0)
    )

    # Aggregate points
    rows = []
    for (model, ml), g in df.groupby(["model", "message_limit"]):
        ml = int(ml)
        samples_total = g["num_samples"].sum()
        eff_tokens_sum = g["effective_tokens"].sum()
        tokens_sum = g["total_tokens"].sum()
        accuracy = g["accuracy"].mean() if "accuracy" in g.columns else 0.0

        rows.append({
            "model": model,
            "message_limit": ml,
            "num_runs": int(len(g)),
            "samples_total": float(samples_total),
            "total_tokens_sum": float(tokens_sum),
            "effective_tokens_sum": float(eff_tokens_sum),
            "tokens_per_sample": float(tokens_sum / samples_total),
            "effective_tokens_per_sample": float(eff_tokens_sum / samples_total),
            "avg_tokens_per_run": float(g["total_tokens"].mean()),
            "avg_effective_tokens_per_run": float(g["effective_tokens"].mean()),
            "accuracy": float(accuracy),
        })

    pts = pd.DataFrame(rows).sort_values(["model", "message_limit"])

    # Ensure output directories exist
    out_points.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    pts.to_csv(out_points, index=False)

    def _pretty_model_name(model: str) -> str:
        mapping = {
            "claude-opus-4-5@20251101": "Claude Opus 4.5",
            "claude-sonnet-4-5@20250929": "Claude Sonnet 4.5",
            "claude-haiku-4-5@20251001": "Claude Haiku 4.5",
            "gemini-3-pro-preview": "Gemini 3 Pro Preview",
            "gemini-3-flash-preview": "Gemini 3 Flash Preview",
            "gemini-2.5-pro": "Gemini 2.5 Pro",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gpt-5.2-2025-12-11": "GPT-5.2",
            "gpt-5.1-2025-11-13": "GPT-5.1",
            "gpt-5-mini-2025-08-07": "GPT-5 Mini",
            "gpt-5-nano-2025-08-07": "GPT-5 Nano",
        }
        return mapping.get(model, model)

    def get_model_family(model):
        model_lower = model.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            return "Anthropic"
        elif "gpt" in model_lower or "openai" in model_lower:
            return "OpenAI"
        elif "gemini" in model_lower or "google" in model_lower:
            return "Google"
        return "Other"

    def get_gradient_colors(base_hex: str, n: int) -> list:
        """Generate n shades from dark to light based on base color."""
        from matplotlib.colors import to_rgb, to_hex
        r, g, b = to_rgb(base_hex)
        colors = []
        for i in range(n):
            factor = 0.7 + (0.3 * i / max(1, n - 1))
            colors.append(to_hex((r * factor, g * factor, b * factor)))
        return colors

    # Group models by family and assign colors
    model_colors = {}
    for family, base_color in FAMILY_COLORS.items():
        family_models = sorted([m for m in pts["model"].unique() if get_model_family(m) == family])
        if family_models:
            shades = get_gradient_colors(base_color, len(family_models))
            for model, color in zip(family_models, shades):
                model_colors[model] = color

    linestyles = ['-', '--', '-.', ':']
    marker_map = {15: "o", 30: "s", 50: "^"}

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    for idx, (model, g) in enumerate(pts.groupby("model")):
        g = g.sort_values("message_limit")
        model_short = model.split("/")[-1]
        label = _pretty_model_name(model_short)
        color = model_colors.get(model, "#BBBBBB")
        family = get_model_family(model)

        family_models = sorted([m for m in pts["model"].unique() if get_model_family(m) == family])
        model_idx = family_models.index(model) if model in family_models else 0
        linestyle = linestyles[model_idx % len(linestyles)]

        ax.plot(
            g["effective_tokens_per_sample"],
            g["accuracy"],
            linewidth=0.5,
            label=label,
            color=color,
            linestyle=linestyle,
        )
        for _, r in g.iterrows():
            ax.plot(
                r["effective_tokens_per_sample"],
                r["accuracy"],
                marker=marker_map[int(r["message_limit"])],
                markersize=6,
                color=color,
                markeredgewidth=0.05,
                markeredgecolor='white',
            )

    ax.set_xscale("log")
    ax.set_xlabel("Tokens per sample")
    ax.set_ylabel("Accuracy")

    x_ticks = [3e4, 5e4, 1e5, 2e5, 4e5]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['30k', '50k', '100k', '200k', '400k'])

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, label="15"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=6, label="30"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=6, label="50"),
    ]
    ax.legend(
        handles=marker_handles,
        title="Message limit",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        ncol=1,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

    fig.subplots_adjust(top=0.78)
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"✓ Saved: {out_fig}")


def plot_taskfamily_efficiency_frontier(
    csv_in: Path = Path("analysis/agent_query_analysis_msg_limits.csv"),
    out_points: Path = Path("analysis/comprehensive/tables/taskfamily_agent_vs_zeroshot_efficiency_points.csv"),
    out_fig: Path = Path("analysis/comprehensive/plots/fig_taskfamily_agent_vs_zeroshot_efficiency_frontier.png"),
    exclude_incomplete: bool = False
):
    """
    Task-family agentic-vs-zeroshot efficiency frontier (Awards / Citations / Faculty / SOTA).

    - Aggregate within each family using sample-weighted accuracy and token-per-sample
    - Family detection by substring matching on task name
    """
    def task_family(task: str) -> str:
        t = task.lower()
        if "award" in t:
            return "Awards"
        if "citation" in t or "citations" in t:
            return "Citations"
        if "faculty" in t or "professor" in t or "pi_" in t:
            return "Faculty"
        if "sota" in t or "leaderboard" in t or "benchmark" in t:
            return "SOTA"
        return "Other"

    df = pd.read_csv(csv_in)
    required = {"task","model","message_limit","num_samples","total_tokens","accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "status" in df.columns:
        df = df[df["status"]=="success"].copy()
    df = df[~df["task"].astype(str).str.contains("no_offline_prompt", regex=False)].copy()

    # Filter out incomplete tasks if toggle is enabled
    if exclude_incomplete:
        try:
            summary_dfs = []
            for msg_limit in [15, 30, 50]:
                summary_path = Path(f"logs_msg{msg_limit}_summary.csv")
                if summary_path.exists():
                    summary_df = pd.read_csv(summary_path)
                    if "samples_hit_limit" in summary_df.columns and "total_samples" in summary_df.columns:
                        summary_df["all_hit_limit"] = (
                            summary_df["total_samples"] == summary_df["samples_hit_limit"]
                        ) & (summary_df["total_samples"] > 0)
                        incomplete = summary_df[summary_df["all_hit_limit"]][["task_name", "model"]].copy()
                        incomplete.rename(columns={"task_name": "task"}, inplace=True)
                        summary_dfs.append(incomplete)

            if summary_dfs:
                all_incomplete = pd.concat(summary_dfs, ignore_index=True).drop_duplicates()
                df["is_incomplete"] = df.apply(
                    lambda row: ((all_incomplete["task"] == row["task"]) &
                                (all_incomplete["model"] == row["model"])).any(),
                    axis=1
                )
                n_excluded = df["is_incomplete"].sum()
                if n_excluded > 0:
                    print(f"ℹ️ Excluding {n_excluded} incomplete task-model combinations (all samples hit limit)")
                    df = df[~df["is_incomplete"]].copy()
        except Exception as e:
            print(f"⚠️ Could not load hit_limit information: {e}")

    df["effective_tokens"] = df["total_tokens"] + df.get("cache_read_tokens",0) + df.get("cache_write_tokens",0)
    df["setting"] = np.where(df["task"].astype(str).str.contains("simple", regex=False), "zeroshot", "agentic")
    df["family"] = df["task"].astype(str).apply(task_family)

    families = ["Awards","Citations","Faculty","SOTA"]
    df = df[df["family"].isin(families)].copy()

    def wavg(x: pd.Series, w: pd.Series) -> float:
        return float(np.average(x, weights=w))

    def eff_tokens_per_sample(g: pd.DataFrame) -> float:
        return float(g["effective_tokens"].sum() / g["num_samples"].sum())

    zs = (df[df["setting"]=="zeroshot"]
          .groupby(["family"], as_index=False)
          .apply(lambda g: pd.Series({
              "zs_tokens_per_sample": eff_tokens_per_sample(g),
              "zs_accuracy": wavg(g["accuracy"], g["num_samples"]),
              "zs_samples": float(g["num_samples"].sum()),
          }))
          .reset_index(drop=True))

    ag = (df[df["setting"]=="agentic"]
          .groupby(["family","message_limit"], as_index=False)
          .apply(lambda g: pd.Series({
              "ag_tokens_per_sample": eff_tokens_per_sample(g),
              "ag_accuracy": wavg(g["accuracy"], g["num_samples"]),
              "ag_samples": float(g["num_samples"].sum()),
          }))
          .reset_index(drop=True))

    pts = ag.merge(zs, on="family", how="inner")
    pts["token_overhead_x"] = pts["ag_tokens_per_sample"] / pts["zs_tokens_per_sample"]
    pts["accuracy_gain"] = pts["ag_accuracy"] - pts["zs_accuracy"]

    # Ensure output directories exist
    out_points.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    pts.sort_values(["family","message_limit"]).to_csv(out_points, index=False)

    marker_map = {15:"o", 30:"s", 50:"^"}

    # Task family colors (colorblind-safe)
    fam_color = {
        "Awards": "#EE6677",
        "Citations": "#4477AA",
        "Faculty": "#228833",
        "SOTA": "#CCBB44",
    }

    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    # Draw connecting lines for each family
    for fam in families:
        g = pts[pts["family"]==fam].sort_values("message_limit")
        ax.plot(g["token_overhead_x"], g["accuracy_gain"],
                linewidth=0.8, alpha=0.7, color=fam_color[fam])

    # Draw markers
    for ml, mk in marker_map.items():
        g = pts[pts["message_limit"]==ml]
        ax.scatter(g["token_overhead_x"], g["accuracy_gain"],
                   marker=mk, s=38, alpha=0.9,
                   c=[fam_color[f] for f in g["family"]], edgecolors="none")

    ax.set_xscale("log")
    ax.set_xlim(left=19)
    x_ticks = [20, 50, 100, 200]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['20', '50', '100', '200'])

    ax.axhline(0.0, linewidth=0.5, linestyle="--", color='black', alpha=0.5)
    ax.axvline(1.0, linewidth=0.8, linestyle="--", color='gray', alpha=0.5)

    ax.set_xlabel("Token overhead vs zeroshot (×, log scale)")
    ax.set_ylabel("Accuracy gain vs zeroshot")

    from matplotlib.lines import Line2D
    fam_handles = [
        Line2D([0],[0], color=fam_color[f], linewidth=0.8, label=f)
        for f in families
    ]
    fig.legend(handles=fam_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False, fontsize=8)

    ml_handles = [
        Line2D([0],[0], marker="o", linestyle="None", markersize=6, label="15"),
        Line2D([0],[0], marker="s", linestyle="None", markersize=6, label="30"),
        Line2D([0],[0], marker="^", linestyle="None", markersize=6, label="50"),
    ]
    ax.legend(handles=ml_handles, title="Message limit", frameon=False,
              fontsize=8, title_fontsize=8, ncol=1, loc="lower right",
              bbox_to_anchor=(1.05, 0.3))

    fig.subplots_adjust(top=0.82, bottom=0.15)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_fig}")


def generate_efficiency_frontier_plots() -> None:
    """Generate all efficiency frontier plots."""
    print("\n" + "=" * 60)
    print("GENERATING EFFICIENCY FRONTIER PLOTS")
    print("=" * 60 + "\n")

    try:
        plot_agent_vs_zeroshot_efficiency_frontier()
    except Exception as e:
        print(f"  ⚠️ Failed to generate agent vs zeroshot efficiency frontier: {e}")

    try:
        plot_agent_vs_zeroshot_performance()
    except Exception as e:
        print(f"  ⚠️ Failed to generate agent vs zeroshot performance plots: {e}")

    try:
        plot_cost_frontier()
    except Exception as e:
        print(f"  ⚠️ Failed to generate cost frontier: {e}")

    try:
        plot_taskfamily_efficiency_frontier()
    except Exception as e:
        print(f"  ⚠️ Failed to generate task family efficiency frontier: {e}")

    print("\n✅ Efficiency frontier plots complete!")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all plots for the analysis.

    Args:
        df: Loaded DataFrame with all experimental data
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("=" * 60 + "\n")

    # Benchmark performance plots
    plot_overall_performance(df, output_dir)
    plot_task_breakdown(df, output_dir)
    plot_ablation_scatter(df, output_dir)
    plot_scaling_lines(df, output_dir)
    plot_scaling_by_family(df, output_dir)
    plot_model_family_comparison(df, output_dir)
    plot_hit_limit_analysis(df, output_dir)
    plot_simple_vs_agentic(df, output_dir)
    plot_scaling_gain_waterfall(df, output_dir)
    plot_task_difficulty_ranking(df, output_dir)
    plot_post_cutoff_comparison(df, output_dir)

    # Combined multi-panel figures
    plot_combined_three_panel(df, output_dir)
    plot_combined_three_panel_with_violin(df, output_dir)
    plot_combined_three_panel_with_waterfall(df, output_dir)

    print("\n✅ All plots generated successfully!")
    print(f"   Output directory: {output_dir / 'plots'}")


def generate_agent_behavior_plots(output_dir: Path, csv_dir: Path = Path('.')) -> None:
    """Generate all agent behavior analysis plots.

    Args:
        output_dir: Directory to save plots
        csv_dir: Directory containing CSV data files
    """
    print("\n" + "=" * 60)
    print("GENERATING AGENT BEHAVIOR ANALYSIS PLOTS")
    print("=" * 60 + "\n")

    try:
        plot_strata_distribution(output_dir, csv_dir / 'aba_stratum_counts.csv')
    except FileNotFoundError:
        print("  ⚠️ Skipping strata distribution (CSV not found)")

    plot_outcomes_by_limit(output_dir)

    try:
        plot_failure_taxonomy(output_dir, csv_dir / 'aba_failure_taxonomy.csv')
    except FileNotFoundError:
        print("  ⚠️ Skipping failure taxonomy (CSV not found)")

    try:
        plot_nonconvergence(output_dir, csv_dir / 'aba_nonconvergence.csv')
    except FileNotFoundError:
        print("  ⚠️ Skipping non-convergence analysis (CSV not found)")

    try:
        plot_bottlenecks(output_dir, csv_dir / 'aba_bottlenecks.csv')
    except FileNotFoundError:
        print("  ⚠️ Skipping bottleneck analysis (CSV not found)")

    plot_correct_flags(output_dir)

    print("\n✅ All agent behavior plots generated successfully!")
    print(f"   Output directory: {output_dir / 'plots'}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for standalone plot generation.

    Usage:
        python plots_new.py                        # Generate all plots
        python plots_new.py --benchmark-only       # Only benchmark plots
        python plots_new.py --agent-behavior-only  # Only agent behavior plots
    """
    import argparse
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data_loader import load_all_data

    parser = argparse.ArgumentParser(
        description="Generate all plots for Proof-of-Time analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only generate benchmark plots (skip agent behavior plots)"
    )
    parser.add_argument(
        "--agent-behavior-only",
        action="store_true",
        help="Only generate agent behavior plots (skip benchmark plots)"
    )
    parser.add_argument(
        "--efficiency-only",
        action="store_true",
        help="Only generate efficiency frontier plots"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for plots (default: current directory)"
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing CSV data files for agent behavior plots"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    csv_dir = args.csv_dir

    print("\n" + "=" * 60)
    print("PROOF OF TIME: PLOT GENERATION")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"CSV directory: {csv_dir}")
    print(f"Titles enabled: {INCLUDE_TITLES}")
    print(f"Axis labels enabled: {INCLUDE_AXIS_LABELS}")

    # Generate benchmark plots (requires loading data)
    if not args.agent_behavior_only:
        print("\n📁 Loading benchmark data...")
        try:
            df = load_all_data()
            print(f"   Loaded {len(df):,} rows")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return 1

        print("\n📊 Generating benchmark plots...")
        try:
            generate_all_plots(df, output_dir)
        except Exception as e:
            print(f"❌ Failed to generate benchmark plots: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n⏭️  Skipping benchmark plots (--agent-behavior-only)")

    # Generate agent behavior plots
    if not args.benchmark_only and not args.efficiency_only:
        print("\n📈 Generating agent behavior plots...")
        try:
            generate_agent_behavior_plots(output_dir, csv_dir)
        except Exception as e:
            print(f"❌ Failed to generate agent behavior plots: {e}")
            import traceback
            traceback.print_exc()
            return 1
    elif args.efficiency_only:
        print("\n⏭️  Skipping agent behavior plots (--efficiency-only)")
    else:
        print("\n⏭️  Skipping agent behavior plots (--benchmark-only)")

    # Generate efficiency frontier plots
    if args.efficiency_only or (not args.benchmark_only and not args.agent_behavior_only):
        print("\n📊 Generating efficiency frontier plots...")
        try:
            generate_efficiency_frontier_plots()
        except Exception as e:
            print(f"❌ Failed to generate efficiency frontier plots: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n⏭️  Skipping efficiency frontier plots")

    # Done
    print("\n" + "=" * 60)
    print(f"✅ DONE! Plots saved to: {output_dir / 'plots'}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
