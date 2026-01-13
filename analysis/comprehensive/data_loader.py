"""Data loading and normalization for Proof-of-Time comprehensive analysis.

This module provides:
- Task normalization (base_task, variant, family)
- Cross-run merge (msg15/msg30/msg50)
- Model family extraction
- Basic filtering (e.g., exclude Gemini 3 preview rows)

Note:
    CSV files are expected at the root of the project with columns:
    task_name, model, accuracy, mean_score, total_samples, samples_hit_limit, ...
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define paths to the CSV files (absolute paths from project root)
# Using the new summary files with updated results
CSV_FILES: Dict[int, Path] = {
    15: PROJECT_ROOT / "logs_msg15_summary.csv",
    30: PROJECT_ROOT / "logs_msg30_summary.csv",
    50: PROJECT_ROOT / "logs_msg50_summary.csv",
}

# Models to exclude from plots/tables.
# Note: With the new results, we include all models (Gemini 3, Gemini 2.5 Flash now have full runs)
EXCLUDED_MODEL_SUBSTRINGS: set[str] = {
    # No exclusions in current analysis - all models have complete data
}

# Model family mapping for analysis
MODEL_FAMILIES: Dict[str, str] = {
    "claude": "Anthropic",
    "gpt-5": "OpenAI",
    "gemini": "Google",
}

# Task family descriptions for documentation
TASK_FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "citation": "Citation count prediction (MCQ, ranking, bucket)",
    "award_historical": "Peer review award tier classification (historical, combined emnlp_awards + emnlp_historical_awards)",
    "peer_review_award": "Peer review award tier classification (post-cutoff, combined ACL 2025 + EMNLP 2025)",
    "emnlp_awards": "EMNLP paper award tier classification (historical)",
    "emnlp_awards_acl2025": "ACL 2025 paper award tier classification (post-cutoff)",
    "emnlp_awards_emnlp2025": "EMNLP 2025 paper award tier classification (post-cutoff)",
    "emnlp_awards_post_cutoff": "Peer review award tier classification (post-cutoff, combined)",
    "emnlp_historical": "Historical EMNLP paper classification",
    "faculty": "Faculty research prediction tasks",
    "sota": "SOTA benchmark bucket prediction",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedTask:
    """Normalized task name components.

    Attributes:
        base_task: Core task name without variant suffixes
        variant: One of the three experimental modes:
            - "simple_task": Mode 1 - Zero-shot (no tools, no sandbox)
            - "no_offline_prompt": Mode 2 - ReAct agent (tools + sandbox, no agentic prompt)
            - "offline_prompt": Mode 3 - ReAct + Agentic Prompt (tools + sandbox + offline preamble)
        family: Task family (citation, faculty, emnlp_awards, etc.)
    """

    base_task: str
    variant: str  # "offline_prompt", "no_offline_prompt", "simple_task"
    family: str   # e.g., "citation", "faculty", "emnlp_awards"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning 0.0 on division by zero or NaN."""
    if denominator == 0 or math.isnan(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def normalize_task_name(task_name: str) -> NormalizedTask:
    """Normalize a raw Inspect task name into components.

    Args:
        task_name: Raw task name from CSV (e.g., "citation_multiple_choice_no_offline_prompt")

    Returns:
        NormalizedTask with base_task, variant, and family extracted

    Examples:
        >>> normalize_task_name("citation_bucket_prediction")
        NormalizedTask(base_task='citation_bucket_prediction', variant='offline_prompt', family='citation')
        
        >>> normalize_task_name("citation_bucket_prediction_no_offline_prompt")
        NormalizedTask(base_task='citation_bucket_prediction', variant='no_offline_prompt', family='citation')
        
        >>> normalize_task_name("emnlp_awards_mcq_simple_task")
        NormalizedTask(base_task='emnlp_awards_mcq', variant='simple_task', family='emnlp_awards')
    """
    name = task_name.strip()

    # Determine variant by suffix
    variant = "offline_prompt"  # default (standard condition)
    if name.endswith("_simple_task"):
        variant = "simple_task"
        name = name[: -len("_simple_task")]
    elif name.endswith("_no_offline_prompt_task"):
        variant = "no_offline_prompt"
        name = name[: -len("_no_offline_prompt_task")]
    elif name.endswith("_no_offline_prompt"):
        variant = "no_offline_prompt"
        name = name[: -len("_no_offline_prompt")]

    # Remove trailing `_task` suffix if present
    if name.endswith("_task"):
        name = name[: -len("_task")]

    base_task = name if name else task_name.strip()

    # Extract family from base_task
    family = _extract_family(base_task)

    return NormalizedTask(base_task=base_task, variant=variant, family=family)


def _extract_family(base_task: str) -> str:
    """Extract task family from base_task name.

    Args:
        base_task: Normalized base task name

    Returns:
        Task family string (citation, faculty, emnlp_awards, emnlp_historical, sota, other)
    """
    if base_task.startswith("citation_"):
        return "citation"
    elif base_task.startswith("faculty_"):
        return "faculty"
    elif base_task.startswith("emnlp_"):
        if "awards" in base_task:
            # Distinguish between historical and post-cutoff award tasks
            if "acl2025" in base_task:
                return "emnlp_awards_acl2025"
            elif "emnlp2025" in base_task:
                return "emnlp_awards_emnlp2025"
            return "emnlp_awards"
        elif "historical" in base_task:
            # Check if this is a historical_awards task
            if "historical_awards" in base_task or "historical-awards" in base_task:
                # Same logic as awards tasks
                if "acl2025" in base_task:
                    return "emnlp_awards_acl2025"
                elif "emnlp2025" in base_task:
                    return "emnlp_awards_emnlp2025"
                return "emnlp_awards"
            return "emnlp_historical"
        return "emnlp"
    elif base_task.startswith("sota_"):
        return "sota"
    return "other"


def get_combined_task_group(base_task: str, model_name: str) -> str:
    """Get the combined task group for evaluation purposes.

    Award tasks should be combined for evaluation:

    For most models (Claude, GPT-5, Gemini 2.5):
    - Historical: emnlp_awards_mcq + emnlp_historical_awards_mcq (2021-2024)
    - Post-cutoff: All ACL 2025 + EMNLP 2025 award tasks

    For Gemini 3 models (gemini-3-pro-preview, gemini-3-flash-preview):
    - Historical: Base tasks (2021-2024) + ACL 2025 tasks (trained after July 2025)
    - Post-cutoff: Only EMNLP 2025 tasks (published November 2025)

    Args:
        base_task: Normalized base task name
        model_name: Full model identifier (e.g., "google/gemini-3-pro-preview")

    Returns:
        Combined task group identifier:
        - "emnlp_awards_historical_combined" for historical tasks
        - "emnlp_awards_post_cutoff_combined" for post-cutoff tasks
        - base_task (unchanged) for non-award tasks
    """
    # Detect Gemini 3 models (trained after ACL 2025)
    is_gemini_3 = "gemini-3" in model_name.lower() or "gemini-exp-1206" in model_name.lower()

    # For award/historical_award tasks, return combined group
    if "awards" in base_task and base_task.startswith("emnlp_"):
        # Check which temporal variant
        if "acl2025" in base_task:
            # ACL 2025 (July 2025): Historical for Gemini 3, post-cutoff for others
            if is_gemini_3:
                return "emnlp_awards_historical_combined"
            else:
                return "emnlp_awards_post_cutoff_combined"
        elif "emnlp2025" in base_task:
            # EMNLP 2025 (November 2025): Always post-cutoff for all models
            return "emnlp_awards_post_cutoff_combined"
        else:
            # Base awards tasks (2021-2024): Always historical for all models
            return "emnlp_awards_historical_combined"

    # For non-award tasks, return the base task as-is (no combining)
    return base_task


def is_historical_task(base_task: str, model_name: str) -> bool:
    """Determine if a task is historical (within training cutoff) for a given model.

    Most models (Claude, GPT-5, Gemini 2.5):
    - Historical: 2021-2024 data
    - Post-cutoff: ACL 2025 + EMNLP 2025

    Gemini 3 models (trained after July 2025):
    - Historical: 2021-2024 data + ACL 2025
    - Post-cutoff: EMNLP 2025 only

    Args:
        base_task: Normalized base task name
        model_name: Full model identifier

    Returns:
        True if task is within model's training cutoff, False otherwise
    """
    # Detect Gemini 3 models (trained after ACL 2025)
    is_gemini_3 = "gemini-3" in model_name.lower() or "gemini-exp-1206" in model_name.lower()

    if "awards" in base_task and base_task.startswith("emnlp_"):
        # Award tasks have model-specific cutoffs
        if "emnlp2025" in base_task:
            # EMNLP 2025 (November 2025): Post-cutoff for ALL models
            return False
        elif "acl2025" in base_task:
            # ACL 2025 (July 2025): Historical for Gemini 3, post-cutoff for others
            return is_gemini_3
        else:
            # Base awards tasks (2021-2024): Historical for ALL models
            return True

    # All non-award tasks are historical (within training cutoff)
    return True


def extract_model_family(model_name: str) -> str:
    """Extract model family from full model identifier.

    Args:
        model_name: Full model name (e.g., "anthropic/vertex/claude-opus-4-5@20251101")

    Returns:
        Model family string (Anthropic, OpenAI, Google, Other)
    """
    model_lower = model_name.lower()
    for pattern, family in MODEL_FAMILIES.items():
        if pattern in model_lower:
            return family
    return "Other"


def extract_short_model(model_name: str) -> str:
    """Extract clean short model name for display.

    Args:
        model_name: Full model name (e.g., "anthropic/vertex/claude-opus-4-5@20251101")

    Returns:
        Short display name (e.g., "claude-opus-4-5")
    """
    # Take last path component and remove version suffix
    short = model_name.split("/")[-1]
    # Remove @version suffix if present
    if "@" in short:
        short = short.split("@")[0]
    return short


def _is_excluded_model(model: str) -> bool:
    """Check if a model should be excluded from analysis.

    Args:
        model: Full model name

    Returns:
        True if the model should be excluded
    """
    model_l = str(model).lower()
    return any(substr in model_l for substr in EXCLUDED_MODEL_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Main data loading
# ---------------------------------------------------------------------------

def load_all_data(csv_paths: Optional[Dict[int, Path]] = None) -> pd.DataFrame:
    """Load and merge all summary CSVs into a single DataFrame.

    Args:
        csv_paths: Optional dict mapping message_limit -> Path to CSV.
                   Defaults to CSV_FILES if not provided.

    Returns:
        Combined DataFrame with normalized columns:
        - task_name: Original task name
        - model: Full model identifier
        - accuracy: Task accuracy
        - total_samples: Number of samples evaluated
        - samples_hit_limit: Samples that hit message limit
        - message_limit: Max messages allowed (15, 30, 50)
        - base_task: Normalized task name
        - variant: Task variant (offline_prompt, no_offline_prompt, simple_task)
        - task_family: Task family (citation, faculty, etc.)
        - short_model: Display-friendly model name
        - model_family: Model family (Anthropic, OpenAI, Google)
        - hit_limit_rate: Fraction of samples that hit message limit

    Raises:
        ValueError: If no data could be loaded from any CSV
    """
    if csv_paths is None:
        csv_paths = CSV_FILES

    dfs: List[pd.DataFrame] = []

    for limit, path in csv_paths.items():
        if not path.exists():
            print(f"⚠️  Warning: {path} not found. Skipping.")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️  Error reading {path}: {e}")
            continue

        # Validate required columns
        required = {"task_name", "model", "accuracy"}
        missing = required.difference(df.columns)
        if missing:
            print(f"⚠️  Warning: {path} missing columns {missing}. Skipping.")
            continue

        # Add message_limit from filename/dict key
        df["message_limit"] = limit

        # Normalize task names
        normalized = df["task_name"].apply(normalize_task_name)
        df["base_task"] = normalized.apply(lambda x: x.base_task)
        df["variant"] = normalized.apply(lambda x: x.variant)
        df["task_family"] = normalized.apply(lambda x: x.family)

        # Extract model metadata
        df["short_model"] = df["model"].apply(extract_short_model)
        df["model_family"] = df["model"].apply(extract_model_family)

        # Add combined task group and historical flag (model-dependent)
        df["combined_task_group"] = df.apply(
            lambda row: get_combined_task_group(row["base_task"], row["model"]),
            axis=1
        )
        df["is_historical"] = df.apply(
            lambda row: is_historical_task(row["base_task"], row["model"]),
            axis=1
        )

        # Ensure numeric types
        numeric_cols = ["accuracy", "total_samples", "samples_hit_limit", "mean_score"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0 if col in {"total_samples", "samples_hit_limit"} else pd.NA

        # Calculate hit rate
        df["hit_limit_rate"] = df.apply(
            lambda row: _safe_div(
                float(row.get("samples_hit_limit", 0)),
                float(row.get("total_samples", 1))
            ),
            axis=1,
        )

        dfs.append(df)
        print(f"✓ Loaded {len(df)} rows from {path}")

    if not dfs:
        raise ValueError("No data loaded from any CSV.")

    combined = pd.concat(dfs, ignore_index=True)

    # Filter excluded models
    n_before = len(combined)
    combined = combined[~combined["model"].apply(_is_excluded_model)].reset_index(drop=True)
    n_filtered = n_before - len(combined)
    if n_filtered > 0:
        print(f"✓ Filtered out {n_filtered} rows from excluded models")

    return combined


def get_task_family_description(family: str) -> str:
    """Get human-readable description for a task family.

    Args:
        family: Task family name

    Returns:
        Description string or the family name if no description exists
    """
    return TASK_FAMILY_DESCRIPTIONS.get(family, family)


def get_unique_tasks(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique base tasks in the dataset.

    Args:
        df: Loaded DataFrame

    Returns:
        Sorted list of unique base_task values
    """
    return sorted(df["base_task"].unique())


def get_unique_models(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique short model names.

    Args:
        df: Loaded DataFrame

    Returns:
        Sorted list of unique short_model values
    """
    return sorted(df["short_model"].unique())


def get_combined_evaluation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows by combined_task_group for evaluation.

    For award tasks that should be evaluated together (e.g., emnlp_awards_mcq +
    emnlp_historical_awards_mcq), this combines the samples and recalculates metrics.

    Args:
        df: DataFrame from load_all_data()

    Returns:
        DataFrame with combined task groups, aggregated by:
        - model
        - message_limit
        - variant
        - combined_task_group
    """
    # Group by the dimensions that should remain separate
    group_cols = ["model", "short_model", "model_family", "message_limit", "variant", "combined_task_group", "is_historical"]

    # Aggregate numeric columns
    agg_dict = {
        "total_samples": "sum",
        "samples_hit_limit": "sum",
        "mean_score": lambda x: np.nan if x.isna().all() else x.mean(),  # Average mean_score
    }

    # For accuracy, we need to recalculate based on total correct samples
    # If mean_score represents accuracy (0-1), we can weight by total_samples
    # Otherwise, use accuracy column directly
    if "accuracy" in df.columns:
        # Weighted average of accuracy by total_samples
        def weighted_accuracy(group):
            if group["total_samples"].sum() == 0:
                return 0.0
            return (group["accuracy"] * group["total_samples"]).sum() / group["total_samples"].sum()

        grouped = df.groupby(group_cols, as_index=False).apply(
            lambda g: pd.Series({
                "total_samples": g["total_samples"].sum(),
                "samples_hit_limit": g["samples_hit_limit"].sum(),
                "accuracy": weighted_accuracy(g),
                "mean_score": g["mean_score"].mean() if not g["mean_score"].isna().all() else np.nan,
            })
        )
    else:
        grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Recalculate hit_limit_rate
    grouped["hit_limit_rate"] = grouped.apply(
        lambda row: _safe_div(row["samples_hit_limit"], row["total_samples"]),
        axis=1
    )

    # Use combined_task_group as the base_task for consistency
    grouped["base_task"] = grouped["combined_task_group"]

    # Derive task_family from combined group
    def get_family_from_combined(combined_group):
        if "awards" in combined_group and combined_group.startswith("emnlp_"):
            if "historical" in combined_group:
                return "emnlp_awards"  # Historical awards
            else:
                return "emnlp_awards_post_cutoff"  # Post-cutoff awards
        return combined_group.split("_")[0] if "_" in combined_group else combined_group

    grouped["task_family"] = grouped["combined_task_group"].apply(get_family_from_combined)

    return grouped
