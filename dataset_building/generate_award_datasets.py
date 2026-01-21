#!/usr/bin/env python3

"""
Unified Award Dataset Generator

Generates all award-related MCQ datasets:
1. accepted_papers.csv - Sandbox data with all papers
2. pre-cutoff_mcq.jsonl - Historical award papers (pre-2025)
3. post-cutoff_mcq.jsonl - 2025 award papers (ACL/EMNLP/NAACL)

Consolidates functionality from:
- generate_mcq_dataset.py
- generate_historical_mcq.py
- generate_2025_awards_mcq.py
- generate_2025_historical_mcq.py
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd
from datasets import load_dataset


# Constants
DEFAULT_CHOICES = ["Findings", "Main", "Outstanding", "Best"]
TARGET_CONFERENCES = {"ACL", "EMNLP", "NAACL"}
TARGET_TRACKS = {"main", "findings"}
YEAR_CUTOFF = 2025
HF_DATASETS = [
    "AIM-Harvard/EMNLP-Accepted-Papers",
    "AIM-Harvard/ACL-Accepted-Papers",
    "AIM-Harvard/NAACL-Accepted-Papers",
]


# ============================================================================
# Text Cleaning and Normalization Utilities
# ============================================================================

def clean_title(raw: str | None) -> str:
    """Clean and extract title text from various formats (handles tex-math dicts)."""
    if not isinstance(raw, str):
        return ""
    stripped = raw.strip()
    if not stripped:
        return ""

    # Fix malformed UTF-8 sequences from latin-1 encoding
    stripped = stripped.replace("â\x80\x99", "'")
    stripped = stripped.replace("â\x80\x9c", '"')
    stripped = stripped.replace("â\x80\x9d", '"')
    stripped = stripped.replace("â", "'")

    # Handle dictionary format: {'tex-math': '...', '#text': 'actual title'}
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = ast.literal_eval(stripped)
        except Exception:
            data = None
        if isinstance(data, dict):
            text_val = str(data.get("#text", "")) if data.get("#text") else ""
            tex_val = str(data.get("tex-math", "")) if data.get("tex-math") else ""

            if tex_val and text_val:
                # Clean LaTeX commands from tex-math
                tex_clean = re.sub(r'\\text[a-z]*{', '', tex_val)
                tex_clean = re.sub(r'[{}]', '', tex_clean)
                tex_clean = re.sub(r'\\+', '', tex_clean)
                tex_clean = tex_clean.strip()

                # Insert tex content where double spaces appear
                if "  " in text_val:
                    result = re.sub(r'\s{2,}', f' {tex_clean} ', text_val, count=1)
                    return result.strip()
                else:
                    return f"{tex_clean} {text_val}".strip()
            elif text_val:
                return text_val.strip()
            elif tex_val:
                return tex_val.strip()

            # Fallback: concatenate all string values
            parts = [str(v).strip() for v in data.values() if isinstance(v, str)]
            if parts:
                return " ".join(parts)

    return stripped


def clean_abstract(raw: str | None) -> str:
    """Clean abstract text, handling various formats."""
    if not isinstance(raw, str):
        return ""
    stripped = raw.strip()
    if not stripped:
        return ""

    # Handle dictionary format
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = ast.literal_eval(stripped)
        except Exception:
            data = None
        if isinstance(data, dict):
            if "#text" in data and isinstance(data["#text"], str):
                return data["#text"].strip()
            parts = [str(v).strip() for v in data.values() if isinstance(v, str)]
            if parts:
                return " ".join(parts)

    return stripped


def normalize_key(text: str | None) -> str | None:
    """Normalize text to canonical form for matching."""
    if not isinstance(text, str):
        return None

    # Apply Unicode normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Handle common character encoding issues
    # Apostrophes and quotes
    normalized = normalized.replace("'", "'").replace("'", "'")
    normalized = normalized.replace(""", '"').replace(""", '"')
    normalized = normalized.replace("â", "'")
    normalized = re.sub(r'[\u2018-\u201b]', "'", normalized)

    # Dashes and hyphens
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = normalized.replace("Ð", "-")

    # Remove special characters and extra whitespace
    normalized = re.sub(r'[^\w\s\-\']', ' ', normalized)
    normalized = " ".join(normalized.lower().split())

    return normalized if normalized else None


def parse_year(text: str | None) -> int | None:
    """Extract year from text."""
    if not isinstance(text, str):
        return None
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def classify_category(category: str) -> str:
    """Classify award category into recognition tier."""
    lowered = category.lower()
    if "outstanding" in lowered:
        return "Outstanding"
    if "best" in lowered or "runner-up" in lowered or "runner up" in lowered:
        return "Best"
    if "findings" in lowered:
        return "Findings"
    return "Main"


# ============================================================================
# Data Loading
# ============================================================================

def load_huggingface_papers(min_year: int | None = None, max_year: int | None = None) -> pd.DataFrame:
    """Load papers from multiple HuggingFace datasets with optional year filtering."""
    print(f"Loading papers from {len(HF_DATASETS)} HuggingFace datasets...")

    all_dfs = []
    for hf_dataset in HF_DATASETS:
        print(f"  Loading {hf_dataset}...")
        try:
            # Load all splits from the dataset
            dataset_dict = load_dataset(hf_dataset)

            # Combine all splits into a single DataFrame
            for split_name, split_data in dataset_dict.items():
                split_df = split_data.to_pandas()
                split_df["source_split"] = split_name  # Add split name as a column
                split_df["source_dataset"] = hf_dataset  # Track which dataset it came from
                all_dfs.append(split_df)
        except Exception as e:
            print(f"  Warning: Failed to load {hf_dataset}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No datasets could be loaded")

    df = pd.concat(all_dfs, ignore_index=True)

    # Parse years
    df["year"] = df["source_split"].apply(parse_year)

    # Filter by year if specified
    if min_year is not None:
        df = df[df["year"] >= min_year]
    if max_year is not None:
        df = df[df["year"] <= max_year]

    print(f"Loaded {len(df)} papers from {len(all_dfs)} splits across {len(HF_DATASETS)} datasets")
    return df


def load_2025_papers(papers_2025_dir: Path) -> pd.DataFrame:
    """Load 2025 papers from CSV files."""
    print(f"Loading 2025 papers from {papers_2025_dir}")

    dfs = []
    for csv_file in papers_2025_dir.glob("*.csv"):
        # Try UTF-8 first, fall back to latin-1 if that fails
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin-1')

        # Extract conference and track from filename (e.g., "acl2025_main.csv" -> "ACL", "main")
        stem = csv_file.stem.lower()
        stem = stem.replace("2025", "").replace("_", " ").strip()
        parts = stem.split()

        # Identify conference (acl, emnlp, naacl, etc.)
        conf_name = parts[0].upper() if parts else "UNKNOWN"
        # Identify track (main, findings)
        track = parts[1] if len(parts) > 1 else "main"

        df["conference"] = conf_name
        df["paper_track"] = track  # Use different column name to avoid conflict
        df["year"] = 2025
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {papers_2025_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} papers from 2025")
    return combined


def load_excel_awards(xlsx_path: Path) -> pd.DataFrame:
    """Load award information from Excel workbook."""
    print(f"Loading award data from {xlsx_path}")

    xl = pd.ExcelFile(xlsx_path)
    dfs = []

    for sheet in xl.sheet_names:
        if sheet not in TARGET_CONFERENCES:
            continue
        df = xl.parse(sheet)
        df["sheet"] = sheet
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No target conference sheets found in {xlsx_path}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} award entries")
    return combined


def build_paper_lookup(papers_df: pd.DataFrame) -> Dict[str, Dict]:
    """Build lookup dictionary from papers DataFrame."""
    lookup: Dict[str, Dict] = {}

    for _, row in papers_df.iterrows():
        # Try multiple title variations
        title_raw = row.get("title", "")
        title_clean = clean_title(title_raw)

        # Create keys
        key_raw = normalize_key(title_raw)
        key_clean = normalize_key(title_clean)

        row_dict = row.to_dict()

        # Store under both keys
        if key_raw:
            lookup[key_raw] = row_dict
        if key_clean and key_clean != key_raw:
            lookup[key_clean] = row_dict

    return lookup


# ============================================================================
# Dataset Generation
# ============================================================================

@dataclass
class GenerationStats:
    """Track generation statistics."""
    generated: int = 0
    skipped_no_match: int = 0
    skipped_no_abstract: int = 0
    skipped_examples: List[Dict] = None

    def __post_init__(self):
        if self.skipped_examples is None:
            self.skipped_examples = []


def generate_award_mcq(
    awards_df: pd.DataFrame,
    papers_lookup: Dict[str, Dict],
    *,
    cutoff_year: int = YEAR_CUTOFF,
    include_pre_cutoff: bool = True,
    include_post_cutoff: bool = True,
    random_seed: int | None = None,
) -> tuple[List[Dict], GenerationStats]:
    """Generate MCQ records from award data."""

    rng = random.Random(random_seed)
    records: List[Dict] = []
    stats = GenerationStats()

    for _, award_row in awards_df.iterrows():
        # Extract year
        year = parse_year(str(award_row.get("conference", "")))

        # Filter by cutoff
        if year:
            if year < cutoff_year and not include_pre_cutoff:
                continue
            if year >= cutoff_year and not include_post_cutoff:
                continue

        # Find matching paper
        title_raw = award_row.get("title", "")
        title_clean = clean_title(title_raw)

        key_raw = normalize_key(title_raw)
        key_clean = normalize_key(title_clean)

        paper = papers_lookup.get(key_raw) or papers_lookup.get(key_clean)

        if not paper:
            stats.skipped_no_match += 1
            if len(stats.skipped_examples) < 10:
                stats.skipped_examples.append({
                    "title": title_raw,
                    "reason": "no match in papers",
                    "sheet": award_row.get("sheet", "unknown")
                })
            continue

        # Get abstract
        abstract = clean_abstract(paper.get("abstract"))
        if not abstract:
            stats.skipped_no_abstract += 1
            if len(stats.skipped_examples) < 10:
                stats.skipped_examples.append({
                    "title": title_raw,
                    "reason": "missing abstract",
                    "sheet": award_row.get("sheet", "unknown")
                })
            continue

        # Classify category
        category = str(award_row.get("category", "") or "")
        answer = classify_category(category)

        # Optionally include authors
        include_authors = False
        authors_line = None
        authors = paper.get("authors")
        if isinstance(authors, str) and authors.strip():
            include_authors = rng.random() < 0.5
            if include_authors:
                authors_line = f"Authors: {authors.strip()}"

        # Build context
        context_lines = [f"Title: {title_clean}"]
        context_lines.append(f"Abstract: {abstract}")
        if authors_line:
            context_lines.append(authors_line)

        # Metadata
        cutoff_period = "post_cutoff" if year and year >= cutoff_year else "pre_cutoff"
        metadata = {
            "conference": award_row.get("conference", ""),
            "category": category,
            "sheet": award_row.get("sheet", ""),
            "year": year,
            "cutoff_period": cutoff_period,
            "authors_included": include_authors,
        }

        # Create record
        record = {
            "question": "Which recognition tier (Findings/Main/Outstanding/Best) best fits this paper?",
            "choices": DEFAULT_CHOICES,
            "answer": answer,
            "context": "\n".join(context_lines),
            "metadata": metadata,
        }

        records.append(record)
        stats.generated += 1

    return records, stats


def generate_historical_mcq(
    papers_df: pd.DataFrame,
    awarded_titles: Set[str],
    *,
    samples_per_track: int = 30,
    random_seed: int | None = None,
    year_filter: str = "pre_cutoff",
    conference_filter: str | None = None,
) -> tuple[List[Dict], GenerationStats]:
    """Generate historical main/findings MCQ samples (excluding award winners).

    Args:
        papers_df: DataFrame containing papers
        awarded_titles: Set of normalized titles to exclude
        samples_per_track: Number of samples per track
        random_seed: Random seed for reproducibility
        year_filter: "pre_cutoff" (<2025), "post_cutoff" (>=2025), or "all"
        conference_filter: Optional conference name to filter (e.g., "ACL", "EMNLP")
    """

    rng = random.Random(random_seed)
    records: List[Dict] = []
    stats = GenerationStats()

    # Filter papers by year based on filter
    if year_filter == "pre_cutoff":
        papers_filtered = papers_df[papers_df["year"] < YEAR_CUTOFF].copy()
    elif year_filter == "post_cutoff":
        papers_filtered = papers_df[papers_df["year"] >= YEAR_CUTOFF].copy()
    else:  # "all"
        papers_filtered = papers_df.copy()

    # Filter by conference if specified
    if conference_filter:
        # Split on | to handle multiple conferences (e.g., "ACL|NAACL")
        conf_filters = [c.strip().lower() for c in conference_filter.split("|")]
        mask = pd.Series([False] * len(papers_filtered), index=papers_filtered.index)

        for conf_lower in conf_filters:
            mask |= (
                (papers_filtered["source_split"].str.lower().str.contains(conf_lower, na=False)) |
                (papers_filtered["conference"].str.lower().str.contains(conf_lower, na=False)) |
                (papers_filtered["tags"].str.lower().str.contains(conf_lower, na=False))
            )

        papers_filtered = papers_filtered[mask]

    # Exclude awarded papers
    papers_filtered["title_key"] = papers_filtered["title"].apply(lambda t: normalize_key(clean_title(t)))
    papers_filtered = papers_filtered[~papers_filtered["title_key"].isin(awarded_titles)]

    # Group by track (extracted from source_split column, paper_track, or tags)
    track_groups = defaultdict(list)
    for _, row in papers_filtered.iterrows():
        source_split = str(row.get("source_split", "")).lower()
        tags = str(row.get("tags", "")).lower()
        paper_track = str(row.get("paper_track", "")).lower()

        # Extract track from split name, paper_track column, or tags
        if "findings" in source_split or "findings" in paper_track or "findings" in tags:
            track = "findings"
        elif "main" in source_split or "main" in paper_track or "main" in tags:
            track = "main"
        else:
            continue
        track_groups[track].append(row)

    # Sample from each track
    for track, papers in track_groups.items():
        if len(papers) < samples_per_track:
            print(f"Warning: Only {len(papers)} {track} papers available, requested {samples_per_track}")
            sampled = papers
        else:
            sampled = rng.sample(papers, samples_per_track)

        for paper in sampled:
            abstract = clean_abstract(paper.get("abstract"))
            if not abstract:
                stats.skipped_no_abstract += 1
                continue

            title = clean_title(paper.get("title"))
            answer = "Findings" if track == "findings" else "Main"

            # Build context
            context_lines = [f"Title: {title}"]
            context_lines.append(f"Abstract: {abstract}")

            # Metadata
            metadata = {
                "track": track,
                "year": paper.get("year"),
                "cutoff_period": "pre_cutoff",
                "source": "historical_sampling",
            }

            # Create record
            record = {
                "question": "Which recognition tier (Findings/Main/Outstanding/Best) best fits this paper?",
                "choices": DEFAULT_CHOICES,
                "answer": answer,
                "context": "\n".join(context_lines),
                "metadata": metadata,
            }

            records.append(record)
            stats.generated += 1

    return records, stats


# ============================================================================
# Output Writing
# ============================================================================

def write_jsonl(records: Iterable[Dict], output_path: Path) -> None:
    """Write records to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output_path}")


def write_accepted_papers_csv(papers_df: pd.DataFrame, output_path: Path) -> None:
    """Write accepted papers CSV for sandbox with essential columns only.

    Columns: title, abstract, authors, tags
    tags format: "conference_name year track" (e.g., "emnlp 2022 main")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare simplified dataframe
    simplified_df = pd.DataFrame()

    # Title (clean it)
    simplified_df["title"] = papers_df["title"].apply(clean_title)

    # Abstract (use abstract or abs column, clean it)
    abstract_col = papers_df.get("abstract", papers_df.get("abs", pd.Series(dtype=str)))
    simplified_df["abstract"] = abstract_col.apply(clean_abstract)

    # Authors
    simplified_df["authors"] = papers_df.get("authors", "")

    # Tags: Format as "conference year track" (e.g., "emnlp 2022 main")
    def format_tag(row):
        # Extract conference from source_split or conference column
        source_split = str(row.get("source_split", "")).lower()
        conf = str(row.get("conference", "")).lower()
        paper_track = str(row.get("paper_track", "")).lower()

        # Try to extract conference name (acl, emnlp, naacl)
        conf_name = None
        for target_conf in ["emnlp", "acl", "naacl"]:
            if target_conf in source_split or target_conf in conf:
                conf_name = target_conf
                break

        # Extract year
        year = row.get("year", "")

        # Extract track (main, findings, outstanding, best)
        track = ""
        if "findings" in source_split or "findings" in paper_track:
            track = "findings"
        elif "main" in source_split or "main" in paper_track:
            track = "main"

        # Build tag string
        parts = []
        if conf_name:
            parts.append(conf_name)
        if year:
            parts.append(str(year))
        if track:
            parts.append(track)

        return " ".join(parts) if parts else ""

    simplified_df["tags"] = papers_df.apply(format_tag, axis=1)

    # Write to CSV
    simplified_df.to_csv(output_path, index=False)
    print(f"Wrote {len(simplified_df)} papers to {output_path}")


def print_stats(name: str, stats: GenerationStats) -> None:
    """Print generation statistics."""
    print(f"\n{name} Statistics:")
    print(f"  Generated: {stats.generated}")
    print(f"  Skipped (no match): {stats.skipped_no_match}")
    print(f"  Skipped (no abstract): {stats.skipped_no_abstract}")
    if stats.skipped_examples:
        print(f"  Example skips (showing {len(stats.skipped_examples)}):")
        for ex in stats.skipped_examples[:5]:
            print(f"    - {ex['sheet']}: {ex['title'][:50]}... ({ex['reason']})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all award-related MCQ datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all datasets
  python generate_award_datasets.py

  # Custom paths
  python generate_award_datasets.py \\
    --excel data/pot-best-papers-updated.xlsx \\
    --papers-2025-dir data/2025papers \\
    --output-dir benchmarks/award_react
        """
    )

    # Input arguments
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("data/pot-best-papers-updated.xlsx"),
        help="Excel file with award information"
    )
    parser.add_argument(
        "--papers-2025-dir",
        type=Path,
        default=Path("data/2025papers"),
        help="Directory containing 2025 paper CSVs"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/award_react"),
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--sandbox-csv",
        type=Path,
        help="Path for accepted_papers.csv (default: <output-dir>/sandbox/data/accepted_papers.csv)"
    )
    parser.add_argument(
        "--pre-cutoff-jsonl",
        type=Path,
        help="Path for pre-cutoff MCQ dataset (default: <output-dir>/pre-cutoff_mcq.jsonl)"
    )
    parser.add_argument(
        "--post-cutoff-acl-naacl-jsonl",
        type=Path,
        help="Path for post-cutoff ACL/NAACL MCQ dataset (default: <output-dir>/post-cutoff_acl_naacl.jsonl)"
    )
    parser.add_argument(
        "--post-cutoff-emnlp-jsonl",
        type=Path,
        help="Path for post-cutoff EMNLP MCQ dataset (default: <output-dir>/post-cutoff_emnlp.jsonl)"
    )

    # Generation parameters
    parser.add_argument(
        "--historical-samples",
        type=int,
        default=30,
        help="Number of historical main/findings papers to sample per track"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set default output paths
    output_dir = args.output_dir
    sandbox_csv = args.sandbox_csv or output_dir / "sandbox" / "data" / "accepted_papers.csv"
    pre_cutoff_jsonl = args.pre_cutoff_jsonl or output_dir / "pre-cutoff_mcq.jsonl"
    post_cutoff_acl_naacl_jsonl = args.post_cutoff_acl_naacl_jsonl or output_dir / "post-cutoff_acl_naacl.jsonl"
    post_cutoff_emnlp_jsonl = args.post_cutoff_emnlp_jsonl or output_dir / "post-cutoff_emnlp.jsonl"

    print("=" * 70)
    print("UNIFIED AWARD DATASET GENERATOR")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading HuggingFace papers...")
    hf_papers = load_huggingface_papers()

    print("\n[2/6] Loading 2025 papers...")
    papers_2025 = load_2025_papers(args.papers_2025_dir)

    print("\n[3/6] Loading award data...")
    awards_df = load_excel_awards(args.excel)

    # Build lookup with ALL papers (including 2025) for matching
    all_papers = pd.concat([hf_papers, papers_2025], ignore_index=True)
    papers_lookup = build_paper_lookup(all_papers)
    print(f"Built lookup with {len(papers_lookup)} title keys")

    # Get awarded titles for exclusion
    awarded_titles = {
        normalize_key(clean_title(title))
        for title in awards_df["title"].dropna()
    }
    awarded_titles = {t for t in awarded_titles if t}

    # Generate pre-cutoff MCQ
    print("\n[5/6] Generating pre-cutoff MCQ dataset...")

    # Award papers (pre-2025)
    award_records_pre, award_stats_pre = generate_award_mcq(
        awards_df,
        papers_lookup,
        include_pre_cutoff=True,
        include_post_cutoff=False,
        random_seed=args.seed,
    )
    print_stats("Award papers (pre-cutoff)", award_stats_pre)

    # Historical main/findings
    hist_records, hist_stats = generate_historical_mcq(
        hf_papers,
        awarded_titles,
        samples_per_track=args.historical_samples,
        random_seed=args.seed,
    )
    print_stats("Historical main/findings", hist_stats)

    # Combine and write
    pre_cutoff_records = award_records_pre + hist_records
    write_jsonl(pre_cutoff_records, pre_cutoff_jsonl)

    # Collect titles from pre-cutoff dataset for sandbox exclusion
    pre_cutoff_titles = set()
    for record in pre_cutoff_records:
        # Extract title from context (format: "Title: <title>\nAbstract: ...")
        context = record.get("context", "")
        if context.startswith("Title: "):
            title_line = context.split("\n")[0]
            title = title_line.replace("Title: ", "").strip()
            normalized_title = normalize_key(clean_title(title))
            if normalized_title:
                pre_cutoff_titles.add(normalized_title)

    print(f"\nCollected {len(pre_cutoff_titles)} titles from pre-cutoff dataset for sandbox exclusion")

    # Create sandbox CSV with only 2021-2024 papers, excluding pre-cutoff evaluation papers
    print("\n[4/6] Creating sandbox CSV (excluding pre-cutoff evaluation papers)...")
    # Filter out papers that appear in pre-cutoff evaluation set
    hf_papers["title_key"] = hf_papers["title"].apply(lambda t: normalize_key(clean_title(t)))
    sandbox_papers = hf_papers[~hf_papers["title_key"].isin(pre_cutoff_titles)].copy()
    sandbox_papers = sandbox_papers.drop(columns=["title_key"])

    original_count = len(hf_papers)
    filtered_count = len(sandbox_papers)
    excluded_count = original_count - filtered_count

    print(f"  Original papers: {original_count}")
    print(f"  Excluded (in evaluation): {excluded_count}")
    print(f"  Remaining for sandbox: {filtered_count}")

    write_accepted_papers_csv(sandbox_papers, sandbox_csv)

    # Generate post-cutoff MCQ
    print("\n[6/6] Generating post-cutoff MCQ dataset...")

    # Award papers (post-2025)
    award_records_post, award_stats_post = generate_award_mcq(
        awards_df,
        papers_lookup,
        include_pre_cutoff=False,
        include_post_cutoff=True,
        random_seed=args.seed,
    )
    print_stats("Award papers (post-cutoff)", award_stats_post)

    # Historical 2025 papers (ACL/NAACL)
    hist_2025_acl_naacl, hist_2025_acl_naacl_stats = generate_historical_mcq(
        all_papers,
        awarded_titles,
        samples_per_track=args.historical_samples,
        random_seed=args.seed,
        year_filter="post_cutoff",
        conference_filter="ACL|NAACL",
    )
    print_stats("Historical 2025 ACL/NAACL", hist_2025_acl_naacl_stats)

    # Historical 2025 papers (EMNLP)
    hist_2025_emnlp, hist_2025_emnlp_stats = generate_historical_mcq(
        all_papers,
        awarded_titles,
        samples_per_track=args.historical_samples,
        random_seed=args.seed,
        year_filter="post_cutoff",
        conference_filter="EMNLP",
    )
    print_stats("Historical 2025 EMNLP", hist_2025_emnlp_stats)

    # Split post-cutoff by conference and combine with historical
    acl_naacl_awards = [r for r in award_records_post if r["metadata"]["sheet"] in {"ACL", "NAACL"}]
    emnlp_awards = [r for r in award_records_post if r["metadata"]["sheet"] == "EMNLP"]

    acl_naacl_records = acl_naacl_awards + hist_2025_acl_naacl
    emnlp_records = emnlp_awards + hist_2025_emnlp

    write_jsonl(acl_naacl_records, post_cutoff_acl_naacl_jsonl)
    write_jsonl(emnlp_records, post_cutoff_emnlp_jsonl)

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Sandbox CSV: {sandbox_csv} ({filtered_count} papers, excluded {excluded_count} evaluation papers)")
    print(f"Pre-cutoff MCQ: {pre_cutoff_jsonl} ({len(pre_cutoff_records)} samples)")
    print(f"Post-cutoff ACL/NAACL MCQ: {post_cutoff_acl_naacl_jsonl} ({len(acl_naacl_records)} samples)")
    print(f"Post-cutoff EMNLP MCQ: {post_cutoff_emnlp_jsonl} ({len(emnlp_records)} samples)")
    print()


if __name__ == "__main__":
    main()
