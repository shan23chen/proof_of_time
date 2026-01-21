#!/usr/bin/env python3
"""
Generate citation prediction benchmark datasets.

This script consolidates citation dataset generation by:
1. Loading historical papers from HuggingFace datasets (2021-2024)
2. Fetching citation counts from Semantic Scholar API for ALL papers
3. Generating historical corpus with citation counts for sandbox
4. Sampling 2025 papers and fetching their citation counts
5. Creating three task types: multiple choice, ranking, and bucket prediction

Outputs:
- benchmarks/citation_react/sandbox/data/historical_papers_2021_2024.jsonl (WITH citation counts)
- benchmarks/citation_react/multiple_choice.jsonl
- benchmarks/citation_react/bucket_prediction.jsonl
- benchmarks/citation_react/ranking.jsonl

Usage:
    # Generate all datasets (WARNING: will overwrite existing files)
    python dataset_building/generate_citation_datasets.py

    # Dry run to preview what would be generated
    python dataset_building/generate_citation_datasets.py --dry-run

    # Resume from checkpoint (if interrupted during citation fetching)
    python dataset_building/generate_citation_datasets.py --resume

Note: Fetching citations for ~38K historical papers takes several hours.
Use --limit to process only first N papers for testing.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from datasets import load_dataset

try:
    from tqdm.asyncio import tqdm_asyncio
except Exception:
    tqdm_asyncio = None

# ============================== CONFIGURATION ==============================

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "citation_react"
SANDBOX_DIR = OUTPUT_DIR / "sandbox" / "data"

# Input data files
AWARD_PAPERS_CSV = DATA_DIR / "best_paper_citations_detailed.csv"
PAPERS_2025_DIR = DATA_DIR / "2025papers"

# Output files (dry-run mode will not create these)
HISTORICAL_PAPERS_OUTPUT = SANDBOX_DIR / "historical_papers_2021_2024.jsonl"
MULTIPLE_CHOICE_OUTPUT = OUTPUT_DIR / "multiple_choice.jsonl"
BUCKET_PREDICTION_OUTPUT = OUTPUT_DIR / "bucket_prediction.jsonl"
RANKING_OUTPUT = OUTPUT_DIR / "ranking.jsonl"

# HuggingFace datasets for historical papers (2021-2024)
HF_DATASETS = [
    "AIM-Harvard/EMNLP-Accepted-Papers",
    "AIM-Harvard/ACL-Accepted-Papers",
    "AIM-Harvard/NAACL-Accepted-Papers",
]

# Citation buckets
CITATION_BUCKETS = [
    ("A", "0-10 citations (very low impact)", 0, 9),
    ("B", "10-25 citations (low impact)", 10, 25),
    ("C", "25-50 citations (medium impact)", 26, 50),
    ("D", "50-150 citations (high impact)", 51, 150),
    ("E", "150+ citations (very high impact)", 151, 10**9),
]

# Semantic Scholar API configuration
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.getenv("SCHOLAR_API_KEY", None)
RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Task generation parameters
SAMPLE_N_PER_FILE = 100  # Papers to sample per input file
N_TASKS_PER_TYPE = 50    # Tasks to generate per task type per input file
MC_K = 4                 # Number of papers in multiple choice
RANK_K = 4               # Number of papers in ranking (fixed at 4)

# Checkpoint file for resume support
CHECKPOINT_FILE = LOG_DIR / "citation_generation_checkpoint.jsonl"

# ============================== LOGGING SETUP ==============================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"citation_dataset_generation_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================== SEMANTIC SCHOLAR ==============================

_last_request_time = 0.0
_rate_limit_lock = asyncio.Lock()


async def rate_limit(delay: float) -> None:
    """Enforce rate limiting between requests."""
    global _last_request_time
    async with _rate_limit_lock:
        now = time.time()
        since_last = now - _last_request_time
        if since_last < delay:
            await asyncio.sleep(delay - since_last)
        _last_request_time = time.time()


async def search_paper_by_title(
    session: aiohttp.ClientSession, title: str, delay: float
) -> Optional[Dict[str, Any]]:
    """Search a paper by title and return citation count + metadata."""
    if not title or pd.isna(title):
        return None

    headers = {"x-api-key": API_KEY} if API_KEY else {}
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": str(title),
        "fields": "paperId,title,citationCount,year,url",
        "limit": 1,
    }

    await rate_limit(delay)

    try:
        async with session.get(
            url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            if resp.status == 429:
                logger.warning(
                    f"429 rate limit for '{str(title)[:60]}...', backing off 5s"
                )
                await asyncio.sleep(5)
                await rate_limit(delay)
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp2:
                    if resp2.status != 200:
                        logger.warning(
                            f"HTTP {resp2.status} on retry for '{str(title)[:60]}...'"
                        )
                        return None
                    data = await resp2.json()
            elif resp.status != 200:
                logger.warning(f"HTTP {resp.status} for '{str(title)[:60]}...'")
                return None
            else:
                data = await resp.json()

        results = (data or {}).get("data", []) or []
        if not results:
            return None

        paper = results[0] or {}
        return {
            "paper_id": paper.get("paperId"),
            "matched_title": paper.get("title"),
            "citation_count": paper.get("citationCount", 0),
            "year": paper.get("year"),
            "s2_url": paper.get("url", ""),
        }
    except asyncio.TimeoutError:
        logger.error(f"Timeout for '{str(title)[:60]}...'")
        return None
    except Exception as e:
        logger.error(f"Error for '{str(title)[:60]}...': {e}")
        return None


# ============================== UTILITY FUNCTIONS ==============================


def _safe_str(x) -> str:
    """Convert value to string safely."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _get_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column from candidates (case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV with encoding fallback."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def normalize_title(title: str) -> str:
    """Normalize title for matching (remove special chars, lowercase)."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


def infer_year(row: pd.Series, filename: str) -> Optional[int]:
    """Infer year from row data or filename."""
    year_re = re.compile(r"(19|20)\d{2}")

    # Try explicit year columns
    for col in ["year", "Year", "publication_year", "pub_year"]:
        if col in row.index:
            v = row[col]
            try:
                if pd.isna(v):
                    continue
                y = int(v)
                if 1900 <= y <= 2100:
                    return y
            except Exception:
                pass

    # Try conference field
    conf = _safe_str(row.get("conference", ""))
    m = year_re.search(conf)
    if m:
        return int(m.group(0))

    # Try filename
    m = year_re.search(filename)
    if m:
        return int(m.group(0))

    return None


def infer_conference(df: pd.DataFrame, input_path: Path) -> str:
    """Infer conference name from data or filename."""
    # Prefer explicit columns
    if "conference" in df.columns:
        val = df["conference"].dropna()
        if len(val) > 0:
            return str(val.iloc[0])

    if "tags" in df.columns:
        val = df["tags"].dropna()
        if len(val) > 0:
            return str(val.iloc[0])

    # Fallback to filename heuristics
    stem = input_path.stem.lower()
    conf = input_path.stem

    if "acl" in stem and "2025" in stem:
        conf = "ACL 2025"
    elif "naacl" in stem and "2025" in stem:
        conf = "NAACL 2025"
    elif "emnlp" in stem and "2025" in stem:
        conf = "EMNLP 2025"

    if "finding" in stem:
        conf += " Findings"
    elif "main" in stem:
        conf += " Main"

    return conf


def bucket_for_citations(cc: int) -> Tuple[str, str]:
    """Map citation count to bucket letter and label."""
    for letter, label, lo, hi in CITATION_BUCKETS:
        if lo <= cc <= hi:
            return letter, label
    return "E", CITATION_BUCKETS[-1][1]


# ============================== PAPER DATA STRUCTURES ==============================


class Paper:
    """Represents a paper with metadata."""

    __slots__ = (
        "title",
        "abstract",
        "authors",
        "conference",
        "year",
        "citation_count",
        "source_file",
    )

    def __init__(
        self,
        title: str,
        abstract: str,
        authors: str,
        conference: str,
        year: Optional[int],
        citation_count: int,
        source_file: str,
    ):
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.conference = conference
        self.year = year
        self.citation_count = citation_count
        self.source_file = source_file


# ============================== DATA LOADING ==============================


def load_huggingface_papers() -> pd.DataFrame:
    """Load historical papers (2021-2024) from HuggingFace datasets."""
    logger.info("Loading historical papers from HuggingFace datasets...")

    all_dfs = []
    for hf_dataset in HF_DATASETS:
        logger.info(f"  Loading {hf_dataset}...")
        dataset_dict = load_dataset(hf_dataset)

        for split_name, split_data in dataset_dict.items():
            split_df = split_data.to_pandas()
            split_df["source_split"] = split_name
            split_df["hf_dataset"] = hf_dataset
            all_dfs.append(split_df)

    papers_df = pd.concat(all_dfs, ignore_index=True)

    # Filter to 2021-2024
    papers_df = papers_df[
        (papers_df["year"] >= 2021) & (papers_df["year"] <= 2024)
    ].copy()

    logger.info(
        f"  Loaded {len(papers_df)} papers from {len(HF_DATASETS)} datasets (2021-2024)"
    )
    return papers_df


async def annotate_with_citations(df: pd.DataFrame, delay: float) -> pd.DataFrame:
    """Add citation_count column by querying Semantic Scholar."""
    logger.info(f"Fetching citations for {len(df)} papers...")
    citation_counts: List[Optional[int]] = []

    async with aiohttp.ClientSession() as session:
        iterator = range(len(df))
        if tqdm_asyncio is not None:
            pbar = tqdm_asyncio(total=len(df), desc="Fetching citations")
        else:
            pbar = None

        for i in iterator:
            title = df.loc[i, "title"]
            result = await search_paper_by_title(session, title, delay)
            citation_counts.append(result.get("citation_count") if result else None)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    df_out = df.copy()
    df_out["citation_count"] = citation_counts
    logger.info(
        f"  Successfully fetched {sum(1 for c in citation_counts if c is not None)}/{len(df)} citations"
    )
    return df_out


def sample_and_prepare_papers(
    input_csv: Path, sample_n: int, seed: int
) -> Tuple[pd.DataFrame, str]:
    """Sample papers from CSV and prepare with standard schema."""
    logger.info(f"Sampling {sample_n} papers from {input_csv.name}...")

    df = read_csv_safe(input_csv)

    title_col = _get_column(df, ["title", "Title"])
    if not title_col:
        raise ValueError(
            f"No title column in {input_csv}. Expected 'title'. Columns={list(df.columns)}"
        )

    authors_col = _get_column(df, ["authors", "author", "Authors", "Author"])
    abstract_col = _get_column(df, ["abstract", "abs", "Abstract", "Abs"])

    conference = infer_conference(df, input_csv)

    df2 = df.copy()
    df2["_title"] = df2[title_col].astype(str)
    df2["_authors"] = df2[authors_col].astype(str) if authors_col else ""
    df2["_abstract"] = df2[abstract_col].astype(str) if abstract_col else ""
    df2["conference"] = conference

    # Drop empty titles
    df2 = df2[df2["_title"].notna() & (df2["_title"].str.strip() != "")]

    n = min(sample_n, len(df2))
    sampled = df2.sample(n=n, random_state=seed).reset_index(drop=True)

    out = sampled[["_title", "_authors", "_abstract", "conference"]].rename(
        columns={"_title": "title", "_authors": "authors", "_abstract": "abstract"}
    )

    logger.info(f"  Sampled {len(out)} papers | conference='{conference}'")
    return out, conference


def load_papers_from_csv(csv_path: Path) -> List[Paper]:
    """Load papers from CSV into Paper objects."""
    df = read_csv_safe(csv_path)

    title_col = _get_column(df, ["title", "paper_title"])
    abs_col = _get_column(df, ["abstract", "paper_abstract"])
    auth_col = _get_column(df, ["authors", "author", "paper_authors"])
    conf_col = _get_column(df, ["conference", "venue", "conf"])
    cite_col = _get_column(df, ["citation_count", "citations", "cited_by_count"])

    missing = [
        ("title", title_col),
        ("abstract", abs_col),
        ("authors", auth_col),
        ("conference", conf_col),
        ("citation_count", cite_col),
    ]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise ValueError(
            f"{csv_path}: missing required columns: {missing}. Have: {list(df.columns)}"
        )

    papers: List[Paper] = []
    for _, row in df.iterrows():
        title = _safe_str(row[title_col]).strip()
        if not title:
            continue

        abstract = _safe_str(row[abs_col]).strip()
        authors = _safe_str(row[auth_col]).strip()
        conf_raw = _safe_str(row[conf_col]).strip()
        year = infer_year(row, csv_path.name)

        # Extract conference name from raw text
        token = re.findall(r"[A-Za-z]+", conf_raw)
        conf = token[0].upper() if token else conf_raw

        cc_raw = row[cite_col]
        try:
            cc = int(0 if pd.isna(cc_raw) else cc_raw)
        except Exception:
            m = re.search(r"\d+", _safe_str(cc_raw))
            cc = int(m.group(0)) if m else 0

        papers.append(Paper(title, abstract, authors, conf, year, cc, csv_path.name))

    return papers


# ============================== TASK GENERATION ==============================


def group_by_year_conf(papers: List[Paper]) -> Dict[Tuple[Optional[int], str], List[Paper]]:
    """Group papers by (year, conference)."""
    groups: Dict[Tuple[Optional[int], str], List[Paper]] = {}
    for p in papers:
        groups.setdefault((p.year, p.conference), []).append(p)
    return groups


def pick_group(
    groups: Dict[Tuple[Optional[int], str], List[Paper]],
    k_needed: int,
    rng: np.random.Generator,
) -> Optional[Tuple[Optional[int], str]]:
    """Pick a random group that has at least k_needed papers."""
    eligible = [k for k, v in groups.items() if len(v) >= k_needed]
    if not eligible:
        return None
    return eligible[int(rng.integers(0, len(eligible)))]


def format_context_single(p: Paper, authors_included: bool) -> str:
    """Format context for single paper (bucket prediction)."""
    lines = [f"Title: {p.title}"]
    if authors_included and p.authors:
        lines.append(f"Authors: {p.authors}")
    if p.abstract:
        lines.append(f"Abstract: {p.abstract}")
    lines.append("")

    for letter, label, _, _ in CITATION_BUCKETS:
        lines.append(f"{letter}. {label}")

    return "\n".join(lines)


def format_context_options(papers: List[Paper], letters: List[str], authors_included: bool) -> str:
    """Format context for multiple papers (MCQ/ranking)."""
    blocks = []
    for L, p in zip(letters, papers):
        lines = [f"{L}. Title: {p.title}"]
        if authors_included and p.authors:
            lines.append(f"Authors: {p.authors}")
        if p.abstract:
            lines.append(f"Abstract: {p.abstract}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def make_bucket_task(p: Paper, authors_included: bool) -> dict:
    """Generate bucket prediction task."""
    ans_letter, bucket_label = bucket_for_citations(p.citation_count)
    return {
        "question": "Which citation range does this paper fall into?",
        "choices": [b[0] for b in CITATION_BUCKETS],
        "answer": ans_letter,
        "context": format_context_single(p, authors_included),
        "metadata": {
            "conference": p.conference,
            "year": p.year,
            "sampling_strategy": "same_conference_same_year",
            "authors_included": bool(authors_included),
            "paper_title": p.title,
            "actual_citation_count": int(p.citation_count),
            "bucket_label": bucket_label,
        },
    }


def make_mc_task(papers: List[Paper], authors_included: bool) -> dict:
    """Generate multiple choice task."""
    letters = [chr(65 + i) for i in range(len(papers))]
    ccs = [p.citation_count for p in papers]
    best_idx = int(np.argmax(ccs))

    return {
        "question": "Which of these papers has the highest citation count?",
        "choices": letters,
        "answer": letters[best_idx],
        "context": format_context_options(papers, letters, authors_included),
        "metadata": {
            "conference": papers[0].conference,
            "year": papers[0].year,
            "sampling_strategy": "same_conference_same_year",
            "authors_included": bool(authors_included),
            "all_papers_citations": [
                {
                    "position": letters[i],
                    "title": papers[i].title,
                    "citation_count": int(papers[i].citation_count),
                }
                for i in range(len(papers))
            ],
        },
    }


def make_ranking_task(papers: List[Paper], authors_included: bool) -> dict:
    """Generate ranking task (always 4 papers with A, B, C, D)."""
    letters = ["A", "B", "C", "D"]
    ccs = np.array([p.citation_count for p in papers], dtype=int)
    order = list(np.argsort(-ccs))  # descending
    answer = ", ".join([letters[i] for i in order])
    correct_titles = [papers[i].title for i in order]

    return {
        "question": "Rank these papers from most cited to least cited.",
        "choices": letters,
        "answer": answer,
        "context": format_context_options(papers, letters, authors_included),
        "metadata": {
            "conference": papers[0].conference,
            "year": papers[0].year,
            "sampling_strategy": "same_conference_same_year",
            "authors_included": bool(authors_included),
            "correct_order_titles": correct_titles,
            "all_papers_citations": [
                {
                    "position": letters[i],
                    "title": papers[i].title,
                    "citation_count": int(papers[i].citation_count),
                }
                for i in range(len(papers))
            ],
        },
    }


def generate_tasks_from_papers(
    papers: List[Paper], n_tasks: int, mc_k: int, rank_k: int, authors_included: bool
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Generate all three task types from papers."""
    rng = np.random.default_rng(SEED)
    groups = group_by_year_conf(papers)

    bucket_tasks: List[dict] = []
    mc_tasks: List[dict] = []
    ranking_tasks: List[dict] = []

    # Bucket prediction: sample individual papers
    n = min(n_tasks, len(papers))
    idxs = rng.choice(len(papers), size=n, replace=False)
    for i in idxs:
        p = papers[int(i)]
        bucket_tasks.append(make_bucket_task(p, authors_included))

    # Multiple choice: sample within same (year, conf)
    made = 0
    tries = 0
    while made < n_tasks and tries < n_tasks * 50:
        tries += 1
        key = pick_group(groups, mc_k, rng)
        if key is None:
            break
        pool = groups[key]
        sel = rng.choice(len(pool), size=mc_k, replace=False)
        ps = [pool[int(j)] for j in sel]
        mc_tasks.append(make_mc_task(ps, authors_included))
        made += 1

    # Ranking: always 4 papers
    made = 0
    tries = 0
    while made < n_tasks and tries < n_tasks * 50:
        tries += 1
        key = pick_group(groups, rank_k, rng)
        if key is None:
            break
        pool = groups[key]
        sel = rng.choice(len(pool), size=rank_k, replace=False)
        ps = [pool[int(j)] for j in sel]
        ranking_tasks.append(make_ranking_task(ps, authors_included))
        made += 1

    logger.info(
        f"  Generated {len(bucket_tasks)} bucket, {len(mc_tasks)} MCQ, {len(ranking_tasks)} ranking tasks"
    )

    return bucket_tasks, mc_tasks, ranking_tasks


# ============================== OUTPUT ==============================


def write_jsonl(path: Path, records: List[dict], dry_run: bool = False) -> int:
    """Write records to JSONL file."""
    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(records)} records to {path}")
        return len(records)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(records)} records to {path}")
    return len(records)


def write_historical_papers(papers_df: pd.DataFrame, output_path: Path, dry_run: bool = False):
    """Write historical papers to JSONL for sandbox (now includes citation_count)."""
    if dry_run:
        logger.info(
            f"[DRY RUN] Would write {len(papers_df)} historical papers to {output_path}"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in papers_df.iterrows():
            record = {
                "title": str(row.get("title", "")),
                "abstract": str(row.get("abstract", "")),
                "authors": str(row.get("authors", "")),
                "year": int(row.get("year", 0)),
                "conference": str(row.get("source_split", "")),
                "citation_count": int(row.get("citation_count", 0)) if pd.notna(row.get("citation_count")) else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(papers_df)} historical papers to {output_path}")


# ============================== CHECKPOINT MANAGEMENT ==============================


def save_checkpoint(papers_with_citations: List[dict]):
    """Save progress to checkpoint file for resume support."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        for paper in papers_with_citations:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    logger.info(f"Checkpoint saved: {len(papers_with_citations)} papers")


def load_checkpoint() -> Optional[List[dict]]:
    """Load progress from checkpoint file if it exists."""
    if not CHECKPOINT_FILE.exists():
        return None

    papers = []
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    logger.info(f"Loaded checkpoint: {len(papers)} papers with citations")
    return papers


# ============================== MAIN PIPELINE ==============================


async def main(dry_run: bool = False, resume: bool = False, limit: Optional[int] = None):
    """Main execution pipeline."""
    logger.info("=" * 80)
    logger.info("Citation Dataset Generation Pipeline")
    logger.info("=" * 80)

    if dry_run:
        logger.info("*** DRY RUN MODE - No files will be written ***")
        logger.info("")

    # Step 1: Load historical papers (2021-2024) and fetch their citation counts
    logger.info("\n[Step 1] Loading historical papers from HuggingFace...")
    historical_papers_df = load_huggingface_papers()

    if limit:
        logger.info(f"  Limiting to first {limit} papers for testing")
        historical_papers_df = historical_papers_df.head(limit).reset_index(drop=True)

    # Check for checkpoint
    checkpoint_data = None
    if resume:
        checkpoint_data = load_checkpoint()

    if checkpoint_data:
        logger.info(f"  Resuming from checkpoint with {len(checkpoint_data)} papers")
        # Convert checkpoint to DataFrame
        historical_papers_df = pd.DataFrame(checkpoint_data)
    else:
        # Fetch citations for historical papers
        logger.info(f"\n[Step 1b] Fetching citations for {len(historical_papers_df)} historical papers...")
        logger.info("  This may take several hours for ~38K papers")
        logger.info("  Use Ctrl+C to interrupt - you can resume with --resume flag")

        if not dry_run:
            historical_papers_df = await annotate_with_citations(
                historical_papers_df, delay=RATE_LIMIT_DELAY
            )

            # Save checkpoint every 100 papers (done in annotate_with_citations via callback)
            # Convert to list of dicts for checkpoint
            checkpoint_records = historical_papers_df.to_dict("records")
            save_checkpoint(checkpoint_records)

    # Write historical papers with citation counts
    write_historical_papers(historical_papers_df, HISTORICAL_PAPERS_OUTPUT, dry_run)

    # Step 2: Sample and fetch citations for 2025 papers
    logger.info("\n[Step 2] Sampling and fetching citations for 2025 papers...")

    csv_files = list(PAPERS_2025_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {PAPERS_2025_DIR}")
        return

    combined_rows: List[pd.DataFrame] = []

    for input_csv in csv_files:
        sampled, conference = sample_and_prepare_papers(
            input_csv, SAMPLE_N_PER_FILE, SEED
        )
        annotated = await annotate_with_citations(sampled, delay=RATE_LIMIT_DELAY)

        annotated2 = annotated.copy()
        annotated2["source_file"] = input_csv.name
        combined_rows.append(annotated2)

    if not combined_rows:
        logger.error("No papers were processed. Exiting.")
        return

    combined_df = pd.concat(combined_rows, ignore_index=True)

    # Filter out papers without citation counts
    valid_df = combined_df[combined_df["citation_count"].notna()].copy()
    logger.info(
        f"\nCombined dataset: {len(valid_df)} papers with valid citation counts"
    )

    # Step 3: Generate tasks
    logger.info("\n[Step 3] Generating citation prediction tasks...")

    # Convert DataFrame to Paper objects
    all_papers = []
    for _, row in valid_df.iterrows():
        paper = Paper(
            title=str(row["title"]),
            abstract=str(row["abstract"]),
            authors=str(row["authors"]),
            conference=str(row["conference"]),
            year=infer_year(row, str(row.get("source_file", ""))),
            citation_count=int(row["citation_count"]),
            source_file=str(row.get("source_file", "")),
        )
        all_papers.append(paper)

    bucket_tasks, mc_tasks, ranking_tasks = generate_tasks_from_papers(
        all_papers,
        n_tasks=N_TASKS_PER_TYPE,
        mc_k=MC_K,
        rank_k=RANK_K,
        authors_included=False,
    )

    # Step 4: Write output files
    logger.info("\n[Step 4] Writing output files...")

    write_jsonl(BUCKET_PREDICTION_OUTPUT, bucket_tasks, dry_run)
    write_jsonl(MULTIPLE_CHOICE_OUTPUT, mc_tasks, dry_run)
    write_jsonl(RANKING_OUTPUT, ranking_tasks, dry_run)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  Historical papers: {HISTORICAL_PAPERS_OUTPUT} ({len(historical_papers_df)} papers)")
    logger.info(f"  Bucket prediction: {BUCKET_PREDICTION_OUTPUT} ({len(bucket_tasks)} tasks)")
    logger.info(f"  Multiple choice: {MULTIPLE_CHOICE_OUTPUT} ({len(mc_tasks)} tasks)")
    logger.info(f"  Ranking: {RANKING_OUTPUT} ({len(ranking_tasks)} tasks)")
    logger.info(f"\nLog file: {LOG_FILE}")

    if dry_run:
        logger.info("\n*** DRY RUN - No files were actually written ***")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate citation prediction benchmark datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be generated without writing files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (if interrupted during citation fetching)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of historical papers to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, resume=args.resume, limit=args.limit))
