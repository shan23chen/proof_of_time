#!/usr/bin/env python3
"""
Get citation counts for papers from HuggingFace dataset shanchen/Accepted-Papers-Aggregated

Uses Semantic Scholar API with title-only search.
Rate limit: 1 request per second (enforced by asyncio.sleep)
"""

import os
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm_asyncio
import aiohttp
from datasets import load_dataset
import pandas as pd
import logging

# -------------------- LOGGING SETUP --------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"citations_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.getenv("SCHOLAR_API_KEY", None)  # Optional but recommended

# Rate limiting: 1 request per second
RATE_LIMIT_DELAY = 1.0  # seconds between requests

OUTPUT_DIR = Path("citation_results")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "paper_citations.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "citation_summary.csv"

# -------------------- RATE LIMITING --------------------
last_request_time = 0
rate_limit_lock = asyncio.Lock()

async def rate_limit():
    """Enforce 1 request per second rate limit."""
    global last_request_time
    async with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time

        if time_since_last < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last
            await asyncio.sleep(sleep_time)

        last_request_time = time.time()

# -------------------- SEMANTIC SCHOLAR API --------------------
async def search_paper_by_title(
    session: aiohttp.ClientSession,
    title: str
) -> Optional[Dict[str, Any]]:
    """
    Search for a paper by title and get citation count.

    Returns dict with:
        - paper_id: Semantic Scholar ID
        - title: Paper title
        - citation_count: Number of citations
        - url: Semantic Scholar URL
        - year: Publication year
    """
    headers = {"x-api-key": API_KEY} if API_KEY else {}

    # Search endpoint
    search_url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": title,
        "fields": "paperId,title,citationCount,year,url",
        "limit": 1  # Get top match only
    }

    await rate_limit()  # Enforce rate limit

    try:
        async with session.get(
            search_url,
            headers=headers,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            # Handle rate limiting with exponential backoff
            if resp.status == 429:
                logger.warning(f"Rate limited (429) for '{title[:60]}...', retrying after 5 seconds")
                await asyncio.sleep(5)  # Wait longer before retry
                # Retry once
                async with session.get(search_url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp2:
                    if resp2.status == 429:
                        logger.error(f"Rate limited again (429) for '{title[:60]}...', giving up")
                        return None
                    elif resp2.status != 200:
                        logger.warning(f"HTTP {resp2.status} on retry for '{title[:60]}...'")
                        return None
                    data = await resp2.json()
                    results = data.get("data", [])
            elif resp.status != 200:
                logger.warning(f"HTTP {resp.status} for '{title[:60]}...'")
                return None
            else:
                data = await resp.json()
                results = data.get("data", [])

            if not results:
                logger.debug(f"No results for '{title[:60]}...'")
                return None

            # Get first (best) match
            paper = results[0]

            return {
                "paper_id": paper.get("paperId"),
                "title": paper.get("title"),
                "citation_count": paper.get("citationCount", 0),
                "year": paper.get("year"),
                "url": paper.get("url", ""),
            }

    except asyncio.TimeoutError:
        logger.error(f"Timeout for '{title[:60]}...'")
        return None
    except Exception as e:
        logger.error(f"Error for '{title[:60]}...': {e}")
        return None

# -------------------- DATASET LOADING --------------------
def load_papers_dataset() -> pd.DataFrame:
    """Load papers from HuggingFace dataset."""
    logger.info("Loading dataset from HuggingFace...")
    ds = load_dataset("shanchen/Accepted-Papers-Aggregated", split="train")

    # Convert to DataFrame
    df = pd.DataFrame(ds)
    logger.info(f"Loaded {len(df)} papers")

    # Show dataset info
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Source datasets: {df['source_dataset'].value_counts().to_dict()}")

    return df

# -------------------- PROCESSING --------------------
async def process_paper(
    session: aiohttp.ClientSession,
    paper: Dict[str, Any],
    index: int
) -> Dict[str, Any]:
    """Process a single paper: search and get citation count."""
    title = paper.get("title", "")

    if not title:
        return {
            "index": index,
            "original_title": title,
            "error": "empty title",
            "citation_count": None,
        }

    # Search Semantic Scholar
    result = await search_paper_by_title(session, title)

    if result:
        return {
            "index": index,
            "original_title": title,
            "matched_title": result.get("title"),
            "paper_id": result.get("paper_id"),
            "citation_count": result.get("citation_count", 0),
            "year": result.get("year"),
            "url": result.get("url"),
            "authors": paper.get("authors", ""),
            "source_dataset": paper.get("source_dataset", ""),
            "source_tags": paper.get("tags", ""),
            "error": None,
        }
    else:
        return {
            "index": index,
            "original_title": title,
            "error": "not found",
            "citation_count": None,
            "authors": paper.get("authors", ""),
            "source_dataset": paper.get("source_dataset", ""),
            "source_tags": paper.get("tags", ""),
        }

async def process_all_papers(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Process all papers in the dataset."""
    papers = df.to_dict("records")

    if limit:
        papers = papers[:limit]
        logger.info(f"Processing first {limit} papers (limited for testing)")

    results = []

    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Process papers sequentially (required for rate limiting)
        with tqdm_asyncio(total=len(papers), desc="Processing papers") as pbar:
            for i, paper in enumerate(papers):
                result = await process_paper(session, paper, i)
                results.append(result)

                # Save incrementally every 100 papers
                if (i + 1) % 100 == 0:
                    save_results(results, OUTPUT_FILE)

                pbar.update(1)

    return results

# -------------------- SAVING --------------------
def save_results(results: List[Dict[str, Any]], output_file: Path):
    """Save results to JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} results to {output_file}")

def save_summary(results: List[Dict[str, Any]], summary_file: Path):
    """Save summary statistics to CSV."""
    df = pd.DataFrame(results)

    # Summary statistics
    total = len(df)
    found = df[df["error"].isna()].shape[0]
    not_found = df[df["error"] == "not found"].shape[0]

    # Citation statistics (only for found papers)
    found_df = df[df["error"].isna()].copy()

    summary = {
        "total_papers": total,
        "found": found,
        "not_found": not_found,
        "success_rate": f"{100 * found / total:.1f}%",
    }

    if not found_df.empty:
        summary.update({
            "total_citations": int(found_df["citation_count"].sum()),
            "avg_citations": f"{found_df['citation_count'].mean():.1f}",
            "median_citations": int(found_df["citation_count"].median()),
            "max_citations": int(found_df["citation_count"].max()),
            "min_citations": int(found_df["citation_count"].min()),
        })

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to {summary_file}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS:")
    logger.info("="*60)
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60 + "\n")

    # Top cited papers
    if not found_df.empty:
        logger.info("Top 10 most cited papers:")
        top_10 = found_df.nlargest(10, "citation_count")[
            ["original_title", "citation_count", "year"]
        ]
        for idx, row in top_10.iterrows():
            citations = int(row['citation_count']) if pd.notna(row['citation_count']) else 0
            logger.info(f"  [{citations:5d}] {row['original_title'][:70]}...")

# -------------------- MAIN --------------------
async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Get citation counts for papers from HuggingFace dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file",
    )
    args = parser.parse_args()

    logger.info("Starting citation count extraction...")
    logger.info(f"API Key configured: {'Yes' if API_KEY else 'No (using free tier)'}")
    logger.info(f"Rate limit: {RATE_LIMIT_DELAY} seconds per request")

    # Load dataset
    df = load_papers_dataset()

    # Resume from existing results if requested
    start_index = 0
    existing_results = []

    if args.resume and OUTPUT_FILE.exists():
        logger.info(f"Resuming from {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_results = [json.loads(line) for line in f]
        start_index = len(existing_results)
        logger.info(f"Resuming from index {start_index}")
        df = df.iloc[start_index:].reset_index(drop=True)

    # Process papers
    results = await process_all_papers(df, limit=args.limit)

    # Combine with existing results if resuming
    if existing_results:
        results = existing_results + results

    # Save final results
    save_results(results, OUTPUT_FILE)
    save_summary(results, SUMMARY_FILE)

    logger.info("Done!")
    logger.info(f"Results saved to: {OUTPUT_FILE}")
    logger.info(f"Summary saved to: {SUMMARY_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
