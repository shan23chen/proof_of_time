# Dataset Building Scripts

This directory contains scripts for generating Inspect AI benchmark datasets from raw data sources.

## Active Scripts

### Award Prediction

**`generate_award_datasets.py`** - Generate award prediction benchmark datasets

Consolidates award paper data from multiple sources into evaluation datasets.

```bash
# Preview what would be generated
python dataset_building/generate_award_datasets.py --dry-run

# Generate datasets
python dataset_building/generate_award_datasets.py
```

**Outputs:**
- `benchmarks/award_react/pre-cutoff_mcq.jsonl` (258 samples: pre-2025 awards + historical)
- `benchmarks/award_react/post-cutoff_acl_naacl.jsonl` (110 samples: ACL/NAACL 2025)
- `benchmarks/award_react/post-cutoff_emnlp.jsonl` (68 samples: EMNLP 2025)
- `benchmarks/award_react/sandbox/data/accepted_papers.csv` (sandbox corpus)

**Sources:**
- `data/pot-best-papers-updated.xlsx` (award paper list)
- HuggingFace datasets: `AIM-Harvard/{ACL,EMNLP,NAACL}-Accepted-Papers`
- `data/2025papers/*.csv` (2025 conference papers)

---

### Citation Prediction

**`generate_citation_datasets.py`** - Generate citation prediction benchmark datasets

Fetches citation counts for ~38K historical papers (2021-2024) and 2025 papers, then generates prediction tasks.

```bash
# Preview what would be generated
python dataset_building/generate_citation_datasets.py --dry-run

# Generate datasets (requires SCHOLAR_API_KEY for higher rate limits)
# WARNING: Takes several hours to fetch citations for 38K historical papers
export SCHOLAR_API_KEY="your-key"
python dataset_building/generate_citation_datasets.py

# Test with first 100 papers only
python dataset_building/generate_citation_datasets.py --limit 100

# Resume if interrupted during citation fetching
python dataset_building/generate_citation_datasets.py --resume
```

**Outputs:**
- `benchmarks/citation_react/sandbox/data/historical_papers_2021_2024.jsonl` (38,306 papers **with citation counts**)
- `benchmarks/citation_react/multiple_choice.jsonl` (200 tasks)
- `benchmarks/citation_react/bucket_prediction.jsonl` (200 tasks)
- `benchmarks/citation_react/ranking.jsonl` (200 tasks)

**Sources:**
- HuggingFace datasets: `AIM-Harvard/{ACL,EMNLP,NAACL}-Accepted-Papers` (2021-2024)
- `data/2025papers/*.csv` (2025 papers for evaluation)
- Semantic Scholar API (citation counts for ALL papers)

**Features:**
- Checkpoint/resume support for long-running API calls
- Incremental progress saving every 100 papers
- Rate limiting (1 req/sec) with backoff for 429 errors

---

### Faculty Future Work Prediction

**`generate_faculty_futurework.py`** - Generate faculty research direction prediction datasets

Creates MCQ tasks predicting faculty research focus areas and paper attributions.

```bash
uv run python dataset_building/generate_faculty_futurework.py \
  --source-dir faculty_publications \
  --sandbox-dir benchmarks/future_work_react/sandbox/data \
  --professor-field-output benchmarks/future_work_react/professor_field_mcq.jsonl \
  --professor-article-output benchmarks/future_work_react/professor_article_mcq.jsonl \
  --field-focus-output benchmarks/future_work_react/field_focus_mcq.jsonl
```

**Outputs:**
- `professor_field_mcq.jsonl` - Predict professor's research field
- `professor_article_mcq.jsonl` - Attribute paper to correct professor
- `field_focus_mcq.jsonl` - Classify papers by field
- `sandbox/data/faculty_publications.jsonl` - Faculty publication corpus

**Sources:**
- `faculty_publications/*.csv` (76 faculty members, ~16k publications)

---

### SOTA Forecast

**`generate_sota_forecast.py`** - Generate SOTA metric prediction datasets

Creates bucket prediction tasks for frontier model benchmark scores.

```bash
uv run python dataset_building/generate_sota_forecast.py \
  --sandbox-dir benchmarks/sota_forecast/sandbox/data \
  --dataset-path benchmarks/sota_forecast/mcq_dataset.jsonl
```

**Outputs:**
- `mcq_dataset.jsonl` - SOTA score bucket prediction tasks
- `sandbox/data/sota_metrics.json` - Historical SOTA metrics

---

## Utility Scripts

**`aggregate_hf_accepted_papers.py`** - Aggregate papers from HuggingFace datasets

Consolidates multiple conference datasets into a single corpus.

**`extract_topics.py`** - Extract topics and themes from papers

Analyzes paper titles/abstracts to identify research topics and domains.

**`future_work.py`** - Extract future work sections from papers

Parses papers to identify stated future research directions.

---

### Output Format

- **JSONL** for evaluation datasets (one JSON object per line)
- **CSV** for sandbox corpora (structured data for agent analysis)
- **JSON** for configuration/metadata files

### Encoding Handling

Scripts include UTF-8/latin-1 fallback for robust CSV reading:
```python
def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
```

### Title Normalization

Unicode NFKC normalization for fuzzy title matching across sources:
```python
def normalize_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.lower().split())
```

---

## Requirements

```bash
# Core dependencies
pip install datasets pandas numpy

# For citation datasets (Semantic Scholar API)
pip install aiohttp tqdm

# For running with uv
uv sync
```

---

## Notes

- Scripts include `--dry-run` modes to preview outputs without overwriting files
- Large datasets may take 10-30 minutes to generate (especially citation tasks with API calls)
- Semantic Scholar API key recommended for citation generation: https://www.semanticscholar.org/product/api
