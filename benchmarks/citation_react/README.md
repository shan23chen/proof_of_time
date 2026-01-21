# Citation Prediction Benchmark - Inspect AI

This benchmark evaluates LLM agents on citation prediction tasks using a sandboxed environment with historical paper data.

## Overview

The agent must predict citation counts for **2025 papers** using only historical data (2021-2024) available in a Docker sandbox:

- **Historical papers database** (2021-2024): 38,330 papers with citation counts, authors, titles, venues
- **EMNLP papers** (2021-2024): 7,024 papers with topics, authors, metadata
- **Evaluation set** (2025): 147 papers - the agent predicts these based on historical patterns

This setup mimics real-world citation prediction: using past data to predict future impact.

You can download our generated questions from: https://huggingface.co/datasets/AIM-Harvard/proof-of-time/tree/main/benchmarks

## Tasks

### 1. Multiple Choice (`citation_multiple_choice`)

**Question**: Which of these 4 papers has the highest citation count?

**Example**:

```
Papers to evaluate:
A. Paper Title 1
B. Paper Title 2
C. Paper Title 3
D. Paper Title 4

Answer with only the letter (A, B, C, or D).
```

**Dataset**: 10 tasks (all from 2025 papers)

### 2. Ranking (`citation_ranking`)

**Question**: Rank these 4 papers from most to least cited.

**Example**:

```
Papers to rank:
A. Paper Title 1
B. Paper Title 2
C. Paper Title 3
D. Paper Title 4

Answer as a sequence (e.g., "B, D, A, C").
```

**Dataset**: 10 tasks (all from 2025 papers)

### 3. Bucket Prediction (`citation_bucket_prediction`)

**Question**: Predict the citation range for this paper.

**Buckets**:

- A. 0-10 citations (very low impact)
- B. 10-25 citations (low impact)
- C. 25-60 citations (moderate impact)
- D. 60-150 citations (high impact)
- E. 150+ citations (very high impact)

**Dataset**: 10 tasks (all from 2025 papers)

### 4. All Tasks Combined (`citation_all_tasks`)

Run all 30 tasks together.

## Running the Benchmark

### Prerequisites

```bash
# Install Inspect AI
pip install inspect-ai

# Ensure Docker is running
docker ps
```

### Run Individual Tasks

```bash
# Multiple choice only
inspect eval benchmarks/citation_react/benchmark.py@citation_multiple_choice \
  --model openai/gpt-4o-mini

# Ranking only
inspect eval benchmarks/citation_react/benchmark.py@citation_ranking \
  --model openai/gpt-4o-mini

# Bucket prediction only
inspect eval benchmarks/citation_react/benchmark.py@citation_bucket_prediction \
  --model openai/gpt-4o-mini
```

### Run All Tasks

```bash
inspect eval benchmarks/citation_react/benchmark.py@citation_all_tasks \
  --model openai/gpt-4o-mini
```

### With Different Models

```bash
# Claude
inspect eval benchmarks/citation_react/benchmark.py@citation_all_tasks \
  --model anthropic/claude-3-5-sonnet-20241022

# GPT-4
inspect eval benchmarks/citation_react/benchmark.py@citation_all_tasks \
  --model openai/gpt-4o
```

## Sandbox Environment

### Data Files

Located in `sandbox/data/` (mounted as read-only at `/dataset/`):

- `historical_papers_2021_2024.jsonl`: 38,330 papers from 2021-2024 with citations

### Available Tools

- `bash()`: Execute shell commands
- `python()`: Run Python code for analysis
- `bash_session()`: Persistent shell session
- `text_editor()`: Read/write files
- `think()`: Chain-of-thought reasoning

### Network Access

**Disabled** (`network_mode: none`) - agent must use only sandbox data

## Agent Strategy

The React agent is prompted to:

1. **Load historical database** - Explore JSONL files
2. **Analyze patterns** - Find correlations (topics, authors, venues → citations)
3. **Search for similar papers** - Match by keywords, authors, year
4. **Consider temporal trends** - Recent papers have fewer citations
5. **Make evidence-based predictions** - Base on sandbox data only

## Expected Performance

| Task              | Random Baseline | Expected Agent |
| ----------------- | --------------- | -------------- |
| Multiple Choice   | 25%             | 40-50%         |
| Ranking           | ~4%             | 20-30%         |
| Bucket Prediction | 20%             | 35-45%         |

## Evaluation Metrics

### Accuracy

- **Multiple Choice**: Exact match (A, B, C, or D)
- **Ranking**: Exact sequence match
- **Bucket Prediction**: Exact match (A-E)

### Additional Analysis

Results include:

- Per-task-type accuracy
- Per-sampling-strategy breakdown
- Ground truth citation counts for error analysis

## Sampling Strategies

Tasks use two sampling strategies to control confounds:

1. **Same conference, same year, same award bucket**

   - Tests pure citation prediction ability
   - Papers from same context (venue + year + prestige)
2. **Same year, different conferences, same award bucket**

   - Tests cross-venue comparison
   - Controls for temporal and prestige effects

## Data Sources

Generated from:

- `citation_eval/tasks/sandbox_agent_tasks.json` (60 tasks)
- `citation_eval/data/emnlp_papers.jsonl` (historical database)
- `citation_eval/data/best_papers_historical.jsonl` (ground truth)

## Troubleshooting

### Task file not found

```bash
# Regenerate sandbox tasks
uv run python citation_eval/inspect_citation_eval.py
```

### Docker issues

```bash
# Check Docker is running
docker ps

# Pull Python image
docker pull python:3.11-slim
```

### Data not accessible in sandbox

```bash
# Verify data exists
ls -lh benchmarks/citation_react/sandbox/data/

# Should show:
# - emnlp_papers.jsonl (~12MB)
# - best_papers_historical.jsonl (~76KB)
```

## Example Run

```bash
$ inspect eval benchmarks/citation_react/benchmark.py@citation_multiple_choice \
    --model openai/gpt-4o-mini

Target: /Users/.../benchmarks/citation_react/benchmark.py@citation_multiple_choice
Model: openai/gpt-4o-mini
Samples: 20
Sandbox: docker

[1/20] Running sample multiple_choice_sandbox_agent_same_conference_same_year_000
  Agent: Loading historical database...
  Agent: Found 7024 EMNLP papers
  Agent: Searching for similar papers...
  Agent: Analyzing citation patterns...
  Agent: Answer: B

Results:
  Accuracy: 45% (9/20)
  Mean score: 0.45
```

## License

MIT License - see main repository LICENSE file.
