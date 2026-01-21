# *CL conference Award Tier Classification Benchmark - Inspect AI

This benchmark evaluates LLM agents on classifying EMNLP papers into award tiers using a sandboxed environment with accepted-paper data.

## Overview

The agent must assign one of four recognition tiers using only sandbox data:

- **Best**: Best Paper Award winners (paradigm-shifting work)
- **Outstanding**: Outstanding Paper Awards (exceptional contributions)
- **Main**: Main conference track (accepted papers)
- **Findings**: Findings track (good work, below main bar)

The sandbox provides accepted-paper metadata so the agent can compare a target paper to historical patterns and award winners.

You can download our generated questions from: https://huggingface.co/datasets/AIM-Harvard/proof-of-time/tree/main/benchmarks

## Tasks

### 1. Award MCQ (`emnlp_awards_mcq`)

**Question**: Which recognition tier best fits this paper?

**Example**:

```
Title: The Paper Title
Abstract: The abstract text...
Authors: Author A; Author B

Question: Which recognition tier (Findings/Main/Outstanding/Best) best fits this paper?
Options: Findings, Main, Outstanding, Best

Answer with only one word: Best, Outstanding, Main, or Findings.
```

**Dataset**: `mcq_dataset.jsonl` (award-based prompts).

### 2. Historical MCQ (`emnlp_historical_mcq`)

**Question**: Classify historical accepted papers (main/findings) into tiers.

**Notes**: These tasks are sampled from accepted-paper corpora and are designed to test whether the agent can distinguish Main vs. Findings using the same rubric.

**Dataset**: `historical_mcq_dataset.jsonl` (historical main/findings prompts).

### 3. Simple Baseline (`emnlp_awards_mcq_simple`)

A no-tools baseline that answers directly without sandbox inspection.

### Variants

Each core task also has:

- **`_no_offline`** variants (without the shared offline preamble)
- **`_local`** variants (no Docker sandbox; direct file access)

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
# Award MCQ
inspect eval benchmarks/award_react/benchmark.py@emnlp_awards_mcq_task \
  --model openai/gpt-4o-mini

# Historical MCQ
inspect eval benchmarks/award_react/benchmark.py@emnlp_historical_mcq_task \
  --model openai/gpt-4o-mini

# Simple baseline (no tools)
inspect eval benchmarks/award_react/benchmark.py@emnlp_awards_mcq_simple_task \
  --model openai/gpt-4o-mini
```

### With Different Models

```bash
# Claude
inspect eval benchmarks/award_react/benchmark.py@emnlp_awards_mcq_task \
  --model anthropic/claude-3-5-sonnet-20241022

# GPT-4
inspect eval benchmarks/award_react/benchmark.py@emnlp_awards_mcq_task \
  --model openai/gpt-4o

# Limit samples for quick testing
inspect eval benchmarks/award_react/benchmark.py@emnlp_awards_mcq_task \
  --model openai/gpt-4o-mini \
  --limit 5
```

## Sandbox Environment

### Data Files

Expected in `sandbox/data/` (mounted read-only inside the Docker sandbox):

- `historical_papers_2021_2024.jsonl`: accepted-paper corpus with titles, abstracts, authors, venues, tags

### Task Files

Located in `benchmarks/award_react/`:

- `mcq_dataset.jsonl`: award paper prompts
- `historical_mcq_dataset.jsonl`: historical main/findings prompts

### Available Tools

- `bash()`: Execute shell commands
- `python()`: Run Python code for analysis
- `bash_session()`: Persistent shell session
- `text_editor()`: Read/write files
- `think()`: Chain-of-thought reasoning

### Network Access

**Disabled** - agent must use only sandbox data.

## Agent Strategy

The React agent is prompted to:

1. Load the accepted-paper corpus from `sandbox/data/accepted_papers.csv`
2. Inspect Best/Outstanding examples to learn award patterns
3. Compare the target paper's novelty, impact, and quality to those examples
4. Output a single-word tier label (Best, Outstanding, Main, Findings)

## Evaluation Metrics

### Accuracy

- **Award MCQ**: Exact match on the tier label
- **Historical MCQ**: Exact match on the tier label

## Data Sources

Generated from:

- Award spreadsheet(s) of Best/Outstanding winners
- Accepted-papers corpora (ACL/EMNLP/NAACL)
- Dataset building scripts in `dataset_building/` (see `dataset_building/README.md`)

## Troubleshooting

### Missing dataset files

If `accepted_papers.csv` or the JSONL task files are missing, regenerate them using the dataset building scripts referenced in `dataset_building/README.md`.

### Docker issues

```bash
# Check Docker is running
docker ps

# Pull Python image
docker pull python:3.11-slim
```

## License

MIT License - see main repository LICENSE file.
