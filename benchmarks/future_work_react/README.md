# Faculty Future-Work Prediction Benchmark - Inspect AI

This benchmark evaluates LLM agents on predicting 2025 research focus areas for AI faculty and fields using a sandboxed environment with historical publication data.

## Overview

The agent must predict **2025 research trends** for AI professors and fields using only publication data (2018-2025) available in a Docker sandbox:

- **Faculty publications database**: ~16,000 publications across 76 AI faculty with metadata, venues, and field classifications
- **Per-professor CSV exports**: Individual files with complete publication history per professor
- **Evaluation datasets**: 158 total tasks across three task types

This setup mimics real-world research trend prediction: using publication history to predict current research focus.

You can download our generated questions from: https://huggingface.co/datasets/AIM-Harvard/proof-of-time/tree/main/benchmarks

## Tasks

### 1. Professor Field Prediction (`faculty_professor_field_task`)

**Question**: Given a professor's 2025 publications, which field best captures their research agenda?

**Example**:

```
Professor: Aditi Raghunathan
2025 Titles:
- Assessing diversity collapse in reasoning
- Disentangling sequence memorization and general capability in large language models
- Exact unlearning of finetuning data via model merging at scale
- Memorization Sinks: Isolating Memorization during LLM Training

Options: Foundation Models & LLMs, Healthcare & Biomedicine, 
         Climate & Sustainability, Vision & Multimodal Learning

Answer: Foundation Models & LLMs
```

**Dataset**: 73 tasks

### 2. Professor Article Attribution (`faculty_professor_article_task`)

**Question**: Which of these 2025 papers did the professor author/co-author (or None)?

**Example**:

```
Professor: Andrew Ng
Which of the following 2025 papers did they author/co-author?

A. Paper Title 1 (with abstract and venue)
B. Paper Title 2 (with abstract and venue)
C. Paper Title 3 (with abstract and venue)
D. Paper Title 4 (with abstract and venue)
None. None of these papers belong to the professor.

Answer with: A, B, C, D, or None
```

**Dataset**: 76 tasks (includes cases where correct answer is "None")

### 3. Field Focus Prediction (`faculty_field_focus_task`)

**Question**: Given representative 2025 papers, which field do they belong to?

**Example**:

```
Representative 2025 papers:
1. Paper about LLM reasoning (abstract excerpt)
2. Paper about model safety (abstract excerpt)
3. Paper about AI alignment (abstract excerpt)
4. Paper about robustness (abstract excerpt)

Options: Economics Policy & Society, Vision & Multimodal Learning,
         AI Safety & Alignment, Robotics & Embodied AI

Answer: AI Safety & Alignment
```

**Dataset**: 9 tasks (one per major research field)

### 4. All Tasks Combined (`faculty_all_tasks`)

Run all 158 tasks together (73 + 76 + 9).

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
# Professor field prediction
inspect eval benchmarks/future_work_react/benchmark.py@faculty_professor_field_task \
  --model openai/gpt-4o-mini

# Professor article attribution
inspect eval benchmarks/future_work_react/benchmark.py@faculty_professor_article_task \
  --model openai/gpt-4o-mini

# Field focus prediction
inspect eval benchmarks/future_work_react/benchmark.py@faculty_field_focus_task \
  --model openai/gpt-4o-mini
```

### Run All Tasks

```bash
inspect eval benchmarks/future_work_react/benchmark.py@faculty_all_tasks \
  --model openai/gpt-4o-mini
```

### With Different Models

```bash
# Claude
inspect eval benchmarks/future_work_react/benchmark.py@faculty_all_tasks \
  --model anthropic/claude-3-5-sonnet-20241022

# GPT-4
inspect eval benchmarks/future_work_react/benchmark.py@faculty_all_tasks \
  --model openai/gpt-4o

# Limit samples for quick testing
inspect eval benchmarks/future_work_react/benchmark.py@faculty_professor_field_task \
  --model openai/gpt-4o-mini \
  --limit 5
```

### Simple Baseline (No Agent Tools)

```bash
# Fast baseline that uses direct generation without sandbox tools
inspect eval benchmarks/future_work_react/benchmark.py@faculty_professor_field_simple_task \
  --model openai/gpt-4o-mini
```

## Sandbox Environment

### Data Files

Located in `sandbox/data/` (mounted at `/workspace`):

- `historical_papers_2021_2024.jsonl`: 38,330 papers from 2021-2024 with citations

### Sample Data Structure

**faculty_publications.jsonl**:

```json
{
  "professor": "Aditi Raghunathan",
  "scholar_id": "Ch9iRwQAAAAJ",
  "title": "Assessing diversity collapse in reasoning",
  "authors": "Aditi Raghunathan, ...",
  "abstract": "...",
  "venue": "ICLR 2025",
  "year": 2025,
  "field": "Foundation Models & LLMs",
  "matched_keywords": ["language model", "reasoning", "llm"],
  "source_file": "Aditi_Raghunathan_Ch9iRwQAAAAJ.csv"
}
```

### Available Tools

- `bash()`: Execute shell commands
- `python()`: Run Python code for data analysis
- `bash_session()`: Persistent shell session
- `text_editor()`: Read/write files
- `think()`: Chain-of-thought reasoning

### Network Access

**Disabled** - agent must use only sandbox data

## Research Fields

The benchmark uses 9 major AI research fields:

1. **Foundation Models & LLMs** - Language models, transformers, pretraining
2. **AI Safety & Alignment** - Safety, robustness, fairness, interpretability
3. **Robotics & Embodied AI** - Manipulation, control, navigation
4. **Vision & Multimodal Learning** - Computer vision, multimodal models
5. **Economics, Policy & Society** - AI policy, regulation, societal impact
6. **Healthcare & Biomedicine** - Medical AI, drug discovery, clinical applications
7. **Climate & Sustainability** - Climate modeling, environmental applications
8. **Optimization, Theory & ML Systems** - Theory, systems, infrastructure
9. **General AI Research** - Cross-cutting AI research

Fields are assigned using keyword matching on titles, abstracts, and venues.

## Agent Strategy

The React agent is prompted to:

1. **Load publication database** - Explore JSONL and CSV files
2. **Filter for 2025 publications** - Focus on current research trends
3. **Search for professor/field** - Match by name, keywords, or themes
4. **Analyze patterns** - Identify dominant research areas
5. **Make evidence-based predictions** - Base answers only on sandbox data

## Expected Performance

| Task              | Random Baseline | Expected Agent |
| ----------------- | --------------- | -------------- |
| Professor Field   | 25%             | 60-80%         |
| Professor Article | 20%             | 40-60%         |
| Field Focus       | 11%             | 70-90%         |

*Note: Professor field and article tasks are easier than citation prediction because the correct answer exists in the sandbox data.*

## Building the Datasets

The datasets are generated from faculty publication data:

```bash
python dataset_building/generate_faculty_futurework.py \
  --source-dir faculty_publications \
  --sandbox-dir benchmarks/future_work_react/sandbox/data \
  --professor-field-output benchmarks/future_work_react/professor_field_mcq.jsonl \
  --professor-article-output benchmarks/future_work_react/professor_article_mcq.jsonl \
  --field-focus-output benchmarks/future_work_react/field_focus_mcq.jsonl
```

The script:

- Parses professor CSV files with publications
- Infers research fields using keyword matching
- Generates three MCQ datasets with appropriate distractors
- Copies data to sandbox directory

## Troubleshooting

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
ls -lh benchmarks/future_work_react/sandbox/data/

# Should show:
# - faculty_publications.jsonl (~2.5MB)
# - faculty_publications/ (76 CSV files)
```

### Regenerate datasets

```bash
python dataset_building/generate_faculty_futurework.py
```

## Example Run

```bash
$ inspect eval benchmarks/future_work_react/benchmark.py@faculty_professor_field_task \
    --model openai/gpt-4o-mini \
    --limit 3

Target: .../benchmarks/future_work_react/benchmark.py@faculty_professor_field_task
Model: openai/gpt-4o-mini
Samples: 3
Sandbox: docker

[1/3] Running sample 0
  Agent: Loading faculty publications...
  Agent: Filtering for professor "Aditi Raghunathan"
  Agent: Found 16 total publications, 4 from 2025
  Agent: Analyzing field keywords...
  Agent: Answer: Foundation Models & LLMs
  ✓ Correct

Results:
  Accuracy: 100% (3/3)
```

## Directory Structure

```
benchmarks/future_work_react/
├── __init__.py
├── benchmark.py              # Main benchmark tasks
├── docker_check.py           # Docker sandbox tests
├── README.md                 # This file
├── professor_field_mcq.jsonl      # Dataset: Professor → Field
├── professor_article_mcq.jsonl    # Dataset: Professor → Article
├── field_focus_mcq.jsonl          # Dataset: Papers → Field
└── sandbox/
    ├── compose.yaml          # Docker compose config
    └── data/
        ├── faculty_publications.jsonl        # Aggregated data
        └── faculty_publications/             # Per-professor CSVs
            ├── Aditi_Raghunathan_Ch9iRwQAAAAJ.csv
            ├── Yejin_Choi_vhP-tlcAAAAJ.csv
            └── ... (76 total)
```

## License

MIT License - see main repository LICENSE file.
