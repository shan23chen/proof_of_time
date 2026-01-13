---
license: mit
task_categories:
- question-answering
- text-classification
tags:
- academic-papers
- llm-agents
- benchmarking
- inspect-ai
- react-agents
- citations
- research-trends
size_categories:
- 1K<n<10K
language:
- en
---

# Proof of Time: Academic Paper Analysis Benchmarks

This dataset contains benchmarks for evaluating LLM agents on academic paper analysis tasks that require understanding research trends, citations, and future directions. All evaluation data uses **post-training-cutoff** (2025) papers to avoid data contamination.

## Dataset Description

**Paper**: *Proof of Time: Benchmarking LLM Agents on Academic Paper Analysis* (Under Review)

**Repository**: [https://github.com/shan23chen/proof_of_time](https://github.com/shan23chen/proof_of_time)

This dataset includes:
1. **Benchmark Tasks** (3.8 MB): JSONL files with multiple-choice questions and evaluation samples
2. **Sandbox Data** (66 MB): Historical paper data, faculty publications, and SOTA metrics for agent evaluation

### Why "Proof of Time"?

The benchmark suite focuses on temporal reasoning: agents must analyze historical patterns to make predictions about future research directions, award recipients, and citation impact. Tasks require genuine understanding of research trends rather than memorization.

## Dataset Structure

### Benchmarks Directory (3.8 MB)

Contains 4 task families with 10 evaluation datasets:

#### Award Prediction (641 KB)
Predict which papers will win best paper awards at top NLP conferences.

- `pre-cutoff_mcq.jsonl` (421 KB): Pre-2025 conference awards (ACL/EMNLP/NAACL 2018-2024)
- `post-cutoff_emnlp.jsonl` (29 KB): Post-2025 EMNLP awards
- `post-cutoff_acl_naacl.jsonl` (191 KB): Post-2025 ACL/NAACL awards

#### Citation Forecasting (2.7 MB)
Predict future citation counts for recently published papers.

- `multiple_choice.jsonl` (1.1 MB): Predict highest-cited paper among choices
- `ranking.jsonl` (1.2 MB): Rank papers by predicted citation counts
- `bucket_prediction.jsonl` (368 KB): Classify papers into citation ranges (0-1, 1-5, 5-10, 10-50, 50+)

#### Faculty Future Work (469 KB)
Predict research directions of AI faculty members based on publication history.

- `professor_field_mcq.jsonl` (49 KB): Predict research field for professor's future work
- `professor_article_mcq.jsonl` (404 KB): Predict which article professor would author
- `field_focus_mcq.jsonl` (16 KB): Classify research focus by field

#### SOTA Forecasting (26 KB)
Predict state-of-the-art performance ranges on ML benchmarks.

- `mcq_dataset.jsonl` (26 KB): Predict benchmark performance buckets (0-20, 20-40, 40-60, 60-80, 80-100)

### Sandbox Data Directory (66 MB)

Reference data for ReAct agents to query during evaluation:

- `citation/historical_papers_2021_2024.jsonl` (21 MB): Historical papers with citation counts
- `award/accepted_papers.csv` (19 MB): EMNLP accepted papers (2018-2025)
- `faculty/faculty_publications.jsonl` (20 MB): Aggregated publications for 76 AI faculty
- `faculty/faculty_publications.tar.gz` (5.9 MB): Individual CSV files per faculty member
- `sota/sota_metrics.json` (8.7 KB): Frontier model benchmark scores (October 2025)

## Usage

### With Inspect AI

```python
from datasets import load_dataset
from inspect_ai import eval
from inspect_ai.dataset import json_dataset

# Load dataset from HuggingFace
ds = load_dataset("AIM-Harvard/proof-of-time")

# Run evaluation with Inspect AI
eval(
    task="your_benchmark.py@task_name",
    model="openai/gpt-5-mini-2025-08-07",
    limit=5
)
```

### Quick Start

```bash
# Install dependencies
pip install inspect-ai datasets

# Clone repository with benchmark implementations
git clone https://github.com/shan23chen/proof_of_time.git
cd proof_of_time

# Download dataset
from datasets import load_dataset
ds = load_dataset("AIM-Harvard/proof-of-time")

# Run evaluation
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_task \
    --model openai/gpt-5-mini-2025-08-07 \
    --limit 5
```

## Dataset Fields

Each benchmark JSONL file contains samples with:

- **`question`**: Task prompt for the agent
- **`answer`**: Correct answer (for evaluation)
- **`choices`**: Multiple choice options (if applicable)
- **`metadata`**: Additional context (paper titles, years, venues, authors, etc.)

Example from award prediction:

```json
{
  "question": "Which recognition tier (Findings/Main/Outstanding/Best) best fits the paper?",
  "context": "{title}+{abstract}+{author}"
  "answer": "A",
  "choices": ["Best", "Outstanding", "Main", "Findings"],
}
```

## Benchmark Design

- **ReAct Agents**: Agents use tools (bash, Python, text editor) to explore sandboxed paper datasets
- **Sandboxed Environments**: Docker containers with read-only paper data (no internet access)
- **Offline Prompt**: Custom "Antigravity" prompt inspired by principles of focused exploration
- **Multiple Variants**: Each task has standard (agent), simple (zero-shot), and no-offline-prompt versions

## Supported Models

The benchmark suite has been tested with:

- **OpenAI**: gpt-5.2, gpt-5.1, gpt-5-mini, gpt-5-nano
- **Google**: gemini-3-pro, gemini-3-flash, vertex/gemini-2.5-pro, vertex/gemini-2.5-flash
- **Anthropic**: vertex/claude-opus-4-5, vertex/claude-sonnet-4-5, vertex/claude-haiku-4-5

## Data Sources

- **Award Predictions**: ACL Anthology, EMNLP/ACL/NAACL conference proceedings
- **Citation Forecasting**: Google Scholar citation counts
- **Faculty Predictions**: AI faculty CVs and publication records
- **SOTA Forecasting**: Papers with Code leaderboards

## License

- **Code**: MIT License
- **Data**: Derived from publicly available academic papers and conference proceedings

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{ye2026prooftimebenchmarkevaluating,
      title={Proof of Time: A Benchmark for Evaluating Scientific Idea Judgments}, 
      author={Bingyang Ye and Shan Chen and Jingxuan Tu and Chen Liu and Zidi Xiong and Samuel Schmidgall and Danielle S. Bitterman},
      year={2026},
      eprint={2601.07606},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.07606}, 
}
```

For the dataset:

```bibtex
@dataset{proof-of-time-dataset-2025,
  title={Proof of Time: Academic Paper Analysis Benchmarks},
  author={AIM Harvard},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/AIM-Harvard/proof-of-time}
}
```

## Additional Resources

- **GitHub Repository**: [https://github.com/shan23chen/proof_of_time](https://github.com/shan23chen/proof_of_time)
- **Documentation**: See repository README for detailed usage
- **Setup Guide**: [SETUP.md](https://github.com/shan23chen/proof_of_time/blob/main/SETUP.md)
- **Paper**: [Link when available]

## Contact

- **Issues**: [https://github.com/shan23chen/proof_of_time/issues](https://github.com/shan23chen/proof_of_time/issues)
- **Email**: aim@seas.harvard.edu
- **Organization**: AIM Harvard

## Updates

- **2026-01-08**: Initial release (Tiers 1-2: benchmarks + sandbox data, 69.8 MB total)
