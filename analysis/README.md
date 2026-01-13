# Agent Query and Cost Analysis

This directory contains analysis of agent tool usage, token consumption, and estimated costs from Inspect AI evaluation runs.

## Files

- **agent_query_analysis.csv** - Raw data with per-run statistics (127 eval runs)
- **correlation_matrix.csv** - Correlation coefficients between metrics
- **AGENT_QUERY_ANALYSIS.md** - Comprehensive analysis report with findings and recommendations

## Quick Start

### Generate Analysis

```bash
# Analyze all .eval files in logs/
uv run python scripts/analyze_agent_queries.py --output analysis/agent_query_analysis.csv

# Compute correlations and statistics
uv run python scripts/correlation_analysis.py --input analysis/agent_query_analysis.csv
```

### Analyze Specific Log Directories

```bash
# Analyze message limit sweeps (when available)
uv run python scripts/analyze_agent_queries.py \
    --log-dirs logs/ablations_msg15 logs/ablations_msg30 logs/ablations_msg50 \
    --output analysis/msg_limit_analysis.csv
```

## Key Findings

### Strong Correlations
- **Tool calls ↔ Cost:** r = 0.779 (strong)
- **Tool calls ↔ Total tokens:** r = 0.886 (strong)
- **Tool calls ↔ Messages:** r = 0.993 (very strong)

### Cost Insights
- Average cost per run: $0.15
- Citation tasks most expensive: $0.48/run
- Simple baselines nearly free: < $0.01/run

### Model Efficiency
- Claude Opus 4.5: Thorough (71.5 calls) but expensive ($0.51/run)
- GPT-5.1: Balanced (18.1 calls, $0.28/run)
- GPT-5 Nano: Budget-friendly (5.6 calls, $0.01/run)

## Data Schema

### agent_query_analysis.csv

| Column | Description |
|--------|-------------|
| log_file | Eval log filename |
| task | Benchmark task name |
| model | Model identifier |
| message_limit | Max message limit (if available) |
| status | success/error/cancelled |
| num_samples | Number of samples in run |
| total_messages | Total messages exchanged |
| total_tool_calls | Total agent tool invocations |
| avg_messages_per_sample | Messages per sample |
| avg_tool_calls_per_sample | Tool calls per sample |
| input_tokens | Total input tokens |
| output_tokens | Total output tokens |
| total_tokens | Total tokens (input + output) |
| cache_read_tokens | Cache read tokens |
| cache_write_tokens | Cache write tokens |
| estimated_cost_usd | Estimated cost in USD |
| duration_seconds | Run duration in seconds |

## Scripts

### analyze_agent_queries.py

Processes .eval files and extracts:
- Tool call counts
- Message statistics
- Token usage
- Cost estimates

### correlation_analysis.py

Computes:
- Correlation matrices
- Summary statistics
- Task-level analysis
- Model-level analysis

## Example Usage

```python
import pandas as pd

# Load analysis results
df = pd.read_csv('analysis/agent_query_analysis.csv')

# Filter successful runs
df_success = df[df['status'] == 'success']

# Top 5 most expensive tasks
top_tasks = df_success.groupby('task')['estimated_cost_usd'].sum().sort_values(ascending=False).head(5)
print(top_tasks)

# Cost per tool call by model
df_success['cost_per_call'] = df_success['estimated_cost_usd'] / df_success['total_tool_calls']
model_efficiency = df_success.groupby('model')['cost_per_call'].mean().sort_values()
print(model_efficiency)
```

## Notes

- Cost estimates based on public API pricing as of Jan 2026
- Actual costs may vary based on volume discounts, regional pricing, etc.
- Tool call counts may underestimate if some tools are not properly detected
- Message limit data requires logs from specific ablation directories
