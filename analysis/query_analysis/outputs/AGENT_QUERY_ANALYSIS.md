# Agent Query, Token Usage, and Cost Analysis

**Generated:** 2026-01-02
**Data Source:** 127 evaluation logs from logs/ directory
**Analysis Scripts:** `scripts/analyze_agent_queries.py`, `scripts/correlation_analysis.py`

## Executive Summary

This analysis examines the relationship between agent tool calls (queries), token usage, and estimated costs across 127 evaluation runs from various benchmarks and models.

### Key Findings

1. **Strong Correlation Between Tool Calls and Cost**
   - Tool calls strongly correlate with total cost (r = 0.779)
   - Each additional tool call increases token usage substantially
   - Average: 21.13 tool calls per run, costing $0.1471

2. **Most Expensive Tasks**
   - Citation tasks require the most tool calls (60-147 calls)
   - Award prediction tasks are moderately expensive (20-40 calls)
   - Simple baseline tasks require minimal tool usage (0 calls)

3. **Model Efficiency Varies**
   - Claude Opus 4.5: High cost ($0.51/run) but thorough (71.5 calls avg)
   - GPT-5.1: Balanced cost ($0.28/run) with moderate usage (18.1 calls)
   - GPT-5 Nano: Low cost ($0.01/run) with minimal usage (5.6 calls)

## Overall Statistics

### Evaluation Runs
- **Total runs:** 127
- **Successful:** 83 (65.4%)
- **Failed/Error:** 36 (28.3%)
- **Cancelled:** 8 (6.3%)

### Agent Query Statistics
- **Total tool calls:** 1,754
- **Average per run:** 21.13 calls
- **Average per sample:** 4.51 calls
- **Maximum in single run:** 147 calls (citation_multiple_choice with Claude Opus)

### Token Usage
- **Total tokens:** 15,422,358
- **Average per run:** 185,812 tokens
- **Average input:** 82,125 tokens
- **Average output:** 5,686 tokens
- **Input:Output ratio:** ~14:1

### Cost Analysis
- **Total estimated cost:** $12.21
- **Average per run:** $0.1471
- **Range:** $0.0000 - $1.1557
- **Median:** ~$0.15

## Correlation Analysis

### Tool Calls vs Other Metrics

| Metric | Correlation | Strength |
|--------|-------------|----------|
| Total Messages | 0.993 | **Very Strong** |
| Avg Tool Calls/Sample | 0.968 | **Very Strong** |
| Avg Messages/Sample | 0.964 | **Very Strong** |
| Total Tokens | 0.886 | **Strong** |
| Output Tokens | 0.825 | **Strong** |
| Estimated Cost | 0.779 | **Strong** |
| Input Tokens | 0.384 | Weak |
| Duration | 0.335 | Weak |

**Interpretation:** Tool calls are the primary driver of cost. Each tool call generates multiple messages (system, tool call, tool response), which directly increases token consumption and cost.

### Cost vs Other Metrics

| Metric | Correlation | Strength |
|--------|-------------|----------|
| Tool Calls | 0.779 | **Strong** |
| Total Messages | 0.774 | **Strong** |
| Total Tokens | 0.745 | **Strong** |
| Avg Tool Calls/Sample | 0.739 | **Strong** |
| Avg Messages/Sample | 0.733 | **Strong** |
| Output Tokens | 0.623 | Moderate |
| Input Tokens | 0.295 | Weak |
| Duration | 0.135 | Weak |

**Interpretation:** Cost correlates more with the number of interactions (tool calls, messages) than with raw token counts. This suggests that multi-turn agent reasoning is the main cost driver.

## Analysis by Task

### Top 10 Tasks by Tool Calls

| Task | Runs | Avg Calls | Total Calls | Avg Cost | Total Cost |
|------|------|-----------|-------------|----------|------------|
| **emnlp_awards_mcq_task** | 16 | 27.5 | 440 | $0.21 | $3.28 |
| **citation_multiple_choice** | 5 | 60.8 | 304 | $0.48 | $2.40 |
| **emnlp_historical_mcq_task** | 7 | 38.9 | 272 | $0.31 | $2.18 |
| **faculty_professor_field_task** | 5 | 31.0 | 155 | $0.17 | $0.83 |
| **faculty_professor_article_task** | 5 | 30.6 | 153 | $0.19 | $0.96 |
| **sota_bucket_task** | 5 | 24.8 | 124 | $0.15 | $0.76 |
| **citation_ranking** | 3 | 41.0 | 123 | $0.27 | $0.80 |
| **faculty_field_focus_task** | 5 | 20.0 | 100 | $0.11 | $0.57 |
| **citation_bucket_prediction** | 2 | 27.0 | 54 | $0.15 | $0.29 |
| **historical_awards_mcq_acl2025** | 2 | 8.0 | 16 | $0.00 | $0.01 |

### Task Complexity Tiers

**High Complexity (40+ tool calls avg):**
- `citation_multiple_choice`: 60.8 calls, $0.48/run
- `citation_ranking`: 41.0 calls, $0.27/run
- `emnlp_historical_mcq_task`: 38.9 calls, $0.31/run

**Medium Complexity (20-40 tool calls):**
- `faculty_professor_field_task`: 31.0 calls, $0.17/run
- `faculty_professor_article_task`: 30.6 calls, $0.19/run
- `emnlp_awards_mcq_task`: 27.5 calls, $0.21/run
- `citation_bucket_prediction`: 27.0 calls, $0.15/run
- `sota_bucket_task`: 24.8 calls, $0.15/run

**Low Complexity (< 20 tool calls):**
- `faculty_field_focus_task`: 20.0 calls, $0.11/run
- `historical_awards_mcq_acl2025`: 8.0 calls, $0.00/run
- Simple tasks: 0 calls (baseline prompting)

## Analysis by Model

### Model Performance Summary

| Model | Runs | Avg Calls | Total Cost | Avg Cost/Run | Avg Duration |
|-------|------|-----------|------------|--------------|--------------|
| **Claude Opus 4.5** | 4 | 71.5 | $2.05 | $0.51 | 119.8s |
| **Claude Haiku 4.5** | 4 | 58.8 | $1.41 | $0.35 | 62.0s |
| **Claude Sonnet 4.5** | 5 | 39.0 | $0.82 | $0.16 | 77.8s |
| **GPT-5.1** | 18 | 18.1 | $5.10 | $0.28 | 22.1s |
| **GPT-5 Mini** | 20 | 14.0 | $0.71 | $0.04 | 84.0s |
| **GPT-5 Nano** | 16 | 5.6 | $0.08 | $0.01 | 78.6s |
| **Gemini 2.5 Pro** | 7 | 21.1 | $1.45 | $0.21 | 83.1s |
| **Gemini 2.5 Flash** | 2 | 35.0 | $0.46 | $0.23 | 66.0s |
| **Gemini 3 Pro** | 2 | 62.0 | $0.13 | $0.06 | 279.5s |

### Model Efficiency Insights

**Most Thorough (highest tool usage):**
1. Claude Opus 4.5: 71.5 calls/run
2. Gemini 3 Pro: 62.0 calls/run
3. Citation tasks with Claude Haiku: 58.8 calls/run

**Most Efficient (lowest cost per call):**
1. GPT-5 Nano: $0.0018/call
2. GPT-5 Mini: $0.0025/call
3. Gemini 3 Pro: $0.0010/call

**Best Value (balanced cost and thoroughness):**
1. Claude Sonnet 4.5: 39 calls, $0.16/run
2. Gemini 2.5 Pro: 21.1 calls, $0.21/run
3. GPT-5.1: 18.1 calls, $0.28/run

## Cost Optimization Recommendations

### 1. Task-Specific Model Selection

**For Citation Tasks (high complexity):**
- Use Claude Sonnet 4.5 or Gemini 2.5 Pro for balanced cost/performance
- Avoid Claude Opus 4.5 unless maximum accuracy is required ($0.51/run)

**For Award/Historical Tasks (medium complexity):**
- GPT-5.1 provides good balance ($0.28/run)
- GPT-5 Mini for budget-conscious runs ($0.04/run)

**For Simple Tasks:**
- Use GPT-5 Nano or GPT-5 Mini (< $0.01/run)
- Consider zero-shot prompting without React agent

### 2. Message Limit Tuning

Current analysis shows no message limit data (all runs from main logs/).

**Recommendation:** Re-run analysis on logs from:
- `logs/ablations_msg15/`
- `logs/ablations_msg30/`
- `logs/ablations_msg50/`

This will reveal optimal message limits for cost/accuracy tradeoff.

### 3. Cost-Saving Strategies

**Reduce Tool Calls:**
- Improve offline prompts to encourage fewer, more targeted queries
- Provide better context in initial prompt to reduce exploration
- Cache intermediate results to avoid repeated searches

**Optimize Token Usage:**
- Use shorter examples in few-shot prompts
- Compress dataset descriptions
- Leverage prompt caching where available (GPT-5.1 shows 78K cache reads)

**Batch Processing:**
- Group similar samples to share context
- Use simple baselines for easy samples, agents only for hard cases

## Data Files

- **Raw data:** `analysis/agent_query_analysis.csv`
- **Correlation matrix:** `analysis/correlation_matrix.csv`
- **Analysis scripts:**
  - `scripts/analyze_agent_queries.py`
  - `scripts/correlation_analysis.py`

## Methodology

### Data Collection
1. Parsed 127 .eval files from `logs/` directory using Inspect AI's `read_eval_log()`
2. Extracted: task, model, status, samples, messages, tool calls, token usage, timestamps
3. Calculated estimated costs using model-specific pricing tables

### Tool Call Counting
Tool calls identified by:
- Checking `message.tool_calls` attribute
- Counting bash, python, text_editor, and other tool invocations
- Only counting actual tool executions (not tool responses)

### Cost Estimation
Pricing based on public API rates (as of Jan 2026):
- GPT-5 series: OpenAI pricing
- Claude 4.5 series: Anthropic/Vertex AI pricing
- Gemini series: Google Cloud pricing
- Groq models: Typical inference costs (many free tier available)

**Note:** Actual costs may vary based on:
- Volume discounts
- Promotional pricing
- Cache hit rates
- Regional pricing differences

## Future Analysis

### Recommended Extensions

1. **Message Limit Sweep Analysis**
   - Compare msg15 vs msg30 vs msg50
   - Find optimal limit for each task type

2. **Accuracy vs Cost Tradeoff**
   - Plot accuracy scores against cost
   - Identify pareto-optimal configurations

3. **Time-Series Analysis**
   - Track cost trends over time
   - Identify tasks with increasing complexity

4. **Failure Analysis**
   - Examine 36 failed runs
   - Determine if failures correlate with tool usage patterns

5. **Ablation Study**
   - Compare offline prompt vs no-offline variants
   - Measure impact of prompt engineering on cost
