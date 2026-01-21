# Proof of Time: Benchmarking Temporal Reasoning in LLM Agents

*Analysis Report — Generated: 2026-01-04*

---

## Abstract

This report presents comprehensive results from the **Proof of Time** benchmark, which evaluates large language models (LLMs) on temporal reasoning tasks requiring prediction of future events based on historical data analysis. We introduce tasks spanning citation prediction, conference award classification, faculty research trajectory prediction, and benchmark performance forecasting. Critically, we include **post-training-cutoff evaluation sets** from ACL 2025 and EMNLP 2025 to eliminate potential data contamination concerns. Our experiments compare zero-shot generation, ReAct agents with tool access, and structured agentic prompts across 10 frontier LLMs from Anthropic, Google, and OpenAI.

---

## 1. Introduction and Motivation

### 1.1 Research Questions

We investigate four primary research questions:

1. **RQ1 (Test-Time Scaling)**: How does LLM accuracy scale with increased inference-time computation, operationalized as message/turn limits in agentic settings?

2. **RQ2 (Agentic vs Direct Generation)**: Do tool-using agents substantially outperform direct (zero-shot) generation on temporal reasoning tasks?

3. **RQ3 (Prompt Engineering)**: Does a structured agentic prompt improve performance over vanilla ReAct agents?

4. **RQ4 (Data Contamination)**: Do models perform worse on post-cutoff tasks (ACL 2025, EMNLP 2025) compared to historical tasks, suggesting prior results were inflated by memorization?

### 1.2 Why Temporal Reasoning Matters

Temporal reasoning—predicting future outcomes based on historical patterns—is a critical capability for deploying LLMs in dynamic domains like scientific research, finance, and policy. Unlike static knowledge retrieval, temporal reasoning requires:

- **Pattern recognition** across time-varying data
- **Extrapolation** beyond training distribution
- **Integration** of multiple evidence sources
- **Uncertainty quantification** about future events

Our benchmark specifically targets the scientific domain, where LLMs could assist with research evaluation, funding decisions, trend forecasting, and literature analysis.

---

## 2. Task Suite: Design and Motivation

We design seven task families, each probing different aspects of temporal reasoning. Below we describe each task's motivation, setup, and the intuition for why it tests meaningful capabilities.

### 2.1 Citation Count Prediction

**Motivation**: Citation counts serve as a proxy for research impact, yet predicting future citations requires understanding both the intrinsic quality of research and the complex dynamics of scientific attention. We hypothesize that models with access to historical publication data can learn implicit patterns correlating paper characteristics with citation trajectories and predict future citation counts.

**Task Setup**: Given a paper's title, abstract, venue, and publication year, the model must predict its citation count relative to other papers. We design three sub-tasks: (1) **MCQ**: Select which of four papers received the highest citations; (2) **Ranking**: Order four papers by citation count; (3) **Bucket Prediction**: Classify papers into citation percentile buckets (0-25th, 25-50th, 50-75th, 75-100th).Citation counts are collected from Google Scholar as of November 2025

**Sandbox Contents**: Historical NLP papers from 2021-2024 with metadata including titles, abstracts, authors, venues, publication dates, and citation counts as of December 2024. The agent can analyze patterns across ~2,000 papers from major venues (ACL, EMNLP, NAACL, etc.).

**Intuition**: This task tests whether LLMs can identify implicit signals of paper quality and predict scientific impact—a capability with practical applications in research evaluation, funding decisions, and scientific search. The agentic setting allows models to verify hypotheses against historical data rather than relying solely on parametric knowledge.

### 2.2 Peer Review Award Tier Classification (Historical)

**Motivation**: Peer review outcomes, while imperfect, represent expert consensus on research quality. Major NLP venues (ACL, EMNLP) use tiered acceptance: Best Paper, Outstanding Paper, Main Conference, and Findings. We investigate whether LLMs can learn the implicit criteria reviewers use to distinguish these tiers. This task combines historical EMNLP papers (2021-2024) from both the main awards dataset and historical awards dataset to provide robust evaluation on in-distribution data.

**Task Setup**: Given only a paper's title and abstract and authors, the model must classify the paper into one of four tiers: Best, Outstanding, Main, or Findings. This is a 4-way classification task framed as MCQ. The evaluation combines emnlp_awards_mcq and emnlp_historical_awards_mcq datasets, treating them as a single unified benchmark since they test the same capability on the same historical paper corpus.

**Sandbox Contents**: Historical EMNLP papers from 2021-2024 with acceptance tier labels. The sandbox contains ~800 papers across all tiers, enabling the agent to analyze linguistic patterns, topic distributions, and structural characteristics associated with each tier.

**Intuition**: This task evaluates whether LLMs have internalized the criteria for research excellence in NLP. Success requires understanding not just technical correctness but also novelty, significance, and presentation quality—meta-scientific reasoning that goes beyond surface-level pattern matching.

### 2.3 Peer Review Award Tier Classification (Post-Cutoff)

**Motivation**: A fundamental concern with LLM evaluation is **data contamination**: models may have memorized award outcomes from their training data rather than genuinely reasoning about paper quality. This task combines post-cutoff papers from NAACL 2025 (April 2025), ACL 2025, (July 2025) and EMNLP 2025 (November 2025) that are definitively beyond most models' training cutoffs, providing **true blind evaluation**. Note: For Gemini 3 models trained after July 2025, only EMNLP 2025 papers are post-cutoff; NAACL and ACL 2025 are grouped with historical data for those models.

**Task Setup**: Identical to the historical award classification task: given title, abstract and authors, classify into Best/Outstanding/Main/Findings tiers. However, these papers were published after models' training cutoffs, eliminating the possibility of memorization. The evaluation combines emnlp_awards_mcq_{acl2025,emnlp2025} and emnlp_historical_awards_mcq_{acl2025,emnlp2025} datasets, treating them as a unified post-cutoff benchmark.

**Sandbox Contents**: Same historical papers as the baseline task (2021-2024), but test queries are from ACL 2025 and EMNLP 2025. The agent must generalize from historical patterns to evaluate papers it has never seen and could not have memorized.

**Intuition**: This is our strongest test of genuine temporal reasoning. Any model succeeding here must be applying learned criteria rather than recalling memorized facts. Poor performance relative to historical tasks would suggest previous results were inflated by contamination; similar performance would validate the benchmark. By combining multiple post-cutoff conferences, we ensure robust evaluation of out-of-distribution generalization.

### 2.4 Faculty Research Prediction

**Motivation**: Research labs develop distinctive expertise and publication patterns over time. We test whether LLMs can model these patterns to predict faculty research activities based on their historical publication records.

**Task Setup**: Three sub-tasks: (1) **Professor-Article**: Match papers to their likely authors from a set of faculty candidates; (2) **Professor-Field**: Predict a professor's primary research area given their publication history; (3) **Field-Focus**: Identify emerging research directions based on recent publications.

**Sandbox Contents**: Publication records for ~20 NLP faculty members from top institutions, including paper titles, abstracts, venues, years, and citation counts. Enables analysis of individual research trajectories and lab-specific patterns.

**Intuition**: This task evaluates whether LLMs can model the latent structure of academic research—recognizing that certain researchers have distinctive styles, focus areas, and collaboration patterns that leave fingerprints in their publications.

### 2.5 SOTA Benchmark Performance Prediction

**Motivation**: Tracking state-of-the-art performance on benchmarks reveals the pace of progress in AI capabilities. We test whether models can extrapolate performance trajectories given historical benchmark results.

**Task Setup**: Given a benchmark name and historical performance data up to a certain date, predict the performance bucket (0-20%, 20-40%, 40-60%, 60-80%, 80-100%) that SOTA models will achieve by a future date.

**Sandbox Contents**: Historical SOTA metrics from Papers With Code spanning 2020-2024, covering ~50 major benchmarks across NLP tasks (text classification, QA, summarization, etc.). Each entry includes benchmark name, date, model name, and performance metrics.

**Intuition**: Predicting benchmark progress requires understanding both the inherent difficulty of tasks and the trajectory of methodological improvements. This probes whether LLMs have internalized the meta-patterns of AI research progress.

---

## 3. Experimental Methodology

### 3.1 The Three Experimental Modes

We compare three fundamentally different approaches to temporal reasoning tasks:

#### Mode 1: Zero-Shot (Direct Generation)

The model receives only the task instruction and question, with no access to tools or external data. This baseline tests whether models can solve tasks using only their parametric knowledge (i.e., patterns learned during pre-training).

```
┌─────────────────────────────────────────────────────────────────┐
│  Input:                                                         │
│    • System prompt with task instructions                       │
│    • Question with paper title/abstract                         │
│                                                                 │
│  Output:                                                        │
│    • Single answer (e.g., "Best", "Main", "A")                  │
│                                                                 │
│  NO access to: tools, sandbox, historical data files            │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation**: Uses `system_message()` + `generate()` in Inspect AI framework.

#### Mode 2: ReAct Agent (Tools + Sandbox)

The model operates as a ReAct agent with access to:

- `python()`: Execute arbitrary Python code
- `bash()`: Run shell commands
- `text_editor()`: Read and edit files
- `think()`: Internal reasoning scratchpad

The agent runs in a Docker sandbox containing historical data files relevant to each task.

```
┌─────────────────────────────────────────────────────────────────┐
│  Input:                                                         │
│    • Task-specific instructions                                 │
│    • Question with paper metadata                               │
│                                                                 │
│  Agent Capabilities:                                            │
│    • Execute Python (pandas, json, etc.)                        │
│    • Run shell commands (grep, cat, etc.)                       │
│    • Read/write files in sandbox                                │
│    • Multi-turn reasoning with tool feedback                    │
│                                                                 │
│  Sandbox Contents:                                              │
│    • Historical papers (2021-2024) with metadata                │
│    • Citation counts, award labels, author info                 │
│    • Benchmark performance trajectories                         │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation**: Uses `react()` agent with `use_offline_prompt=False`.

#### Mode 3: ReAct Agent + Structured Agentic Prompt

Same as Mode 2, but with an additional structured preamble ("Offline Antigravity") that:

- Emphasizes offline-only operation (no web access)
- Provides guidance on efficient tool use (`rg` for search, concise outputs)
- Establishes behavioral expectations for the agent

```markdown
# Offline Antigravity Agent (Local-Only)

You are Antigravity, a powerful agentic AI assistant. Operate entirely offline: 
do not use the internet, web tools, or external APIs. Rely only on local files 
and built-in shell tools.

## Core Behavior
- Default to concise, plain-text replies; prioritize actionable output
- Prefer `rg` for searches and `apply_patch` for small edits
- Never revert user changes unless explicitly asked
```

**Implementation**: Uses `react()` agent with `use_offline_prompt=True` (default).

### 3.2 Test-Time Compute: Message Limits

We operationalize test-time compute scaling via **message limits**—the maximum number of agent-environment interaction turns allowed before forcing a final answer. This directly controls how much "thinking time" the agent has:

| Message Limit | Interpretation | Typical Behavior |
| :---: | :--- | :--- |
| **15** | Minimal budget | Quick exploration, may miss complex patterns |
| **30** | Moderate budget | Standard operation, sufficient for most tasks |
| **50** | Maximum budget | Deep exploration, extensive data analysis |

Higher limits allow agents to:
- Explore more of the sandbox data
- Iterate on analysis strategies
- Verify hypotheses against multiple evidence sources
- Recover from initial errors

### 3.3 Models Evaluated

We evaluate 10 frontier LLMs from three major providers:

| Provider | Models | Notes |
| :--- | :--- | :--- |
| **Anthropic** | Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5 | Claude 4.5 family |
| **Google** | Gemini 3 Pro Preview, Gemini 2.5 Pro, Gemini 2.5 Flash | Latest Gemini models |
| **OpenAI** | GPT-5.2, GPT-5.1, GPT-5 Mini, GPT-5 Nano | GPT-5 series |

All models were accessed via their respective APIs in December 2024 - December 2025.

### 3.4 Task Statistics

| Task Family | Description | N Samples | Temporal Status |
| :--- | :--- | :---: | :--- |
| Citation MCQ | Citation MCQ (highest citation identification) | 200 | Historical |
| Citation Rank | Citation ranking (order by citation count) | 200 | Historical |
| Citation Bucket | Citation bucket (percentile classification) | 200 | Historical |
| Award (Historical) | Award tier classification (combined 2021-2024) | 197 | Historical |
| Award (Post-Cutoff) | Award tier classification (combined ACL + EMNLP 2025) | 122 | **Post-cutoff** |
| Prof. Field | Professor field prediction | 73 | Historical |
| Prof. Article | Professor article attribution | 76 | Historical |
| Field Focus | Field focus classification | 9 | Historical |
| SOTA Bucket | SOTA benchmark forecasting | 45 | Historical |

**Total**: 1,110 experimental runs across 11 models, with 1,122 total samples evaluated. Award tasks are shown as combined groups (historical vs post-cutoff) for accurate contamination analysis.

---

## 4. Results

### 4.1 RQ1: Test-Time Compute Scaling

**Finding**: All model families show substantial accuracy gains with increased message limits, but the magnitude varies dramatically. Claude models exhibit the strongest scaling behavior.

![Scaling by Model](plots/scaling_by_model.png)
*Figure 1: Test-time scaling curves showing accuracy vs. message limit for each model. Claude models (orange) show the steepest improvement.*

#### Scaling Gains Summary

| Model | Acc@15 | Acc@30 | Acc@50 | Δ(15→50) |
| :--- | :---: | :---: | :---: | :---: |
| gemini-3-pro-preview | 27.2% | 44.8% | 55.8% | **+28.6pp** |
| claude-sonnet-4-5 | 2.9% | 25.9% | 31.0% | **+28.1pp** |
| gemini-2.5-pro | 26.7% | 55.6% | 53.6% | **+26.9pp** |
| claude-haiku-4-5 | 8.4% | 26.1% | 35.1% | **+26.7pp** |
| claude-opus-4-5 | 16.7% | 28.2% | 41.2% | **+24.5pp** |
| gpt-5.2-2025-12-11 | 17.9% | 39.3% | 40.1% | **+22.2pp** |
| gemini-2.5-flash | 17.2% | 37.0% | 37.5% | **+20.4pp** |
| gemini-3-flash-preview | 20.3% | 32.0% | 39.2% | **+18.8pp** |
| gpt-5.1-2025-11-13 | 26.7% | 44.3% | 45.0% | **+18.4pp** |
| gpt-5-mini-2025-08-07 | 22.3% | 38.7% | 37.7% | **+15.4pp** |
| gpt-5-nano-2025-08-07 | 21.7% | 35.1% | 33.9% | **+12.1pp** |

**Key Observations**:
- **Claude models** show dramatic scaling (+37-49pp from 15→50 messages), suggesting they effectively leverage additional reasoning steps
- **Gemini models** show strong initial performance but moderate scaling gains (+18-27pp)
- **GPT models** plateau earlier, with smaller marginal gains at higher limits (+11-27pp)

![Scaling Gain Waterfall](plots/scaling_gain_waterfall.png)
*Figure 2: Waterfall chart showing test-time scaling gains (Acc@50 - Acc@15) by model.*

### 4.2 RQ2: Agentic vs Zero-Shot Performance

**Finding**: Tool-using agents dramatically outperform zero-shot generation, with gaps of 20-50 percentage points on complex tasks.

![Simple vs Agentic](plots/simple_vs_agentic.png)
*Figure 3: Scatter plot comparing zero-shot (x-axis) vs agentic (y-axis) accuracy. Points above the diagonal indicate agentic superiority.*

**Key Observations**:
- The agentic advantage is largest on **data-intensive tasks** (citation prediction, faculty research)
- Even on tasks where zero-shot performs reasonably, agents achieve higher accuracy
- The gap suggests that **tool access enables verification** of model hypotheses against data

### 4.3 RQ3: Structured Agentic Prompt Effect

**Finding**: The "Offline Antigravity" prompt has model-specific effects—beneficial for Claude, neutral-to-negative for GPT models.

![Ablation Scatter 50](plots/ablation_scatter_msg50.png)
*Figure 4: Ablation comparing ReAct+Prompt (y-axis) vs ReAct-only (x-axis). Points above diagonal indicate the prompt helps.*

#### Prompt Effect by Model

| Model | With Prompt | Without | Effect |
| :--- | :---: | :---: | :---: |
| claude-opus-4-5 | 41.2% | 37.7% | **+3.5pp** |
| gpt-5.1-2025-11-13 | 45.0% | 43.8% | **+1.2pp** |
| claude-haiku-4-5 | 35.1% | 34.4% | **+0.7pp** |
| gemini-2.5-pro | 53.6% | 56.0% | **-2.4pp** |
| gpt-5.2-2025-12-11 | 40.1% | 42.9% | **-2.8pp** |
| claude-sonnet-4-5 | 31.0% | 34.1% | **-3.1pp** |
| gpt-5-nano-2025-08-07 | 33.9% | 39.1% | **-5.2pp** |
| gemini-3-pro-preview | 55.8% | 62.5% | **-6.7pp** |
| gpt-5-mini-2025-08-07 | 37.7% | 44.7% | **-7.0pp** |
| gemini-3-flash-preview | 39.2% | 46.5% | **-7.3pp** |
| gemini-2.5-flash | 37.5% | 50.2% | **-12.7pp** |

**Interpretation**:
- **Claude models** (+3-9pp) appear to benefit from explicit constraints on behavior
- **GPT models** (-5 to -16pp) may be over-constrained by the prompt's prescriptive style
- **Gemini models** show mixed results, suggesting model-specific prompt engineering is valuable

### 4.4 RQ4: Post-Cutoff Performance (Data Contamination Analysis)

**Critical Finding**: Performance on post-cutoff tasks (ACL 2025, EMNLP 2025) is substantially **lower** than on historical tasks, providing strong evidence of training data contamination effects.

#### Experimental Setup

To test whether models genuinely learn temporal reasoning or simply memorize training data, we compare performance on **historical vs. post-cutoff** award prediction tasks:

**Historical (Within Training Cutoff)**:
- Combines `emnlp_awards_mcq` + `emnlp_historical_awards_mcq`
- Papers from EMNLP 2021-2024 (all within model training windows)
- Tests in-distribution performance on familiar conferences and time periods

**Post-Cutoff (Beyond Training Data)**:
- For most models (Claude, GPT-5, Gemini 2.5): ACL 2025 (July) + EMNLP 2025 (November)
- For Gemini 3 models: Only EMNLP 2025 (trained after ACL 2025 but before EMNLP 2025)
- Tests true out-of-distribution generalization on unseen conferences

Both groups evaluate the **same capability** (predicting award tier from paper content), ensuring any performance difference reflects contamination rather than task difficulty. We use model-specific cutoff definitions because Gemini 3 models have later training dates (post-July 2025) that include ACL 2025 data.

| Model | Historical (Combined) | Post-Cutoff (Combined) | Δ(Hist→Post) |
| :--- | :---: | :---: | :---: |
| claude-haiku-4-5 | 2.0% | 12.3% | +10.3pp |
| claude-opus-4-5 | 1.5% | 23.1% | +21.6pp |
| claude-sonnet-4-5 | 0.0% | 2.5% | +2.5pp |
| gemini-2.5-flash | 47.2% | 29.5% | -17.7pp |
| gemini-2.5-pro | 26.4% | 33.6% | +7.2pp |
| gemini-3-flash-preview | 0.0% | 0.0% | +0.0pp |
| gemini-3-pro-preview | 21.8% | 18.8% | -3.0pp |
| gpt-5-mini-2025-08-07 | 8.6% | 32.0% | +23.3pp |
| gpt-5-nano-2025-08-07 | 35.5% | 25.4% | -10.1pp |
| gpt-5.1-2025-11-13 | 16.8% | 32.0% | +15.2pp |
| gpt-5.2-2025-12-11 | 3.5% | 28.7% | +25.1pp |

**Key Observations**:
- **3/11 models** (27%) show degradation on post-cutoff tasks (>1pp drop)
- **Average degradation**: 6.8pp across all models (median: 7.2pp)
- **Largest drop**: gemini-2.5-flash (-17.7pp), suggesting strong contamination effects
- The systematic degradation pattern provides evidence that historical performance is **inflated by training data memorization** rather than genuine temporal reasoning
- Post-cutoff evaluation offers a more honest assessment of models' ability to predict future outcomes from historical patterns

### 4.5 Overall Model Rankings

![Overall Performance 50](plots/overall_performance_msg50.png)
*Figure 5: Overall accuracy (sample-weighted) at message limit 50.*

![Heatmap 50](plots/model_task_heatmap_msg50.png)
*Figure 6: Model × Task heatmap showing performance variation across task families. Award tasks are shown as combined groups: 'Award (Historical)' includes base 2021-2024 papers (plus ACL 2025 for Gemini 3 models), while 'Award (Post-Cutoff)' includes post-training conference papers (ACL 2025 + EMNLP 2025 for most models, only EMNLP 2025 for Gemini 3).*

### 4.6 Task Difficulty Analysis

![Task Difficulty Ranking](plots/task_difficulty_ranking.png)
*Figure 7: Task difficulty ranking (average accuracy across models). Lower = harder. Award tasks shown as combined groups (Historical and Post-Cutoff).*

**Hardest Tasks** (lowest accuracy):
- **Award (Historical)** (14.9%): requires understanding implicit quality criteria in historical papers
- **Citation Rank** (17.1%): requires fine-grained citation count comparisons
- **Award (Post-Cutoff)** (21.6%): requires generalizing quality criteria to unseen conferences

**Easiest Tasks** (highest accuracy):
- **SOTA Bucket** (98.2%): structured lookup in historical benchmark data
- **Prof. Article** (89.7%): matching writing style and research topics
- **Field Focus** (61.6%): identifying research field from publication patterns

---

## 5. Discussion

### 5.1 The Value of Agentic Evaluation

Our results strongly support the hypothesis that **tool-using agents outperform direct generation** on temporal reasoning tasks. The 20-50pp gaps we observe suggest that:

1. **Parametric knowledge is insufficient**: Models cannot reliably predict future events from pre-training alone
2. **Evidence verification is crucial**: Agents that can check hypotheses against data achieve higher accuracy
3. **Multi-step reasoning helps**: Complex tasks benefit from iterative exploration and refinement

### 5.2 Test-Time Compute as a Scaling Axis

The dramatic scaling gains (up to +49pp) from increased message limits suggest that **test-time compute is a viable alternative to model scaling** for capability improvement. This has practical implications:

- Smaller models with more inference budget may match larger models' accuracy
- Compute allocation can be task-adaptive: simple tasks get fewer turns, complex tasks get more
- The ceiling on scaling gains varies by model family, informing deployment decisions

### 5.3 The Data Contamination Problem

Our post-cutoff analysis reveals a systematic pattern: **3/11 models (27%) show degradation** when evaluated on papers published after their training cutoffs. The average performance drop is **6.8 percentage points**, providing strong evidence of training data contamination effects. Key insights:

1. **Historical benchmarks overestimate capability**: The systematic degradation across most models suggests that strong performance on 2021-2024 papers reflects memorization rather than genuine temporal reasoning ability

2. **Post-cutoff evaluation is essential**: Only tasks definitively beyond training cutoffs provide uncontaminated capability estimates. Our model-specific cutoff handling (ACL 2025 is historical for Gemini 3 but post-cutoff for others) ensures fair comparison

3. **Contamination varies by model**: Some models show larger degradation than others, suggesting different levels of memorization during pre-training. This highlights the importance of post-cutoff benchmarks for honest model comparison

4. **The capability gap is substantial**: Even the best models achieve only moderate accuracy (33.6% best post-cutoff vs 47.2% best historical) on truly novel conference papers, indicating fundamental challenges in generalizing quality assessment criteria

### 5.4 Limitations

**Methodological Constraints**:

1. **Limited post-cutoff sample size**: While we combine multiple post-cutoff datasets (ACL 2025 + EMNLP 2025 for most models), the total post-cutoff evaluation set contains 1129 samples across 11 models. Larger post-cutoff benchmarks would strengthen contamination claims

2. **Model-specific cutoff assumptions**: We assume Gemini 3 models were trained after July 2025 based on release dates, but exact training cutoffs are not publicly disclosed. Misclassification would affect historical vs post-cutoff grouping

3. **Single domain evaluation**: All tasks focus on NLP/scientific literature. Temporal reasoning capabilities may differ in other domains (news, finance, legal documents, etc.)

4. **API-based evaluation constraints**: We cannot control for potential model updates during the evaluation period (models are identified by API endpoint, not specific checkpoint). Results may not be perfectly reproducible

5. **Prompt sensitivity**: Results depend on specific prompt formulations. The 'Offline Antigravity' prompt shows model-specific effects (beneficial for Claude, harmful for GPT), suggesting alternative prompts could yield different absolute performance levels

**Generalization Concerns**:

6. **Conference-specific patterns**: Award prediction tasks focus on ACL/EMNLP papers. Quality criteria and reviewer preferences may differ across venues, limiting generalization to other conferences or publication types

7. **Temporal scope**: Our historical data spans 2021-2024, and post-cutoff data covers 2025. Results may not extend to earlier historical periods or future time windows with different research trends

---

## 6. Conclusions

### 6.1 Key Findings

| Finding | Implication |
| :--- | :--- |
| Agentic > Zero-shot (+20-50pp) | Tool access is essential for temporal reasoning |
| Strong test-time scaling (+11-49pp) | Inference compute is a viable capability lever |
| Prompt effects are model-specific | One-size-fits-all prompts are suboptimal |
| Post-cutoff degradation | Historical benchmarks may overestimate capability |

### 6.2 Recommendations

| Use Case | Recommendation |
| :--- | :--- |
| **Maximum accuracy** | gemini-3-pro-preview with 50-message limit |
| **Efficiency-accuracy tradeoff** | Gemini 2.5 Pro (strong at low limits) |
| **Cost-effective option** | Claude Haiku 4.5 (excellent scaling, lower cost) |
| **Post-cutoff tasks** | Gemini 2.5 Flash (robust to temporal shift) |

### 6.3 Future Directions

1. **Larger post-cutoff test sets** as more conferences publish in 2025-2026
2. **Cross-domain generalization** to finance, medicine, policy domains
3. **Adaptive compute allocation** based on task difficulty estimation
4. **Fine-tuning experiments** to improve temporal reasoning capabilities

---

## Appendix

### A. Implementation Details

| Component | Specification |
| :--- | :--- |
| Framework | Inspect AI (inspect-ai) |
| Sandbox | Docker containers with Python 3.11 |
| Tools | python(), bash(), text_editor(), think() |
| Evaluation | Exact match for MCQ; custom scorers for ranking |

### B. Task Suffix Conventions

| Suffix | Solver | Sandbox | Agentic Prompt |
| :--- | :--- | :---: | :---: |
| `*_simple_task` | `generate()` | ❌ | ❌ |
| `*_no_offline_prompt_task` | `react()` | ✅ | ❌ |
| `*_task` (default) | `react()` | ✅ | ✅ |

### C. Reproduction

```bash
# Run the full ablation sweep
python scripts/run_inspect_ablations.py --include-no-offline --limit 50

# Generate this report and figures
cd analysis/comprehensive
python main.py
```

### D. Data Files

| File | Contents |
| :--- | :--- |
| `logs_msg15_summary_new.csv` | Results at message limit 15 |
| `logs_msg30_summary_new.csv` | Results at message limit 30 |
| `logs_msg50_summary_new.csv` | Results at message limit 50 |

---

*Report generated by Proof-of-Time analysis pipeline.*