# proof_of_time

project plan: https://docs.google.com/document/d/13FkKrJ3X19Sr3rgxFWRGWlSjj1Fm_eJZYZMO4hXrz7I/edit?usp=sharing

data: https://huggingface.co/AIM-Harvard

## EMNLP Topic Explorer Dashboard

The interactive EMNLP explorer lives in `analysis/emnlp_topics_dashboard.py`. It supports
filtering papers by year, track (Main vs. Findings), primary and fine-grained topics,
topic clusters, domain tags, benchmark tasks, keyword/author search, and CSV download of
the currently filtered set. Toggle the primary-topic trend metric (share vs. raw counts),
choose how many fine topics appear in the heatmap, explore domain distributions, and scan
auto-generated insight callouts plus top-author/benchmark tables for quick orientation.

Run it locally after installing the dependencies (`pip install datasets dash plotly pandas`):

```bash
python analysis/emnlp_topics_dashboard.py
```

By default the Dash app will be available at http://127.0.0.1:8050. Use the `--host` and
`--port` flags to customise the binding. Pass `--export` to generate refreshed CSV
summaries in `reports/` without launching the web UI.

The export mode now writes:

* `emnlp_topics_summary.csv` — primary-topic counts, shares, and totals per year.
* `emnlp_finetopic_summary.csv` — exploded fine-grained topic counts per year.
* `emnlp_benchmark_summary.csv` — benchmark vs. non-benchmark paper totals by track.
* `emnlp_benchmark_yearly.csv` — annual benchmark totals and shares across tracks.
* `emnlp_domain_summary.csv` — aggregated domain-tag activity across the corpus.
* `emnlp_author_summary.csv` — frequency table of author appearances (deduplicated).

## Inspect AI Benchmarks

This project includes Inspect AI benchmarks for paper classification using ReAct agents. See `inspect/emnlp_react/` for details.

### Faculty Future-Work Benchmark

`inspect/future_work_react/` mirrors the EMNLP prototype but uses per-professor publication CSVs to ask 2025 research forecasting questions (professor focus, article attribution, and field-level focus). Build the sandbox via `python dataset_building/generate_faculty_futurework.py` and run tasks from `inspect/future_work_react/benchmark.py`.

### SOTA Forecast Benchmark

`inspect/sota_forecast/` provides a sandboxed table of October 2025 frontier-model metrics. Agents must look up benchmarks (e.g., MMLU, IFEval, Livebench) and place the SOTA score into coarse performance buckets (a=0-20 … e=80-100). Regenerate data with `python dataset_building/generate_sota_forecast.py`.

### Running with FreeInference

Use Inspect AI's OpenAI provider with a custom base URL:

```bash
export OPENAI_API_KEY="your-freeinference-key"

inspect eval inspect/emnlp_react/benchmark.py@emnlp_awards_mcq_task \
    --model openai/llama-3.3-70b-instruct \
    --model-base-url https://api.freeinference.org/v1 \
    --limit 5
```

See [USING_FREEINFERENCE.md](USING_FREEINFERENCE.md) for more options.
