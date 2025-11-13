# SOTA Forecast Benchmark Prototype

This benchmark asks Inspect AI agents to reason about frontier-model performance metrics. Agents are confined to a sandbox containing a JSON snapshot of the October&nbsp;2025 SOTA table (instruction + base models). Each sample asks the agent to place a benchmark’s SOTA score into a coarse bucket (`a: 0-20`, …, `e: 80-100`) and respond with the letter only.

## Layout

- `benchmark.py` – defines the Inspect tasks and React agent configuration.
- `mcq_dataset.jsonl` – bucketed multiple-choice samples generated from the metrics snapshot.
- `sandbox/data/sota_metrics.json` – ground-truth table copied from the provided evaluation report.

## Building / Refreshing

```bash
python dataset_building/generate_sota_forecast.py \
  --sandbox-dir inspect/sota_forecast/sandbox/data \
  --dataset-path inspect/sota_forecast/mcq_dataset.jsonl
```

The generator script embeds the current SOTA table. Update the list in `load_static_metrics()` to add new benchmarks or refresh scores before re-running.

## Running the Benchmark

```bash
inspect eval inspect/sota_forecast/benchmark.py@sota_bucket_task \
  --model openai/gpt-4o-mini \
  --limit 5
```

The agent must inspect `sandbox/data/sota_metrics.json` using the provided tools (`python()`, `bash()`, etc.) before answering, then output only the bucket letter (`a`–`e`).
