# Inspect Ablations Plan

This folder holds logs for ablation runs comparing the offline Antigravity preamble vs. no-preamble across models and tasks.

## Benchmarks and Variants
- Award React: `emnlp_awards_mcq`, `emnlp_awards_mcq_no_offline`, `emnlp_historical_mcq`, `emnlp_historical_mcq_no_offline`, `emnlp_awards_mcq_simple`.
- Citation React: `citation_multiple_choice`, `citation_multiple_choice_no_offline`, `citation_ranking`, `citation_ranking_no_offline`, `citation_bucket_prediction`, `citation_bucket_prediction_no_offline`, `citation_all_tasks`, `citation_all_tasks_no_offline`.
- Future-Work React: `faculty_professor_field`, `faculty_professor_field_no_offline`, `faculty_professor_article`, `faculty_professor_article_no_offline`, `faculty_field_focus`, `faculty_field_focus_no_offline`, `faculty_all_tasks`, `faculty_all_tasks_no_offline`, `faculty_professor_field_simple`.
- SOTA Forecast: `sota_bucket_task`, `sota_bucket_task_no_offline`, `sota_bucket_simple_task`.

Each of the above (except simple baselines) also has `_local` variants that run without Docker; no_offline variants also have `_local` forms (e.g., `emnlp_awards_mcq_no_offline_local`).

### Task-to-Meaning Map (Data Access)
- Award React: `emnlp_awards_mcq` / `emnlp_awards_mcq_no_offline` classify EMNLP papers; `emnlp_historical_mcq` variants use historical samples; `emnlp_awards_mcq_simple` is the baseline.  
  Data: CSV/JSON in `inspect/award_react/sandbox/`. `_local` variants run without Docker; others use Docker.
- Citation React: `citation_multiple_choice` / `_no_offline` pick top-cited; `citation_ranking` / `_no_offline` order papers; `citation_bucket_prediction` / `_no_offline` bucket; `citation_all_tasks` / `_no_offline` combined.  
  Data: JSONL in `inspect/citation_react/sandbox/`. `_local` variants run without Docker; others use Docker.
- Future-Work React: `faculty_professor_field` / `_no_offline` choose a professor’s field; `faculty_professor_article` / `_no_offline` choose which paper (or None); `faculty_field_focus` / `_no_offline` assign field; `faculty_all_tasks` / `_no_offline` combined; `faculty_professor_field_simple` is the baseline.  
  Data: CSV/JSONL in `inspect/future_work_react/sandbox/`. `_local` variants run without Docker; others use Docker.
- SOTA Forecast: `sota_bucket_task` / `_no_offline` bucket benchmark scores; `sota_bucket_simple_task` is the baseline.  
  Data: JSON in `inspect/sota_forecast/sandbox/`. `_local` variants run without Docker; others use Docker.

## Models (default sweep)
- groq: `llama-4-maverick-17b-128e-instruct`, `llama-3.3-70b-versatile`, `kimi-k2-instruct-0905`, `openai/gpt-oss-120b`, `qwen3-32b`
- openai: `gpt-5.1-2025-11-13`, `gpt-5-mini-2025-08-07`, `gpt-5-nano-2025-08-07`
- google: `gemini-3-pro-preview`, `gemini-2.5-pro`
- anthropic/vertex: `claude-haiku-4-5@20251001`, `claude-opus-4-5@20251101`, `claude-sonnet-4-5@20250929`

## Runner Script
Use `scripts/run_inspect_ablations.py` to sweep models × tasks. Logs land in `logs/ablations/<model-slug>/<task>.log`.

Common flags:
- `--include-no-offline` to run the no-preamble variants.
- `--limit` to override per-task defaults (set `--limit 0` to omit the flag).
- `--models` / `--tasks` to filter.
- `--log-dir` to change the output location.
- `--dry-run` to print commands only.

Examples:
```bash
# Quick smoke: all models, with/without offline prompt, limit 5
python scripts/run_inspect_ablations.py --include-no-offline --limit 5

# Focused run on a couple models and tasks
python scripts/run_inspect_ablations.py \
  --models groq/llama-3.3-70b-versatile openai/gpt-5-nano-2025-08-07 \
  --tasks citation_all_tasks citation_all_tasks_no_offline \
  --include-no-offline --limit 10
```

## Prereqs
- `pip install inspect-ai` and provider creds set (e.g., `GROQ_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`/Vertex auth, `ANTHROPIC_API_KEY` or Vertex service account).
- Docker available for sandboxed tasks.

## Log Layout
- `logs/ablations/<model-slug>/<task>.log` captures stdout/stderr from each `inspect eval` run.
