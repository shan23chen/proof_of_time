#!/usr/bin/env python3
"""
Run Inspect AI benchmarks across models with/without the offline Antigravity prompt.

Logs are written to logs/ablations/<model-slug>/<task>.log.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]


# Model list provided for the ablation sweep.
DEFAULT_MODELS: List[str] = [
    "groq/llama-4-maverick-17b-128e-instruct",
    "groq/llama-3.3-70b-versatile",
    "groq/kimi-k2-instruct-0905",
    "groq/openai/gpt-oss-120b",
    "groq/qwen3-32b",
    "openai/gpt-5.1-2025-11-13",
    "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-5-nano-2025-08-07",
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-pro",
    "anthropic/vertex/claude-haiku-4-5@20251001",
    "anthropic/vertex/claude-opus-4-5@20251101",
    "anthropic/vertex/claude-sonnet-4-5@20250929",
]


@dataclass
class BenchmarkTask:
    name: str
    target: str
    default_limit: Optional[int] = None


# Tasks and variants (with/without offline preamble where available).
TASKS: List[BenchmarkTask] = [
    BenchmarkTask("emnlp_awards_mcq", "inspect/award_react/benchmark.py::emnlp_awards_mcq_task", 5),
    BenchmarkTask("emnlp_awards_mcq_local", "inspect/award_react/benchmark.py::emnlp_awards_mcq_local_task", 5),
    BenchmarkTask("emnlp_awards_mcq_no_offline", "inspect/award_react/benchmark.py::emnlp_awards_mcq_no_offline_prompt_task", 5),
    BenchmarkTask("emnlp_awards_mcq_no_offline_local", "inspect/award_react/benchmark.py::emnlp_awards_mcq_no_offline_prompt_local_task", 5),
    BenchmarkTask("emnlp_historical_mcq", "inspect/award_react/benchmark.py::emnlp_historical_mcq_task", 5),
    BenchmarkTask("emnlp_historical_mcq_local", "inspect/award_react/benchmark.py::emnlp_historical_mcq_local_task", 5),
    BenchmarkTask("emnlp_historical_mcq_no_offline", "inspect/award_react/benchmark.py::emnlp_historical_mcq_no_offline_prompt_task", 5),
    BenchmarkTask("emnlp_historical_mcq_no_offline_local", "inspect/award_react/benchmark.py::emnlp_historical_mcq_no_offline_prompt_local_task", 5),
    BenchmarkTask("emnlp_awards_mcq_simple", "inspect/award_react/benchmark.py::emnlp_awards_mcq_simple_task", 5),
    BenchmarkTask("citation_multiple_choice", "inspect/citation_react/benchmark.py::citation_multiple_choice", 5),
    BenchmarkTask("citation_multiple_choice_local", "inspect/citation_react/benchmark.py::citation_multiple_choice_local", 5),
    BenchmarkTask("citation_multiple_choice_no_offline", "inspect/citation_react/benchmark.py::citation_multiple_choice_no_offline_prompt", 5),
    BenchmarkTask("citation_multiple_choice_no_offline_local", "inspect/citation_react/benchmark.py::citation_multiple_choice_no_offline_prompt_local", 5),
    BenchmarkTask("citation_ranking", "inspect/citation_react/benchmark.py::citation_ranking", 5),
    BenchmarkTask("citation_ranking_local", "inspect/citation_react/benchmark.py::citation_ranking_local", 5),
    BenchmarkTask("citation_ranking_no_offline", "inspect/citation_react/benchmark.py::citation_ranking_no_offline_prompt", 5),
    BenchmarkTask("citation_ranking_no_offline_local", "inspect/citation_react/benchmark.py::citation_ranking_no_offline_prompt_local", 5),
    BenchmarkTask("citation_bucket_prediction", "inspect/citation_react/benchmark.py::citation_bucket_prediction", 5),
    BenchmarkTask("citation_bucket_prediction_local", "inspect/citation_react/benchmark.py::citation_bucket_prediction_local", 5),
    BenchmarkTask("citation_bucket_prediction_no_offline", "inspect/citation_react/benchmark.py::citation_bucket_prediction_no_offline_prompt", 5),
    BenchmarkTask("citation_bucket_prediction_no_offline_local", "inspect/citation_react/benchmark.py::citation_bucket_prediction_no_offline_prompt_local", 5),
    BenchmarkTask("citation_all_tasks", "inspect/citation_react/benchmark.py::citation_all_tasks", 10),
    BenchmarkTask("citation_all_tasks_local", "inspect/citation_react/benchmark.py::citation_all_tasks_local", 10),
    BenchmarkTask("citation_all_tasks_no_offline", "inspect/citation_react/benchmark.py::citation_all_tasks_no_offline_prompt", 10),
    BenchmarkTask("citation_all_tasks_no_offline_local", "inspect/citation_react/benchmark.py::citation_all_tasks_no_offline_prompt_local", 10),
    BenchmarkTask("faculty_professor_field", "inspect/future_work_react/benchmark.py::faculty_professor_field_task", 10),
    BenchmarkTask("faculty_professor_field_local", "inspect/future_work_react/benchmark.py::faculty_professor_field_task_local", 10),
    BenchmarkTask("faculty_professor_field_no_offline", "inspect/future_work_react/benchmark.py::faculty_professor_field_task_no_offline_prompt", 10),
    BenchmarkTask("faculty_professor_field_no_offline_local", "inspect/future_work_react/benchmark.py::faculty_professor_field_task_no_offline_prompt_local", 10),
    BenchmarkTask("faculty_professor_article", "inspect/future_work_react/benchmark.py::faculty_professor_article_task", 10),
    BenchmarkTask("faculty_professor_article_local", "inspect/future_work_react/benchmark.py::faculty_professor_article_task_local", 10),
    BenchmarkTask("faculty_professor_article_no_offline", "inspect/future_work_react/benchmark.py::faculty_professor_article_task_no_offline_prompt", 10),
    BenchmarkTask("faculty_professor_article_no_offline_local", "inspect/future_work_react/benchmark.py::faculty_professor_article_task_no_offline_prompt_local", 10),
    BenchmarkTask("faculty_field_focus", "inspect/future_work_react/benchmark.py::faculty_field_focus_task", 10),
    BenchmarkTask("faculty_field_focus_local", "inspect/future_work_react/benchmark.py::faculty_field_focus_task_local", 10),
    BenchmarkTask("faculty_field_focus_no_offline", "inspect/future_work_react/benchmark.py::faculty_field_focus_task_no_offline_prompt", 10),
    BenchmarkTask("faculty_field_focus_no_offline_local", "inspect/future_work_react/benchmark.py::faculty_field_focus_task_no_offline_prompt_local", 10),
    BenchmarkTask("faculty_all_tasks", "inspect/future_work_react/benchmark.py::faculty_all_tasks", 20),
    BenchmarkTask("faculty_all_tasks_local", "inspect/future_work_react/benchmark.py::faculty_all_tasks_local", 20),
    BenchmarkTask("faculty_all_tasks_no_offline", "inspect/future_work_react/benchmark.py::faculty_all_tasks_no_offline_prompt", 20),
    BenchmarkTask("faculty_all_tasks_no_offline_local", "inspect/future_work_react/benchmark.py::faculty_all_tasks_no_offline_prompt_local", 20),
    BenchmarkTask("faculty_professor_field_simple", "inspect/future_work_react/benchmark.py::faculty_professor_field_simple_task", 10),
    BenchmarkTask("sota_bucket_task", "inspect/sota_forecast/benchmark.py::sota_bucket_task", 20),
    BenchmarkTask("sota_bucket_task_local", "inspect/sota_forecast/benchmark.py::sota_bucket_task_local", 20),
    BenchmarkTask("sota_bucket_task_no_offline", "inspect/sota_forecast/benchmark.py::sota_bucket_task_no_offline_prompt", 20),
    BenchmarkTask("sota_bucket_task_no_offline_local", "inspect/sota_forecast/benchmark.py::sota_bucket_task_no_offline_prompt_local", 20),
    BenchmarkTask("sota_bucket_simple_task", "inspect/sota_forecast/benchmark.py::sota_bucket_simple_task", 20),
]


def slugify(value: str) -> str:
    """Safe path component for model names."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def iter_tasks(include_no_offline: bool, selected: Optional[Iterable[str]]) -> List[BenchmarkTask]:
    selected_set = set(selected) if selected else None
    tasks: List[BenchmarkTask] = []
    for task in TASKS:
        if not include_no_offline and "no_offline" in task.name:
            continue
        if selected_set and task.name not in selected_set:
            continue
        tasks.append(task)
    return tasks


def run_task(model: str, task: BenchmarkTask, limit_override: Optional[int], log_dir: Path, inspect_bin: str, dry_run: bool) -> int:
    limit = limit_override if limit_override is not None else task.default_limit
    cmd: List[str] = [inspect_bin, "eval", task.target, "--model", model]
    if limit:
        cmd += ["--limit", str(limit)]

    model_dir = log_dir / slugify(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / f"{task.name}.log"

    print(f"[run] model={model} task={task.name} limit={limit} log={log_path}")
    if dry_run:
        print(" ".join(cmd))
        return 0

    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[fail] model={model} task={task.name} rc={proc.returncode} (see {log_path})", file=sys.stderr)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI ablations with/without offline prompt.")
    parser.add_argument("--models", nargs="+", help="Subset of models to run (default: preset list).")
    parser.add_argument("--tasks", nargs="+", help="Subset of task names to run (match entries in the script).")
    parser.add_argument("--log-dir", default=ROOT / "logs" / "ablations", type=Path, help="Directory for logs.")
    parser.add_argument("--limit", type=int, default=None, help="Override per-task limit; use 0 to disable limit flag.")
    parser.add_argument("--inspect-bin", default="inspect", help="Inspect executable (default: inspect).")
    parser.add_argument("--include-no-offline", action="store_true", help="Include no_offline prompt variants (default: excluded if flag absent).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models or DEFAULT_MODELS
    tasks = iter_tasks(include_no_offline=args.include_no_offline, selected=args.tasks)
    if not tasks:
        print("No tasks selected. Check --tasks or --include-no-offline.", file=sys.stderr)
        return 1

    failures = 0
    for model in models:
        for task in tasks:
            rc = run_task(
                model=model,
                task=task,
                limit_override=args.limit,
                log_dir=args.log_dir,
                inspect_bin=args.inspect_bin,
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures += 1

    if failures:
        print(f"Completed with {failures} failures (see logs).", file=sys.stderr)
        return 1
    print("Completed all runs successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
