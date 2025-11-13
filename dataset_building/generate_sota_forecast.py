#!/usr/bin/env python3

"""Build sandbox artefacts and MCQ dataset for the SOTA forecast benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

BUCKETS = [
    ("a", 0, 20),
    ("b", 20, 40),
    ("c", 40, 60),
    ("d", 60, 80),
    ("e", 80, 100),
]


@dataclass
class MetricRecord:
    benchmark: str
    metric: str
    score: float
    best_model: str
    dataset_type: str
    notes: str


def load_static_metrics() -> List[MetricRecord]:
    """Hand-curated metrics pulled from the Oct 2025 SOTA tables."""
    data: List[MetricRecord] = []

    def add(benchmark: str, metric: str, score: float, model: str, tier: str, notes: str) -> None:
        data.append(MetricRecord(benchmark, metric, score, model, tier, notes))

    # Instruction – Coding Tasks
    add("LiveCodeBench v6", "Pass@1", 53.7, "Kimi K2 Instruct", "Instruction", "Coding tasks")
    add("OJBench", "Pass@1", 27.1, "Kimi K2 Instruct", "Instruction", "Coding tasks")
    add("MultiPL-E", "Pass@1", 89.6, "Claude Opus 4 (w/o extended thinking)", "Instruction", "Coding tasks")
    add("SWE-bench Verified (Agentless)", "Single Patch Accuracy", 53.0, "Claude Opus 4 (w/o extended thinking)", "Instruction", "Coding tasks")
    add("SWE-bench Verified (Agentic)", "Single Attempt Accuracy", 72.7, "Claude Sonnet 4 (w/o extended thinking)", "Instruction", "Coding tasks")
    add("SWE-bench Verified (Agentic)", "Multiple Attempts Accuracy", 80.2, "Claude Sonnet 4 (w/o extended thinking)", "Instruction", "Coding tasks")
    add("SWE-bench Multilingual (Agentic)", "Single Attempt Accuracy", 51.0, "Claude Sonnet 4 (w/o extended thinking)", "Instruction", "Coding tasks")
    add("TerminalBench (Inhouse Framework)", "Accuracy", 43.2, "Claude Opus 4 (w/o extended thinking)", "Instruction", "Coding tasks")
    add("TerminalBench (Terminus)", "Accuracy", 30.3, "GPT-4.1", "Instruction", "Coding tasks")
    add("Aider-Polyglot", "Accuracy", 70.7, "Claude Opus 4 (w/o extended thinking)", "Instruction", "Coding tasks")

    # Instruction – Tool Use
    add("Tau2 Retail", "Avg@4", 81.8, "Claude Opus 4 (w/o extended thinking)", "Instruction", "Tool use tasks")
    add("Tau2 Airline", "Avg@4", 60.0, "Claude Opus 4 (w/o extended thinking)", "Instruction", "Tool use tasks")
    add("Tau2 Telecom", "Avg@4", 65.8, "Kimi K2 Instruct", "Instruction", "Tool use tasks")
    add("AceBench", "Accuracy", 80.1, "GPT-4.1", "Instruction", "Tool use tasks")

    # Instruction – Math & STEM
    add("AIME 2024", "Avg@64", 69.6, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("AIME 2025", "Avg@64", 49.5, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("MATH-500", "Accuracy", 97.4, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("HMMT 2025", "Avg@32", 38.8, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("CNMO 2024", "Avg@16", 75.0, "Gemini 2.5 Flash Preview (05-20)", "Instruction", "Math & STEM tasks")
    add("PolyMath-en", "Avg@4", 65.1, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("ZebraLogic", "Accuracy", 89.0, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("AutoLogi", "Accuracy", 89.8, "Claude Sonnet 4 (w/o extended thinking)", "Instruction", "Math & STEM tasks")
    add("GPQA-Diamond", "Average@8", 75.1, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("SuperGPQA", "Exact Match Accuracy", 57.2, "Kimi K2 Instruct", "Instruction", "Math & STEM tasks")
    add("Humanity's Last Exam (Text Only)", "Score", 7.1, "Claude Opus 4 (w/o extended thinking)", "Instruction", "Math & STEM tasks")

    # Instruction – General Tasks
    add("MMLU", "Exact Match Accuracy", 92.9, "Claude Opus 4 (w/o extended thinking)", "Instruction", "General tasks")
    add("MMLU-Redux", "Exact Match Accuracy", 94.2, "Claude Opus 4 (w/o extended thinking)", "Instruction", "General tasks")
    add("MMLU-Pro", "Exact Match Accuracy", 86.6, "Claude Opus 4 (w/o extended thinking)", "Instruction", "General tasks")
    add("IFEval", "Prompt Strict Accuracy", 89.8, "Kimi K2 Instruct", "Instruction", "General tasks")
    add("Multi-Challenge", "Accuracy", 54.1, "Kimi K2 Instruct", "Instruction", "General tasks")
    add("SimpleQA", "Correct", 42.3, "GPT-4.1", "Instruction", "General tasks")
    add("Livebench", "Pass@1", 76.4, "Kimi K2 Instruct", "Instruction", "General tasks")

    # Base models – General / Coding / Math
    add("MMLU", "Exact Match Accuracy", 87.8, "Kimi K2 Base", "Base", "General tasks")
    add("MMLU-Pro", "Exact Match Accuracy", 69.2, "Kimi K2 Base", "Base", "General tasks")
    add("MMLU-Redux 2.0", "Exact Match Accuracy", 90.2, "Kimi K2 Base", "Base", "General tasks")
    add("SimpleQA", "Correct", 35.3, "Kimi K2 Base", "Base", "General tasks")
    add("TriviaQA", "Exact Match Accuracy", 85.1, "Kimi K2 Base", "Base", "General tasks")
    add("GPQA-Diamond", "Average@8", 50.5, "DeepSeek-V3-Base", "Base", "Math & STEM tasks")
    add("SuperGPQA", "Exact Match Accuracy", 44.7, "Kimi K2 Base", "Base", "Math & STEM tasks")
    add("LiveCodeBench v6", "Pass@1", 26.3, "Kimi K2 Base", "Base", "Coding tasks")
    add("EvalPlus", "Pass@1", 80.3, "Kimi K2 Base", "Base", "Coding tasks")
    add("MATH", "Exact Match Accuracy", 70.2, "Kimi K2 Base", "Base", "Math & STEM tasks")
    add("GSM8K", "Exact Match Accuracy", 92.1, "Kimi K2 Base", "Base", "Math & STEM tasks")
    add("C-Eval", "Exact Match Accuracy", 92.5, "Kimi K2 Base", "Base", "Chinese tasks")
    add("CSimpleQA", "Correct", 77.6, "Kimi K2 Base", "Base", "Chinese tasks")

    return data


def bucket_for_score(score: float) -> str:
    for letter, low, high in BUCKETS:
        if letter == "e":
            if low <= score <= high:
                return letter
        if low <= score < high:
            return letter
    return "e"


def bucket_choices() -> List[str]:
    return [f"{letter}: {low}-{high}" for letter, low, high in BUCKETS]


def write_jsonl(records, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(records: List[MetricRecord]):
    choices = bucket_choices()
    dataset_rows = []
    for rec in records:
        question = (
            f"As of Oct 2025, what bucket best captures the frontier SOTA {rec.benchmark} "
            f"{rec.metric}? Answer with a/b/c/d/e only."
        )
        context = (
            f"Benchmark: {rec.benchmark}\n"
            f"Metric: {rec.metric}\n"
            f"Model tier: {rec.dataset_type}\n"
            f"Top model: {rec.best_model}\n"
            f"Reported score: {rec.score}"
        )
        dataset_rows.append(
            {
                "question": question,
                "prompt": context + "\nOptions:\n" + "\n".join(choices),
                "choices": choices,
                "answer": bucket_for_score(rec.score),
                "metadata": {
                    "benchmark": rec.benchmark,
                    "metric": rec.metric,
                    "score": rec.score,
                    "best_model": rec.best_model,
                    "model_tier": rec.dataset_type,
                    "notes": rec.notes,
                },
            }
        )
    return dataset_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the SOTA forecast sandbox + dataset.")
    parser.add_argument(
        "--sandbox-dir",
        type=Path,
        default=Path("inspect/sota_forecast/sandbox/data"),
        help="Directory where sandbox data will be written.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("inspect/sota_forecast/mcq_dataset.jsonl"),
    )
    args = parser.parse_args()

    records = load_static_metrics()
    sandbox_dir = args.sandbox_dir
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    sandbox_json = sandbox_dir / "sota_metrics.json"
    with sandbox_json.open("w", encoding="utf-8") as handle:
        json.dump([asdict(rec) for rec in records], handle, ensure_ascii=False, indent=2)

    dataset_rows = build_dataset(records)
    write_jsonl(dataset_rows, args.dataset_path)

    print(f"[write] {len(records)} metrics -> {sandbox_json}")
    print(f"[dataset] {len(dataset_rows)} rows -> {args.dataset_path}")


if __name__ == "__main__":
    main()
