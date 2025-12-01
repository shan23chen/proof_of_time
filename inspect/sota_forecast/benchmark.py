"""Inspect AI benchmark prototype for SOTA forecast bucket questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message
from inspect_ai.tool import bash, bash_session, python, text_editor, think

from inspect.common.prompt_utils import get_offline_preamble

SANDBOX_ROOT = Path(__file__).resolve().parent / "sandbox"
DATASET_PATH = Path(__file__).resolve().parent / "mcq_dataset.jsonl"


def _load_samples(path: Path) -> Iterable[Sample]:
    sandbox_mount = str(SANDBOX_ROOT.resolve())
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            question = payload.get("question") or ""
            prompt = payload.get("prompt")
            input_text = question
            if prompt:
                input_text = f"{question}\n\n{prompt}" if question else prompt
            yield Sample(
                id=idx,
                input=input_text,
                question=question,
                choices=payload.get("choices"),
                target=payload["answer"],
                metadata=payload.get("metadata", {}),
                files={"": sandbox_mount},
            )


def _load_dataset() -> Dataset:
    samples: List[Sample] = list(_load_samples(DATASET_PATH))
    return MemoryDataset(samples)


def build_agent(use_offline_prompt: bool = True) -> react:
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="sota-forecast-react",
        prompt=(
            f"{offline_prefix}"
            "You are analyzing frontier-model benchmark performance. All answers are bucket letters "
            "from the set {a,b,c,d,e} following these ranges: "
            "a=0-20, b=20-40, c=40-60, d=60-80, e=80-100.\n\n"
            "Instructions:\n"
            "1. Use python()/bash() to read sandbox/data/sota_metrics.json.\n"
            "2. Locate the benchmark + metric mentioned in the prompt.\n"
            "3. Output only the bucket letter (lowercase) with no explanation.\n"
        ),
        tools=[think(), python(), bash(), bash_session(), text_editor()],
    )


@task()
def sota_bucket_task() -> Task:
    dataset = _load_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        max_messages=30,
        metadata={"benchmark": "sota_forecast_mcq"},
    )


@task()
def sota_bucket_task_local() -> Task:
    """SOTA bucket task without Docker sandbox (direct file access)."""
    dataset = _load_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=30,
        metadata={"benchmark": "sota_forecast_mcq_local"},
    )


@task()
def sota_bucket_task_no_offline_prompt() -> Task:
    dataset = _load_dataset()
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        max_messages=30,
        metadata={"benchmark": "sota_forecast_mcq_no_offline"},
    )


@task()
def sota_bucket_task_no_offline_prompt_local() -> Task:
    """SOTA bucket task (no preamble) without Docker sandbox."""
    dataset = _load_dataset()
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=30,
        metadata={"benchmark": "sota_forecast_mcq_no_offline_local"},
    )


@task()
def sota_bucket_simple_task() -> Task:
    dataset = _load_dataset()
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "Answer each question with the correct bucket letter (a/b/c/d/e) using the provided prompt only."
            ),
            generate(),
        ],
        scorer=match(),
        metadata={"benchmark": "sota_forecast_mcq_simple"},
    )
