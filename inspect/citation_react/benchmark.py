"""Inspect AI benchmark for citation prediction tasks.

This benchmark evaluates LLMs' ability to predict citation counts for awarded
NLP papers. The agent operates inside a sandbox with:
- Historical EMNLP papers (2021-2024) with citation metadata
- Best paper data with actual citation counts

Tasks:
1. Multiple choice: Which of 4 papers has highest citations?
2. Ranking: Rank 4 papers by citation count
3. Bucket prediction: Predict citation range (5 buckets)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import answer
from inspect_ai.tool import bash, bash_session, python, text_editor, think

from inspect.common.prompt_utils import get_offline_preamble

SANDBOX_ROOT = Path(__file__).resolve().parent / "sandbox"
CITATION_TASKS_PATH = Path(__file__).resolve().parent.parent.parent / "citation_eval" / "tasks" / "sandbox_agent_tasks.json"


def _load_citation_samples(task_type: str = None) -> Iterable[Sample]:
    """Load citation prediction tasks from JSON file."""
    # Get sandbox mount path
    sandbox_mount = str(SANDBOX_ROOT.resolve())

    # Load tasks
    with CITATION_TASKS_PATH.open(encoding="utf-8") as f:
        all_tasks = json.load(f)

    # Filter by task type if specified
    if task_type:
        tasks = [t for t in all_tasks if t["task_type"] == task_type]
    else:
        tasks = all_tasks

    # Convert to Inspect samples
    for task_data in tasks:
        yield Sample(
            id=task_data["task_id"],
            input=task_data["prompt"],
            target=task_data["ground_truth"]["correct_answer"],
            metadata={
                "task_type": task_data["task_type"],
                "sampling_strategy": task_data["sampling_strategy"],
                "ground_truth": task_data["ground_truth"],
            },
            files={"": sandbox_mount},  # Mount sandbox directory with data files
        )


def build_citation_dataset(task_type: str = None) -> Dataset:
    """Create an Inspect dataset from citation tasks."""
    samples: List[Sample] = list(_load_citation_samples(task_type))
    return MemoryDataset(samples)


def build_citation_agent(use_offline_prompt: bool = True):
    """Configure a React agent for citation prediction in sandbox."""
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="citation-prediction-agent",
        prompt=(
            f"{offline_prefix}"
            "You are a research impact analyst working inside a sandbox with "
            "historical citation data. Your goal is to predict citation counts "
            "for papers from 2025 using historical patterns from 2021-2024.\n\n"
            "Available data in sandbox:\n"
            "- historical_papers_2021_2024.jsonl: 38,330 papers (2021-2024) with "
            "  citation counts, authors, titles, years, venues\n"
            "Your task: Predict citations for 2025 papers based on historical "
            "patterns.\n\n"
            "Strategy:\n"
            "1. Load and explore the historical database (2021-2024 papers)\n"
            "2. Identify patterns: Which topics get high citations?\n"
            "3. For each target paper, search for similar papers in history\n"
            "4. Make evidence-based predictions using statistical patterns\n\n"
            "Important: Base predictions ONLY on sandbox data. Do not use "
            "memorized information about specific papers or authors."
        ),
        tools=[
            think(),
            python(),
            bash(),
            bash_session(),
            text_editor(),
        ],
    )


@task()
def citation_multiple_choice() -> Task:
    """Multiple choice: Which of 4 papers has the highest citations?"""
    dataset = build_citation_dataset(task_type="multiple_choice")
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),  # Extract single letter after "ANSWER:"
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_mc"},
    )


@task()
def citation_multiple_choice_local() -> Task:
    """Multiple choice without Docker sandbox (direct file access)."""
    dataset = build_citation_dataset(task_type="multiple_choice")
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_mc_local"},
    )


@task()
def citation_multiple_choice_no_offline_prompt() -> Task:
    """Multiple choice without the shared offline Antigravity preamble."""
    dataset = build_citation_dataset(task_type="multiple_choice")
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_mc_no_offline"},
    )


@task()
def citation_multiple_choice_no_offline_prompt_local() -> Task:
    """Multiple choice (no preamble) without Docker sandbox."""
    dataset = build_citation_dataset(task_type="multiple_choice")
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_mc_no_offline_local"},
    )


@task()
def citation_ranking() -> Task:
    """Ranking: Rank 4 papers from most to least cited."""
    dataset = build_citation_dataset(task_type="ranking")
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),  # Extract full line after "ANSWER:" for ranking
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_ranking"},
    )


@task()
def citation_ranking_local() -> Task:
    """Ranking task without Docker sandbox (direct file access)."""
    dataset = build_citation_dataset(task_type="ranking")
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_ranking_local"},
    )


@task()
def citation_ranking_no_offline_prompt() -> Task:
    """Ranking task without the shared offline Antigravity preamble."""
    dataset = build_citation_dataset(task_type="ranking")
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_ranking_no_offline"},
    )


@task()
def citation_ranking_no_offline_prompt_local() -> Task:
    """Ranking task (no preamble) without Docker sandbox."""
    dataset = build_citation_dataset(task_type="ranking")
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_ranking_no_offline_local"},
    )


@task()
def citation_bucket_prediction() -> Task:
    """Bucket prediction: Predict citation range (5 buckets: 0-10, 10-25, 25-60, 60-150, 150+)."""
    dataset = build_citation_dataset(task_type="bucket_prediction")
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),  # Extract single letter after "ANSWER:"
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_bucket"},
    )


@task()
def citation_bucket_prediction_local() -> Task:
    """Bucket prediction without Docker sandbox (direct file access)."""
    dataset = build_citation_dataset(task_type="bucket_prediction")
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_bucket_local"},
    )


@task()
def citation_bucket_prediction_no_offline_prompt() -> Task:
    """Bucket prediction without the shared offline Antigravity preamble."""
    dataset = build_citation_dataset(task_type="bucket_prediction")
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_bucket_no_offline"},
    )


@task()
def citation_bucket_prediction_no_offline_prompt_local() -> Task:
    """Bucket prediction (no preamble) without Docker sandbox."""
    dataset = build_citation_dataset(task_type="bucket_prediction")
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("letter"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_bucket_no_offline_local"},
    )


@task()
def citation_all_tasks() -> Task:
    """Run all citation prediction tasks together."""
    dataset = build_citation_dataset()  # All tasks
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),  # Extract full line after "ANSWER:" (works for both letter and ranking)
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_all"},
    )


@task()
def citation_all_tasks_local() -> Task:
    """Run all citation prediction tasks without Docker sandbox."""
    dataset = build_citation_dataset()
    agent = build_citation_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_all_local"},
    )


@task()
def citation_all_tasks_no_offline_prompt() -> Task:
    """Run all citation prediction tasks without the shared offline Antigravity preamble."""
    dataset = build_citation_dataset()
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),
        sandbox="docker",
        metadata={"benchmark": "citation_prediction_all_no_offline"},
    )


@task()
def citation_all_tasks_no_offline_prompt_local() -> Task:
    """Run all citation prediction tasks (no preamble) without Docker sandbox."""
    dataset = build_citation_dataset()
    agent = build_citation_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("line"),
        sandbox=None,
        metadata={"benchmark": "citation_prediction_all_no_offline_local"},
    )
