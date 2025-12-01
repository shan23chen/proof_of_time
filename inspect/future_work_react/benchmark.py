"""Inspect AI benchmark for faculty future-work prediction tasks.

This benchmark evaluates LLMs' ability to predict 2025 research focus areas
for AI faculty and fields. The agent operates inside a sandbox with:
- Historical faculty publications (2018-2025) with metadata
- Per-professor CSV exports with full publication data
- Aggregated JSONL file with field classifications

Tasks:
1. Professor field prediction: Given a professor, predict their 2025 research field
2. Professor article attribution: Identify which 2025 paper belongs to a professor
3. Field focus prediction: Given papers, identify the research field they represent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import answer, match
from inspect_ai.solver import generate, system_message
from inspect_ai.tool import bash, bash_session, python, text_editor, think

from inspect.common.prompt_utils import get_offline_preamble

SANDBOX_ROOT = Path(__file__).resolve().parent / "sandbox"
PROFESSOR_FIELD_DATASET = Path(__file__).resolve().parent / "professor_field_mcq.jsonl"
PROFESSOR_ARTICLE_DATASET = Path(__file__).resolve().parent / "professor_article_mcq.jsonl"
FIELD_FOCUS_DATASET = Path(__file__).resolve().parent / "field_focus_mcq.jsonl"


def _load_samples(path: Path) -> Iterable[Sample]:
    """Parse a local JSONL dataset into Inspect `Sample` objects."""
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
                input=input_text or payload.get("context"),
                question=question,
                choices=payload.get("choices"),
                target=payload["answer"],
                metadata=payload.get("metadata", {}),
                files={"": sandbox_mount},
            )


def _load_dataset(path: Path) -> Dataset:
    samples: List[Sample] = list(_load_samples(path))
    return MemoryDataset(samples)


def build_agent(use_offline_prompt: bool = True) -> react:
    """Configure a React agent for faculty research prediction in sandbox."""
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="faculty-research-prediction",
        prompt=(
            f"{offline_prefix}"
            "You are a research trend analyst working inside a sandbox with "
            "faculty publication data. Your goal is to predict 2025 research focus "
            "areas for AI professors and fields based on their actual publications.\n\n"
            "Available data in sandbox:\n"
            "- data/faculty_publications.jsonl: Aggregated publications across all "
            "  professors with titles, abstracts, venues, years, and inferred field labels\n"
            "- data/faculty_publications/*.csv: Individual CSV files per professor "
            "  with complete publication metadata\n\n"
            "Your task: Analyze 2025 publications to predict research focus areas.\n\n"
            "Strategy:\n"
            "1. Load and explore the faculty publication database\n"
            "2. Filter for 2025 publications to identify current trends\n"
            "3. For professor queries: Search for the specific professor's 2025 papers\n"
            "4. For field queries: Identify common themes and keywords across papers\n"
            "5. Make evidence-based predictions using the sandbox data\n\n"
            "Important:\n"
            "- Base predictions ONLY on sandbox data\n"
            "- For multiple choice, respond with the exact letter or label requested\n"
            "- Do not add explanations unless explicitly asked\n"
            "- Cross-check your findings before finalizing the answer"
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
def faculty_professor_field_task() -> Task:
    """Professor-level field MCQ: Given a professor's 2025 papers, predict their research field."""
    dataset = _load_dataset(PROFESSOR_FIELD_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Expects exact field name match
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_professor_field"},
    )


@task()
def faculty_professor_field_task_local() -> Task:
    """Professor field MCQ without Docker sandbox (direct file access)."""
    dataset = _load_dataset(PROFESSOR_FIELD_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_professor_field_local"},
    )


@task()
def faculty_professor_field_task_no_offline_prompt() -> Task:
    """Professor field MCQ without the shared offline Antigravity preamble."""
    dataset = _load_dataset(PROFESSOR_FIELD_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_professor_field_no_offline"},
    )


@task()
def faculty_professor_field_task_no_offline_prompt_local() -> Task:
    """Professor field MCQ (no preamble) without Docker sandbox."""
    dataset = _load_dataset(PROFESSOR_FIELD_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_professor_field_no_offline_local"},
    )


@task()
def faculty_professor_article_task() -> Task:
    """Professor-level article attribution: Identify which 2025 paper belongs to a professor (or None)."""
    dataset = _load_dataset(PROFESSOR_ARTICLE_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("word"),  # Expects A, B, C, D, or None
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_professor_article"},
    )


@task()
def faculty_professor_article_task_local() -> Task:
    """Professor article attribution without Docker sandbox (direct file access)."""
    dataset = _load_dataset(PROFESSOR_ARTICLE_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("word"),  # Expects A, B, C, D, or None
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_professor_article_local"},
    )


@task()
def faculty_professor_article_task_no_offline_prompt() -> Task:
    """Professor article attribution without the shared offline Antigravity preamble."""
    dataset = _load_dataset(PROFESSOR_ARTICLE_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("word"),  # Expects A, B, C, D, or None
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_professor_article_no_offline"},
    )


@task()
def faculty_professor_article_task_no_offline_prompt_local() -> Task:
    """Professor article attribution (no preamble) without Docker sandbox."""
    dataset = _load_dataset(PROFESSOR_ARTICLE_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=answer("word"),  # Expects A, B, C, D, or None
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_professor_article_no_offline_local"},
    )


@task()
def faculty_field_focus_task() -> Task:
    """Field-level focus MCQ: Given 2025 papers, identify the research field they represent."""
    dataset = _load_dataset(FIELD_FOCUS_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Expects exact field name match
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_field_focus"},
    )


@task()
def faculty_field_focus_task_local() -> Task:
    """Field focus MCQ without Docker sandbox (direct file access)."""
    dataset = _load_dataset(FIELD_FOCUS_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Expects exact field name match
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_field_focus_local"},
    )


@task()
def faculty_field_focus_task_no_offline_prompt() -> Task:
    """Field focus MCQ without the shared offline Antigravity preamble."""
    dataset = _load_dataset(FIELD_FOCUS_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Expects exact field name match
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_field_focus_no_offline"},
    )


@task()
def faculty_field_focus_task_no_offline_prompt_local() -> Task:
    """Field focus MCQ (no preamble) without Docker sandbox."""
    dataset = _load_dataset(FIELD_FOCUS_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Expects exact field name match
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_field_focus_no_offline_local"},
    )


@task()
def faculty_all_tasks() -> Task:
    """Run all faculty prediction tasks together (professor field + article + field focus)."""
    # Combine all datasets
    professor_field_samples = list(_load_samples(PROFESSOR_FIELD_DATASET))
    professor_article_samples = list(_load_samples(PROFESSOR_ARTICLE_DATASET))
    field_focus_samples = list(_load_samples(FIELD_FOCUS_DATASET))
    
    all_samples = professor_field_samples + professor_article_samples + field_focus_samples
    dataset = MemoryDataset(all_samples)
    agent = build_agent()
    
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Generic match for all task types
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_all_tasks"},
    )


@task()
def faculty_all_tasks_local() -> Task:
    """Run all faculty prediction tasks without Docker sandbox (direct file access)."""
    professor_field_samples = list(_load_samples(PROFESSOR_FIELD_DATASET))
    professor_article_samples = list(_load_samples(PROFESSOR_ARTICLE_DATASET))
    field_focus_samples = list(_load_samples(FIELD_FOCUS_DATASET))

    all_samples = professor_field_samples + professor_article_samples + field_focus_samples
    dataset = MemoryDataset(all_samples)
    agent = build_agent()

    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Generic match for all task types
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_all_tasks_local"},
    )


@task()
def faculty_all_tasks_no_offline_prompt() -> Task:
    """Run all faculty prediction tasks without the shared offline Antigravity preamble."""
    professor_field_samples = list(_load_samples(PROFESSOR_FIELD_DATASET))
    professor_article_samples = list(_load_samples(PROFESSOR_ARTICLE_DATASET))
    field_focus_samples = list(_load_samples(FIELD_FOCUS_DATASET))

    all_samples = professor_field_samples + professor_article_samples + field_focus_samples
    dataset = MemoryDataset(all_samples)
    agent = build_agent(use_offline_prompt=False)

    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Generic match for all task types
        sandbox="docker",
        max_messages=40,
        metadata={"benchmark": "faculty_all_tasks_no_offline"},
    )


@task()
def faculty_all_tasks_no_offline_prompt_local() -> Task:
    """Run all faculty prediction tasks (no preamble) without Docker sandbox."""
    professor_field_samples = list(_load_samples(PROFESSOR_FIELD_DATASET))
    professor_article_samples = list(_load_samples(PROFESSOR_ARTICLE_DATASET))
    field_focus_samples = list(_load_samples(FIELD_FOCUS_DATASET))

    all_samples = professor_field_samples + professor_article_samples + field_focus_samples
    dataset = MemoryDataset(all_samples)
    agent = build_agent(use_offline_prompt=False)

    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),  # Generic match for all task types
        sandbox=None,
        max_messages=40,
        metadata={"benchmark": "faculty_all_tasks_no_offline_local"},
    )


@task()
def faculty_professor_field_simple_task() -> Task:
    """Fast baseline without tools for professor field predictions."""
    dataset = _load_dataset(PROFESSOR_FIELD_DATASET)
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are an expert on AI faculty. Based on the prompt, answer with exactly one field label."
            ),
            generate(),
        ],
        scorer=match(),
        metadata={"benchmark": "faculty_professor_field_simple"},
    )
