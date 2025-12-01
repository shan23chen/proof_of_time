"""Inspect AI benchmark prototype for EMNLP benchmark queries.

The goal is to demonstrate a React-style agent that works entirely inside a
sandbox populated with the EMNLP accepted papers dataset. The agent is given a
small QA set and must rely on the provided tools (bash, bash session, python,
text editor, think) to surface the correct evidence from the sandbox before it
answers.
"""

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
MCQ_DATASET_PATH = Path(__file__).resolve().parent / "mcq_dataset.jsonl"
HISTORICAL_MCQ_PATH = Path(__file__).resolve().parent / "historical_mcq_dataset.jsonl"


def _load_samples(path: Path) -> Iterable[Sample]:
    """Parse a local JSONL dataset into Inspect `Sample` objects."""
    sandbox_mount = str(SANDBOX_ROOT.resolve())
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            payload = json.loads(line)
            question = payload.get("question", "")
            input_text = question
            context = payload.get("context")
            choices = payload.get("choices")
            if context:
                input_parts = [context.strip(), f"Question: {question}"]
                if choices:
                    options = ", ".join(choices)
                    input_parts.append(f"Options: {options}")
                input_text = "\n\n".join(part for part in input_parts if part)

            yield Sample(
                id=idx,
                input=input_text,
                target=payload["answer"],
                metadata=payload.get("metadata", {}),
                files={"": sandbox_mount},
            )


def build_mcq_dataset() -> Dataset:
    """Create an Inspect dataset from the MCQ JSONL file."""
    samples: List[Sample] = list(_load_samples(MCQ_DATASET_PATH))
    return MemoryDataset(samples)


def build_historical_mcq_dataset() -> Dataset:
    """Create an Inspect dataset from the historical MCQ JSONL file."""
    samples: List[Sample] = list(_load_samples(HISTORICAL_MCQ_PATH))
    return MemoryDataset(samples)

def build_agent(use_offline_prompt: bool = True):
    """Configure a React agent confined to the EMNLP sandbox."""
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="emnlp-react",
        prompt=(
            f"{offline_prefix}"
            "You are a research paper quality assessor. Your task is to classify papers into recognition tiers "
            "by comparing them to previous award-winning and accepted papers.\n\n"
            "Recognition tiers:\n"
            "- Best: Best Paper Award winners (groundbreaking, paradigm-shifting work)\n"
            "- Outstanding: Outstanding Paper Awards (exceptional contributions)\n"
            "- Main: Main conference track (solid accepted papers)\n"
            "- Findings: Findings track (good work, below main bar)\n\n"
            "PROCESS:\n"
            "1. Use python() to read and analyze sandbox/data/accepted_papers.csv\n"
            "2. Look at papers with 'Best' or 'Outstanding' awards to understand patterns\n"
            "3. Compare the given paper's novelty, impact, and quality to these examples\n"
            "4. Make your classification decision\n"
            "5. Respond with ONLY ONE WORD: Best, Outstanding, Main, or Findings\n\n"
            "CRITICAL: After using tools to investigate, provide your final answer as a SINGLE WORD on its own line. "
            "Do not add explanations after your answer. The answer must be exactly one of: Best, Outstanding, Main, Findings"
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
def emnlp_awards_mcq_task() -> Task:
    """Multiple-choice benchmark - ReAct agent analyzes previous papers to make classification."""
    dataset = build_mcq_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        max_messages=30,  # Prevent infinite loops
        metadata={"benchmark": "emnlp_awards_mcq"},
    )


@task()
def emnlp_awards_mcq_local_task() -> Task:
    """Same as emnlp_awards_mcq_task but without Docker sandbox (direct file access)."""
    dataset = build_mcq_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=30,
        metadata={"benchmark": "emnlp_awards_mcq_local"},
    )


@task()
def emnlp_awards_mcq_no_offline_prompt_task() -> Task:
    """Same as emnlp_awards_mcq_task but without the shared offline Antigravity preamble."""
    dataset = build_mcq_dataset()
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        max_messages=30,
        metadata={"benchmark": "emnlp_awards_mcq_no_offline"},
    )


@task()
def emnlp_awards_mcq_no_offline_prompt_local_task() -> Task:
    """No-preamble variant without Docker sandbox (direct file access)."""
    dataset = build_mcq_dataset()
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=30,
        metadata={"benchmark": "emnlp_awards_mcq_no_offline_local"},
    )


@task()
def emnlp_awards_mcq_simple_task() -> Task:
    """Simple MCQ benchmark - direct generation without tools (fast but no investigation)."""
    dataset = build_mcq_dataset()
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are an expert at classifying research papers into conference recognition tiers. "
                "Given a paper's title and abstract, determine which tier it belongs to:\n\n"
                "- **Best**: Best Paper Award winners (groundbreaking, top 0.1% contributions)\n"
                "- **Outstanding**: Outstanding Paper Award (exceptional quality, top 1%)\n"
                "- **Main**: Main conference track (high quality, accepted papers)\n"
                "- **Findings**: Findings track (good work, didn't meet main conference bar)\n\n"
                "Respond with ONLY ONE WORD: Best, Outstanding, Main, or Findings.\n"
                "Do not explain your reasoning. Just output the single tier name."
            ),
            generate(),
        ],
        scorer=match(),
        metadata={"benchmark": "emnlp_awards_mcq_simple"},
    )


@task()
def emnlp_historical_mcq_task() -> Task:
    """Multiple-choice benchmark sampling historical main/findings papers."""
    dataset = build_historical_mcq_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        max_messages=30,  # Limit conversation turns to prevent loops
        metadata={"benchmark": "emnlp_historical_mcq"},
    )


@task()
def emnlp_historical_mcq_local_task() -> Task:
    """Historical MCQ benchmark without Docker sandbox (direct file access)."""
    dataset = build_historical_mcq_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=30,
        metadata={"benchmark": "emnlp_historical_mcq_local"},
    )


@task()
def emnlp_historical_mcq_no_offline_prompt_task() -> Task:
    """Historical MCQ benchmark without the shared offline Antigravity preamble."""
    dataset = build_historical_mcq_dataset()
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        max_messages=30,
        metadata={"benchmark": "emnlp_historical_mcq_no_offline"},
    )


@task()
def emnlp_historical_mcq_no_offline_prompt_local_task() -> Task:
    """Historical MCQ benchmark (no preamble) without Docker sandbox."""
    dataset = build_historical_mcq_dataset()
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        max_messages=30,
        metadata={"benchmark": "emnlp_historical_mcq_no_offline_local"},
    )
