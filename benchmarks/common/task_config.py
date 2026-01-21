"""Shared configuration for Inspect AI benchmark tasks.

This module provides standardized timeout and retry configurations to ensure
robust evaluation across all benchmarks.
"""

from inspect_ai.model import GenerateConfig

# =============================================================================
# Model-level configurations (API retry and timeout behavior)
# =============================================================================

AGENT_TASK_CONFIG = GenerateConfig(
    max_retries=3,           # Retry API calls up to 3 times on failure
    timeout=600,             # 10 min total timeout (includes all retries)
    attempt_timeout=120,     # 2 min timeout per individual attempt
    max_connections=10,      # Limit concurrent API connections
)
"""Configuration for agent-based tasks (React agents with tools)."""

SIMPLE_TASK_CONFIG = GenerateConfig(
    max_retries=3,           # Retry API calls up to 3 times on failure
    timeout=180,             # 3 min total timeout for zero-shot tasks
    attempt_timeout=60,      # 1 min timeout per individual attempt
    max_connections=10,      # Limit concurrent API connections
)
"""Configuration for simple tasks (zero-shot, no tools)."""

# =============================================================================
# Task-level limits (sample execution behavior)
# =============================================================================

AGENT_TASK_LIMITS = {
    "message_limit": 30,     # Max 30 conversation turns per sample
    "time_limit": 600,       # 10 min wall-clock time per sample (includes I/O, retries)
    "working_limit": 480,    # 8 min working time per sample (excludes waiting)
    "fail_on_error": False,  # Don't fail entire eval on first error
    "continue_on_fail": True,# Continue evaluating remaining samples after errors
}
"""Limits for agent-based tasks with 30 message turns."""

AGENT_TASK_LIMITS_40 = {
    "message_limit": 40,     # Max 40 conversation turns per sample (for complex tasks)
    "time_limit": 900,       # 15 min wall-clock time per sample
    "working_limit": 720,    # 12 min working time per sample
    "fail_on_error": False,
    "continue_on_fail": True,
}
"""Limits for complex agent-based tasks requiring more turns (e.g., faculty tasks)."""

SIMPLE_TASK_LIMITS = {
    "time_limit": 120,       # 2 min wall-clock time per sample
    "working_limit": 90,     # 1.5 min working time per sample
    "fail_on_error": False,
    "continue_on_fail": True,
}
"""Limits for simple/zero-shot tasks without agents."""

# =============================================================================
# Usage Examples
# =============================================================================

# Agent task with standard limits (30 messages):
# return Task(
#     dataset=dataset,
#     solver=agent,
#     scorer=match(),
#     sandbox="docker",
#     config=AGENT_TASK_CONFIG,
#     **AGENT_TASK_LIMITS,
#     metadata={"benchmark": "..."},
# )

# Agent task with extended limits (40 messages):
# return Task(
#     dataset=dataset,
#     solver=agent,
#     scorer=match(),
#     sandbox="docker",
#     config=AGENT_TASK_CONFIG,
#     **AGENT_TASK_LIMITS_40,
#     metadata={"benchmark": "..."},
# )

# Simple/zero-shot task:
# return Task(
#     dataset=dataset,
#     solver=[system_message(...), generate()],
#     scorer=match(),
#     config=SIMPLE_TASK_CONFIG,
#     **SIMPLE_TASK_LIMITS,
#     metadata={"benchmark": "..."},
# )
