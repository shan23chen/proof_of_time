from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_FALLBACK = (
    "Offline Antigravity rules: operate entirely inside the sandbox; use only the provided tools "
    "(python, bash, bash_session, text editor, think); do not access the internet or external resources; "
    "keep answers concise and follow the required output format exactly."
)


@lru_cache(maxsize=1)
def get_offline_preamble() -> str:
    """Load the shared offline Antigravity prompt text."""
    prompt_path = Path(__file__).resolve().parents[2] / "offline_agent_prompt.md"
    try:
        content = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return _FALLBACK

    if not content:
        return _FALLBACK

    # Keep the prompt compact by collapsing blank lines.
    lines = [line.rstrip() for line in content.splitlines() if line.strip()]
    return "\n".join(lines)
