from __future__ import annotations

from typing import Any

from explain.templates import explain_issue


def explain(issue: dict[str, Any]) -> dict[str, str]:
    """Return an offline evidence-bound explanation until an LLM adapter is configured."""
    return {"provider": "offline-template", "text": explain_issue(issue)}
