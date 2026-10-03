from __future__ import annotations

from typing import Any


def _value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def explain_issue(issue: dict[str, Any]) -> str:
    """Render a deterministic explanation using only fields already on the issue."""
    issue_type = str(issue.get("issue_type", "reconciliation difference")).replace("_", " ")
    expected = _value(issue.get("expected"))
    actual = _value(issue.get("actual"))
    impact = _value(issue.get("rupee_impact", 0))
    vendor = str(issue.get("vendor", "Unknown"))
    suggestion = str(issue.get("suggested_fix", "Review the source documents."))
    return (
        f"{vendor}: this case is flagged as {issue_type}. "
        f"The expected value is {expected}; the observed value is {actual}. "
        f"The deterministic impact estimate is Rs {impact}. "
        f"Recommended action: {suggestion}"
    )
