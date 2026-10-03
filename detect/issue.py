from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Issue:
    issue_id: str
    issue_type: str
    severity: str
    vendor: str
    record_ids: list[str] = field(default_factory=list)
    expected: Any = None
    actual: Any = None
    rupee_impact: float = 0.0
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_fix: str = "Review manually"
    status: str = "Open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "vendor": self.vendor,
            "record_ids": self.record_ids,
            "expected": self.expected,
            "actual": self.actual,
            "rupee_impact": self.rupee_impact,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
            "status": self.status,
        }


def build_demo_issue(issue_type: str, vendor: str, record_ids: list[str], impact: float) -> Issue:
    """Create a deterministic demo issue object without LLM-generated numbers."""
    return Issue(
        issue_id=f"ISS-{issue_type.upper()}-{len(record_ids)}",
        issue_type=issue_type,
        severity="High" if impact > 5000 else "Medium",
        vendor=vendor,
        record_ids=record_ids,
        expected="Matched and aligned",
        actual="Mismatch or missing record",
        rupee_impact=float(impact),
        confidence=0.86,
        evidence={"source": "synthetic demo", "impact": impact},
        suggested_fix="Review the linked records and confirm the corrected document or payment reference.",
    )
