from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Any

import pandas as pd

from detect.issue import Issue
from ingest.loader import assert_inference_safe


def guard_inference(**tables: pd.DataFrame | None) -> None:
    assert_inference_safe({
        f"{name}.csv": frame for name, frame in tables.items() if frame is not None
    })


def issue(
    issue_type: str,
    severity: str,
    vendor: str,
    record_ids: list[str],
    expected: Any,
    actual: Any,
    rupee_impact: Decimal | float | int,
    evidence: dict[str, Any],
    suggested_fix: str,
    confidence: float = 0.95,
) -> Issue:
    stable_key = "|".join([issue_type, *sorted(record_ids), str(expected), str(actual)])
    issue_id = f"ISS-{hashlib.sha1(stable_key.encode('utf-8')).hexdigest()[:12].upper()}"
    return Issue(
        issue_id=issue_id,
        issue_type=issue_type,
        severity=severity,
        vendor=vendor or "Unknown",
        record_ids=record_ids,
        expected=expected,
        actual=actual,
        rupee_impact=round(float(rupee_impact), 2),
        confidence=min(max(float(confidence), 0.0), 1.0),
        evidence=evidence,
        suggested_fix=suggested_fix,
    )
