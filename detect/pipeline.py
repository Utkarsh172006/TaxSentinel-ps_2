from __future__ import annotations

from typing import Any

import pandas as pd

from detect.anomaly import detect_anomalies
from detect.common import guard_inference
from detect.duplicates import detect_duplicates
from detect.issue import Issue
from detect.mismatch import detect_mismatches
from detect.missing import detect_missing_records
from detect.tax_verify import verify_tax


def detect_all(
    invoices: pd.DataFrame,
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    filings: pd.DataFrame,
    rates: pd.DataFrame,
    classifier: Any | None = None,
) -> list[Issue]:
    """Run all deterministic discrepancy and anomaly detectors on safe inputs."""
    guard_inference(
        invoices=invoices,
        bank=bank,
        ledger=ledger,
        filings=filings,
        rates=rates,
    )
    found = []
    found.extend(detect_mismatches(invoices, bank, ledger, filings))
    found.extend(detect_duplicates(invoices, bank))
    found.extend(detect_missing_records(invoices, bank, ledger, filings, classifier))
    found.extend(verify_tax(invoices, rates, filings))
    anomaly_issues, _ = detect_anomalies(invoices, bank)
    found.extend(anomaly_issues)

    unique: dict[str, Issue] = {}
    for detected in found:
        unique.setdefault(detected.issue_id, detected)
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    return sorted(
        unique.values(),
        key=lambda item: (severity_order.get(item.severity, 4), item.issue_type, item.issue_id),
    )
