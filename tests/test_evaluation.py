from __future__ import annotations

import pandas as pd

from detect.common import issue
from eval.runner import _exact_baseline_pairs, score_issues


def test_issue_scoring_counts_shared_detector_events_once():
    findings = [
        issue("short_payment", "High", "A", ["INV1"], 1000, 900, 100, {}, "Review"),
        issue("short_payment", "High", "B", ["INV2"], 1000, 900, 100, {}, "Review"),
    ]
    labels = pd.DataFrame([{
        "record_id": "INV1",
        "row_id": "TXN1",
        "error_type": "amount_mismatch_small",
    }])

    metrics = score_issues(findings, labels)

    assert metrics["overall"]["tp"] == 1
    assert metrics["overall"]["fp"] == 1
    assert metrics["overall"]["fn"] == 0
    assert metrics["grouped_metrics"]["payment_amount"]["fp"] == 1
    assert metrics["per_error_type"]["amount_mismatch_small"]["recall"] == 1.0


def test_exact_baseline_requires_both_normalized_reference_and_amount():
    invoices = pd.DataFrame([
        {"record_id": "INV1", "invoice_no": "INV-01", "total": 100.0},
        {"record_id": "INV2", "invoice_no": "INV-02", "total": 200.0},
    ])
    bank = pd.DataFrame([
        {"txn_id": "TXN1", "reference": "INV01", "amount": 100.0, "tds_withheld": 0},
        {"txn_id": "TXN2", "reference": "INV-02", "amount": 150.0, "tds_withheld": 0},
    ])

    assert _exact_baseline_pairs(invoices, bank) == {("INV1", "TXN1")}
