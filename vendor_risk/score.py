from __future__ import annotations

from collections import defaultdict

import pandas as pd

from detect.common import guard_inference
from detect.rules import money

SEVERITY_POINTS = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def score_vendors(invoices: pd.DataFrame, issues: list) -> pd.DataFrame:
    """Score vendor issue density from actual invoices and deterministic issues."""
    guard_inference(invoices=invoices)
    records = invoices.to_dict("records")
    by_record = {str(row["record_id"]): row for row in records}
    aggregates: dict[str, dict[str, float | int | set[str]]] = defaultdict(
        lambda: {"invoice_count": 0, "exposure": 0.0, "weighted_points": 0, "issue_ids": set()}
    )
    for row in records:
        vendor = str(row.get("vendor_customer_name", "Unknown"))
        aggregate = aggregates[vendor]
        aggregate["invoice_count"] = int(aggregate["invoice_count"]) + 1
        if str(row.get("document_type", "purchase")).lower() == "purchase":
            aggregate["exposure"] = float(aggregate["exposure"]) + float(money(row.get("total")))

    for item in issues:
        for record_id in item.record_ids:
            row = by_record.get(str(record_id))
            if row is None:
                continue
            vendor = str(row.get("vendor_customer_name", "Unknown"))
            aggregate = aggregates[vendor]
            issue_ids = aggregate["issue_ids"]
            if item.issue_id in issue_ids:
                continue
            issue_ids.add(item.issue_id)
            aggregate["weighted_points"] = int(aggregate["weighted_points"]) + SEVERITY_POINTS.get(item.severity, 0)

    output = []
    for vendor, aggregate in aggregates.items():
        invoice_count = int(aggregate["invoice_count"])
        points = int(aggregate["weighted_points"])
        denominator = max(3 * invoice_count, 1)
        score = min(100.0, round(100.0 * points / denominator, 1))
        output.append({
            "vendor": vendor,
            "risk_score": score,
            "invoice_count": invoice_count,
            "weighted_issue_points": points,
            "purchase_exposure": round(float(aggregate["exposure"]), 2),
            "issue_count": len(aggregate["issue_ids"]),
        })
    return pd.DataFrame(
        output,
        columns=[
            "vendor", "risk_score", "invoice_count", "weighted_issue_points",
            "purchase_exposure", "issue_count",
        ],
    ).sort_values(
        ["risk_score", "purchase_exposure"], ascending=[False, False], ignore_index=True
    )
