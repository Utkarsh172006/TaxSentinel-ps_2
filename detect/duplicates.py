from __future__ import annotations

from datetime import date
from decimal import Decimal
from itertools import combinations
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from detect.common import guard_inference, issue
from detect.rules import load_thresholds, money
from matching.features import normalize_identifier, normalize_vendor_name


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def detect_duplicates(invoices: pd.DataFrame, bank: pd.DataFrame | None = None) -> list:
    """Find duplicate/near-duplicate invoices and duplicated bank transactions."""
    guard_inference(invoices=invoices, bank=bank)
    near_days = int(load_thresholds()["duplicate_near_days"])
    issues = []
    rows = invoices.to_dict("records")
    by_vendor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("GSTIN", "")).upper() or normalize_vendor_name(row.get("vendor_customer_name"))
        by_vendor.setdefault(key, []).append(row)

    emitted: set[tuple[str, str]] = set()
    for group in by_vendor.values():
        for left, right in combinations(group, 2):
            left_id, right_id = str(left["record_id"]), str(right["record_id"])
            pair = tuple(sorted((left_id, right_id)))
            if pair in emitted:
                continue
            left_amount, right_amount = money(left.get("total")), money(right.get("total"))
            if abs(left_amount - right_amount) > 1:
                continue
            left_no = normalize_identifier(left.get("invoice_no"))
            right_no = normalize_identifier(right.get("invoice_no"))
            id_similarity = fuzz.ratio(left_no, right_no)
            left_date, right_date = _date(left.get("date")), _date(right.get("date"))
            date_gap = abs((left_date - right_date).days) if left_date and right_date else 9999
            if left_no == right_no:
                issue_type = "duplicate_exact"
            elif date_gap <= near_days and id_similarity >= 75:
                issue_type = "duplicate_near"
            else:
                continue
            emitted.add(pair)
            issues.append(issue(
                issue_type,
                "High",
                str(left.get("vendor_customer_name", "Unknown")),
                [left_id, right_id],
                "One valid invoice record",
                "Two records with matching vendor and amount",
                min(left_amount, right_amount),
                {
                    "invoice_numbers": [str(left.get("invoice_no")), str(right.get("invoice_no"))],
                    "identifier_similarity": id_similarity,
                    "date_gap_days": date_gap,
                    "GSTIN": str(left.get("GSTIN", "")),
                },
                "Confirm the original document and reverse or merge the duplicate posting.",
                confidence=0.98 if issue_type == "duplicate_exact" else 0.88,
            ))

    if bank is not None and not bank.empty:
        payment_groups: dict[tuple[str, Decimal, int], list[dict[str, Any]]] = {}
        for payment in bank.to_dict("records"):
            reference = normalize_identifier(payment.get("reference"))
            amount = money(abs(payment.get("amount", 0)))
            sign = 1 if float(payment.get("amount", 0) or 0) >= 0 else -1
            if reference:
                payment_groups.setdefault((reference, amount, sign), []).append(payment)
        for (reference, amount, _), group in payment_groups.items():
            for left, right in combinations(group, 2):
                if "PART" in normalize_identifier(left.get("narration")) or "PART" in normalize_identifier(right.get("narration")):
                    continue
                left_date, right_date = _date(left.get("date")), _date(right.get("date"))
                date_gap = abs((left_date - right_date).days) if left_date and right_date else 9999
                if date_gap > 10:
                    continue
                issues.append(issue(
                    "duplicate_payment",
                    "High",
                    str(left.get("narration", "Unknown")),
                    [str(left["txn_id"]), str(right["txn_id"])],
                    "One settlement transaction",
                    "Repeated reference and amount",
                    amount,
                    {"reference": reference, "date_gap_days": date_gap},
                    "Confirm both bank entries and reverse any duplicate settlement posting.",
                    confidence=0.92,
                ))
    return issues
