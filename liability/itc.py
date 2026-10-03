from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from detect.common import guard_inference
from detect.rules import is_valid_gstin, money
from matching.features import normalize_identifier

ITC_RISK_ISSUES = {
    "supplier_not_filed",
    "missing_supplier_filing",
    "gstr2b_tax_mismatch",
    "invalid_gstin",
    "wrong_tax_rate",
    "miscalculated_tax",
    "wrong_tax_split",
    "supplier_gstin_mismatch",
}


def _book_tax(invoice: dict[str, Any]) -> Decimal:
    return sum(
        (money(invoice.get(column, 0)) for column in ("CGST", "SGST", "IGST")),
        Decimal("0"),
    )


def calculate_itc_risk(
    invoices: pd.DataFrame,
    filings: pd.DataFrame,
    issues: list | None = None,
    resolved_issue_ids: set[str] | None = None,
) -> tuple[dict[str, Decimal], list[dict[str, Any]]]:
    """Return eligible ITC by invoice and a row per at-risk purchase invoice."""
    guard_inference(invoices=invoices, filings=filings)
    resolved = resolved_issue_ids or set()
    issue_flags: dict[str, set[str]] = {}
    resolved_flags: dict[str, set[str]] = {}
    for item in issues or []:
        if item.issue_type in ITC_RISK_ISSUES:
            destination = resolved_flags if item.issue_id in resolved else issue_flags
            for record_id in item.record_ids:
                destination.setdefault(str(record_id), set()).add(item.issue_type)

    filing_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for filing in filings.to_dict("records"):
        key = (
            normalize_identifier(filing.get("invoice_no")),
            str(filing.get("supplier_GSTIN", "")).upper(),
        )
        filing_index.setdefault(key, []).append(filing)

    eligible: dict[str, Decimal] = {}
    risk_rows: list[dict[str, Any]] = []
    for invoice in invoices.to_dict("records"):
        if str(invoice.get("document_type", "purchase")).lower() != "purchase":
            continue
        record_id = str(invoice["record_id"])
        book_tax = _book_tax(invoice)
        key = (
            normalize_identifier(invoice.get("invoice_no")),
            str(invoice.get("GSTIN", "")).upper(),
        )
        matching = filing_index.get(key, [])
        reasons = set(issue_flags.get(record_id, set()))
        cleared = resolved_flags.get(record_id, set())
        if not matching:
            if "missing_supplier_filing" not in cleared:
                reasons.add("missing_supplier_filing")
        elif not any(bool(filing.get("appears_in_supplier_filing", True)) for filing in matching):
            if "supplier_not_filed" not in cleared:
                reasons.add("supplier_not_filed")
        if not is_valid_gstin(invoice.get("GSTIN")) and "invalid_gstin" not in cleared:
            reasons.add("invalid_gstin")
        for filing in matching:
            if abs(money(filing.get("tax_amount")) - book_tax) > Decimal("1.00") \
                    and "gstr2b_tax_mismatch" not in cleared:
                reasons.add("gstr2b_tax_mismatch")

        if reasons:
            eligible[record_id] = Decimal("0")
            risk_rows.append({
                "record_id": record_id,
                "vendor": str(invoice.get("vendor_customer_name", "Unknown")),
                "period": str(invoice.get("date", ""))[:7],
                "invoice_no": str(invoice.get("invoice_no", "")),
                "itc_at_risk": book_tax,
                "reasons": sorted(reasons),
            })
        else:
            eligible[record_id] = book_tax
    return eligible, risk_rows
