from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd

from detect.common import guard_inference, issue
from detect.rules import is_valid_gstin, load_thresholds, money


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _effective_rate(
    rates: pd.DataFrame,
    hsn: Any,
    invoice_date: date,
) -> Decimal | None:
    rows = rates[rates["HSN"].astype(str).str.lstrip("0").eq(str(hsn).lstrip("0"))]
    for row in rows.to_dict("records"):
        start = _date(row.get("valid_from"))
        end = _date(row.get("valid_to")) or date(2100, 1, 1)
        if start and start <= invoice_date <= end:
            return Decimal(str(row["standard_tax_rate_pct"]))
    return None


def verify_tax(
    invoices: pd.DataFrame,
    rates: pd.DataFrame,
    filings: pd.DataFrame | None = None,
    home_state: str = "Maharashtra",
) -> list:
    """Verify effective-dated GST rates, components, totals, GSTIN and filing tax."""
    guard_inference(invoices=invoices, rates=rates, filings=filings)
    tolerance = Decimal(str(load_thresholds()["money_tolerance_rupees"]))
    issues = []

    for row in invoices.to_dict("records"):
        record_id = str(row["record_id"])
        invoice_date = _date(row.get("date"))
        vendor = str(row.get("vendor_customer_name", "") or "Unknown")
        taxable = money(row.get("taxable_value"))
        cgst, sgst, igst = (money(row.get(field)) for field in ("CGST", "SGST", "IGST"))
        actual_tax = cgst + sgst + igst
        total = money(row.get("total"))
        declared_rate = Decimal(str(row.get("tax_rate_pct", 0) or 0))
        effective_rate = _effective_rate(rates, row.get("HSN"), invoice_date) if invoice_date else None
        calculated_declared_tax = money(taxable * declared_rate / Decimal("100"))
        calculated_total_tax = money(taxable * effective_rate / Decimal("100")) if effective_rate is not None else None

        if effective_rate is None:
            issues.append(issue(
                "unknown_tax_rate", "Medium", vendor, [record_id],
                "An effective-dated tax rate for this HSN", str(row.get("HSN")),
                Decimal("0"), {"hsn": str(row.get("HSN")), "invoice_date": str(row.get("date"))},
                "Confirm the applicable HSN rate and maintain the effective-dated rate master.",
            ))
        elif declared_rate != effective_rate:
            impact = abs((calculated_total_tax or Decimal("0")) - actual_tax)
            issues.append(issue(
                "wrong_tax_rate", "High", vendor, [record_id],
                f"{effective_rate:.2f}%", f"{declared_rate:.2f}%",
                impact, {
                    "hsn": str(row.get("HSN")),
                    "invoice_date": str(row.get("date")),
                    "effective_rate": float(effective_rate),
                    "actual_tax": float(actual_tax),
                    "expected_tax": float(calculated_total_tax or 0),
                },
                "Correct the tax rate using the effective-dated HSN master and recalculate tax.",
            ))

        if effective_rate is not None and abs(actual_tax - calculated_declared_tax) > tolerance:
            issues.append(issue(
                "miscalculated_tax", "High", vendor, [record_id],
                f"{calculated_declared_tax:.2f}", f"{actual_tax:.2f}",
                abs(calculated_declared_tax - actual_tax),
                {"taxable_value": float(taxable), "declared_rate_pct": float(declared_rate)},
                "Recalculate tax on the taxable value and correct the invoice or books.",
            ))

        interstate = str(row.get("party_state", "")).strip().casefold() != home_state.casefold()
        if interstate:
            expected_parts = (Decimal("0.00"), Decimal("0.00"), actual_tax)
        else:
            half_tax = money(actual_tax / Decimal("2"))
            expected_parts = (half_tax, half_tax, Decimal("0.00"))
        component_diffs = [
            abs(actual - expected)
            for actual, expected in zip((cgst, sgst, igst), expected_parts)
        ]
        if any(diff > tolerance for diff in component_diffs):
            issues.append(issue(
                "wrong_tax_split", "High", vendor, [record_id],
                {"CGST": float(expected_parts[0]), "SGST": float(expected_parts[1]), "IGST": float(expected_parts[2])},
                {"CGST": float(cgst), "SGST": float(sgst), "IGST": float(igst)},
                max(component_diffs),
                {"party_state": row.get("party_state"), "reporting_state": home_state},
                "Recompute whether the supply is intra-state or inter-state, then correct the tax components.",
            ))

        calculated_total = money(taxable + actual_tax)
        if abs(total - calculated_total) > tolerance:
            issues.append(issue(
                "invoice_total_mismatch", "High", vendor, [record_id],
                f"{calculated_total:.2f}", f"{total:.2f}", abs(total - calculated_total),
                {"taxable_value": float(taxable), "tax_components": float(actual_tax)},
                "Correct the invoice total so it equals taxable value plus GST.",
            ))

        gstin = str(row.get("GSTIN", "") or "")
        if not is_valid_gstin(gstin):
            issues.append(issue(
                "invalid_gstin", "High", vendor, [record_id],
                "15-character GSTIN with Z in position 14 and a valid checksum",
                gstin, actual_tax,
                {"gstin_length": len(gstin), "fourteenth_character": gstin[13:14]},
                "Verify the supplier GSTIN against official records before claiming ITC.",
            ))

    if filings is not None and not filings.empty:
        invoice_index = {
            (str(row.get("invoice_no", "")).upper(), str(row.get("GSTIN", "")).upper()): row
            for row in invoices.to_dict("records")
        }
        for filing in filings.to_dict("records"):
            key = (str(filing.get("invoice_no", "")).upper(), str(filing.get("supplier_GSTIN", "")).upper())
            invoice = invoice_index.get(key)
            if invoice is None:
                continue
            book_tax = money(invoice.get("CGST", 0)) + money(invoice.get("SGST", 0)) + money(invoice.get("IGST", 0))
            reported_tax = money(filing.get("tax_amount"))
            if abs(book_tax - reported_tax) > tolerance:
                issues.append(issue(
                    "gstr2b_tax_mismatch", "High",
                    str(invoice.get("vendor_customer_name", "Unknown")),
                    [str(invoice["record_id"]), str(filing["filing_id"])],
                    f"{book_tax:.2f}", f"{reported_tax:.2f}", abs(book_tax - reported_tax),
                    {"invoice_no": invoice.get("invoice_no"), "supplier_gstin": key[1]},
                    "Reconcile the supplier filing to the invoice and request an amendment if needed.",
                ))
    return issues
