from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from detect.common import guard_inference, issue
from detect.rules import load_thresholds, money, name_similarity, normalized_tax_id
from matching.blocking import generate_pair_candidates
from matching.features import normalize_identifier


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _vendor(row: dict[str, Any]) -> str:
    return str(row.get("vendor_customer_name", row.get("supplier_name", "")) or "")


def _filing_candidates(
    invoice: dict[str, Any],
    filing_groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    invoice_date = _date(invoice.get("date"))
    invoice_id = normalize_identifier(invoice.get("invoice_no"))
    invoice_gstin = normalized_tax_id(invoice.get("GSTIN"))
    candidates = []
    possible = []
    if invoice_date is not None:
        for shift in range(-5, 6):
            possible.extend(filing_groups.get((invoice_date + pd.Timedelta(days=shift)).isoformat(), []))
    for filing in possible:
        filing_date = _date(filing.get("invoice_date"))
        date_gap = abs((filing_date - invoice_date).days) if filing_date and invoice_date else 9999
        filing_gstin = normalized_tax_id(filing.get("supplier_GSTIN"))
        vendor_score = name_similarity(_vendor(invoice), filing.get("supplier_name"))
        id_score = fuzz.ratio(invoice_id, normalize_identifier(filing.get("invoice_no"))) if invoice_id else 0
        if date_gap <= 5 and (invoice_gstin == filing_gstin or vendor_score >= 65 or id_score >= 70):
            taxable_gap = abs(
                float(invoice.get("taxable_value", 0) or 0)
                - float(filing.get("taxable_value", 0) or 0)
            )
            expected_taxable = max(abs(float(invoice.get("taxable_value", 0) or 0)), 1.0)
            amount_score = max(0.0, 100.0 - 100.0 * taxable_gap / expected_taxable)
            score = (
                id_score * 1.5
                + vendor_score
                + amount_score
                + int(invoice_gstin == filing_gstin) * 100
                - date_gap * 5
            )
            candidates.append((score, filing))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in candidates[:3]]


def detect_mismatches(
    invoices: pd.DataFrame,
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    filings: pd.DataFrame,
) -> list:
    """Detect amount, date, identifier, name and filing-field discrepancies."""
    guard_inference(invoices=invoices, bank=bank, ledger=ledger, filings=filings)
    tolerance = Decimal(str(load_thresholds()["money_tolerance_rupees"]))
    issues = []
    payment_rows = {str(row["txn_id"]): row for row in bank.to_dict("records")}
    bank_candidates = generate_pair_candidates(invoices, bank)
    candidates_by_invoice: dict[str, list[dict[str, Any]]] = {}
    for candidate in bank_candidates.to_dict("records"):
        candidates_by_invoice.setdefault(str(candidate["invoice_record_id"]), []).append(candidate)
    filing_groups: dict[str, list[dict[str, Any]]] = {}
    for row in filings.to_dict("records"):
        filing_date = _date(row.get("invoice_date"))
        if filing_date:
            filing_groups.setdefault(filing_date.isoformat(), []).append(row)
    ledger_rows = ledger.to_dict("records")
    ledger_by_reference: dict[str, list[dict[str, Any]]] = {}
    for row in ledger_rows:
        ledger_by_reference.setdefault(normalize_identifier(row.get("reference")), []).append(row)
    ledger_by_amount: dict[Decimal, list[dict[str, Any]]] = {}
    for row in ledger_rows:
        amount = money(abs(row.get("debit", 0)) + abs(row.get("credit", 0)))
        ledger_by_amount.setdefault(amount, []).append(row)

    for invoice in invoices.to_dict("records"):
        record_id = str(invoice["record_id"])
        vendor = _vendor(invoice)
        invoice_date = _date(invoice.get("date"))
        invoice_total = money(invoice.get("total"))
        payments = [
            payment_rows[row["payment_id"]]
            for row in candidates_by_invoice.get(record_id, [])
            if row["payment_id"] in payment_rows
        ]
        invoice_key = normalize_identifier(invoice.get("invoice_no"))
        # Do not compare each split payment in isolation; aggregate a coherent reference group.
        direct = [row for row in payments if invoice_key and invoice_key in
                  normalize_identifier(f"{row.get('reference', '')} {row.get('narration', '')}")]
        earlier_payments = [
            row for row in direct
            if invoice_date is not None
            and _date(row.get("date")) is not None
            and _date(row.get("date")) < invoice_date
        ]
        for payment in earlier_payments:
            payment_date = _date(payment.get("date"))
            issues.append(issue(
                "payment_before_invoice",
                "Medium",
                vendor,
                [record_id, str(payment["txn_id"])],
                invoice_date.isoformat(),
                payment_date.isoformat() if payment_date else "",
                Decimal("0"),
                {"invoice_date": invoice_date.isoformat(), "payment_date": payment_date.isoformat() if payment_date else ""},
                "Verify the invoice and bank dates, then correct the source record or payment reference.",
            ))
        if direct:
            groups = {}
            for row in direct:
                # Bulk payments are evaluated by the matcher as an invoice group, not
                # as a full payment against each member invoice.
                if "BULK" in normalize_identifier(row.get("reference")):
                    continue
                group_key = normalize_identifier(row.get("reference")) or str(row.get("txn_id"))
                groups.setdefault(group_key, []).append(row)
            if groups:
                chosen = max(groups.values(), key=lambda rows: len(rows))
                actual = sum(
                    (money(abs(row.get("amount", 0))) + money(abs(row.get("tds_withheld", 0))) for row in chosen),
                    Decimal("0"),
                )
                if abs(invoice_total - actual) > tolerance:
                    issue_type = "short_payment" if actual < invoice_total else "overpayment"
                    issues.append(issue(
                        issue_type,
                        "High",
                        vendor,
                        [record_id, *[str(row["txn_id"]) for row in chosen]],
                        f"{invoice_total:.2f}",
                        f"{actual:.2f}",
                        abs(invoice_total - actual),
                        {"invoice_total": float(invoice_total), "payment_total_including_tds": float(actual)},
                        "Confirm withholding or recover the payment difference.",
                    ))

        likely_ledger = []
        exact_ref = ledger_by_reference.get(invoice_key, [])
        led_candidates = exact_ref or ledger_by_amount.get(invoice_total, [])
        for row in led_candidates:
            amount = money(abs(row.get("debit", 0)) + abs(row.get("credit", 0)))
            posting_date = _date(row.get("date"))
            date_gap = abs((posting_date - invoice_date).days) if posting_date and invoice_date else 9999
            ref_score = fuzz.ratio(invoice_key, normalize_identifier(row.get("reference"))) if invoice_key else 0
            if ref_score >= 65 or (date_gap <= 3 and abs(amount - invoice_total) <= tolerance):
                likely_ledger.append((ref_score - date_gap, row, amount))
        if likely_ledger:
            likely_ledger.sort(key=lambda item: item[0], reverse=True)
            score, row, ledger_amount = likely_ledger[0]
            if normalize_identifier(row.get("reference")) != normalize_identifier(invoice.get("invoice_no")):
                issues.append(issue(
                    "invoice_id_mismatch", "Medium", vendor,
                    [record_id, str(row["entry_id"])],
                    str(invoice.get("invoice_no")), str(row.get("reference")),
                    Decimal("0"),
                    {"identifier_similarity": float(score), "source": "ledger reference"},
                    "Verify the source invoice number and correct the ledger reference.",
                    confidence=min(0.99, max(0.60, score / 100)),
                ))
            if abs(ledger_amount - invoice_total) > tolerance:
                issues.append(issue(
                    "ledger_amount_mismatch", "High", vendor,
                    [record_id, str(row["entry_id"])],
                    f"{invoice_total:.2f}", f"{ledger_amount:.2f}",
                    abs(invoice_total - ledger_amount),
                    {"ledger_reference": row.get("reference")},
                    "Reconcile the ledger posting to the source invoice.",
                ))
            posting_date = _date(row.get("date"))
            if invoice_date and posting_date and posting_date != invoice_date:
                gap = abs((posting_date - invoice_date).days)
                if gap > 1:
                    issues.append(issue(
                        "ledger_date_gap", "Medium", vendor,
                        [record_id, str(row["entry_id"])],
                        invoice_date.isoformat(), posting_date.isoformat(),
                        Decimal("0"),
                        {"date_gap_days": gap},
                        "Confirm the posting period and update the ledger date if required.",
                    ))

        candidate_filings = (
            _filing_candidates(invoice, filing_groups)
            if str(invoice.get("document_type", "purchase")).lower() == "purchase"
            else []
        )
        if candidate_filings:
            filing = candidate_filings[0]
            filing_id = str(filing["filing_id"])
            expected_tax = money(
                Decimal(str(invoice.get("CGST", 0) or 0))
                + Decimal(str(invoice.get("SGST", 0) or 0))
                + Decimal(str(invoice.get("IGST", 0) or 0))
            )
            filing_tax = money(filing.get("tax_amount"))
            if abs(expected_tax - filing_tax) > tolerance:
                issues.append(issue(
                    "gstr2b_tax_mismatch", "High", vendor,
                    [record_id, filing_id],
                    f"{expected_tax:.2f}", f"{filing_tax:.2f}",
                    abs(expected_tax - filing_tax),
                    {"invoice_number": invoice.get("invoice_no"), "gstin": invoice.get("GSTIN")},
                    "Ask the supplier to correct the filed tax amount or investigate the invoice.",
                ))
            expected_period = str(invoice.get("date", ""))[:7]
            actual_period = str(filing.get("filing_period", ""))
            if actual_period != expected_period:
                issues.append(issue(
                    "wrong_tax_period", "Medium", vendor,
                    [record_id, filing_id],
                    expected_period, actual_period, Decimal("0"),
                    {"invoice_date": str(invoice.get("date"))},
                    "Verify the return period and request correction if the filing period is wrong.",
                ))
            if normalized_tax_id(invoice.get("invoice_no")) != normalized_tax_id(filing.get("invoice_no")):
                issues.append(issue(
                    "invoice_id_mismatch", "Medium", vendor,
                    [record_id, filing_id],
                    str(invoice.get("invoice_no")), str(filing.get("invoice_no")), Decimal("0"),
                    {"source": "supplier filing"},
                    "Verify the invoice number with the supplier and correct the books or return.",
                ))
            if name_similarity(_vendor(invoice), filing.get("supplier_name")) < 75:
                issues.append(issue(
                    "vendor_name_mismatch", "Low", vendor,
                    [record_id, filing_id],
                    _vendor(invoice), str(filing.get("supplier_name")), Decimal("0"),
                    {"name_similarity": name_similarity(_vendor(invoice), filing.get("supplier_name"))},
                    "Confirm the supplier legal name and standardize the vendor master.",
                    confidence=0.80,
                ))
            invoice_gstin = normalized_tax_id(invoice.get("GSTIN"))
            filing_gstin = normalized_tax_id(filing.get("supplier_GSTIN"))
            if invoice_gstin != filing_gstin:
                issues.append(issue(
                    "supplier_gstin_mismatch", "High", vendor,
                    [record_id, filing_id],
                    invoice_gstin, filing_gstin, expected_tax,
                    {"invoice_id": invoice.get("invoice_no")},
                    "Confirm the supplier GSTIN and assess the related ITC before claiming it.",
                ))

    return issues
