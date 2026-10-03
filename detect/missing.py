from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd

from detect.common import guard_inference, issue
from detect.rules import load_thresholds, money
from matching.features import normalize_identifier
from matching.pipeline import reconcile


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def detect_missing_records(
    invoices: pd.DataFrame,
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    filings: pd.DataFrame,
    classifier: Any | None = None,
) -> list:
    """Find missing/unmatched records while exempting legitimate recent open items."""
    guard_inference(invoices=invoices, bank=bank, ledger=ledger, filings=filings)
    thresholds = load_thresholds()
    open_after = date.fromisoformat(thresholds["open_item_after"])
    tolerance = Decimal(str(thresholds["money_tolerance_rupees"]))
    result = reconcile(invoices, bank, ledger, filings, classifier)
    accepted_invoice_ids = {
        invoice_id for match in result.matches if match.confidence >= 0.60 for invoice_id in match.invoice_ids
    }
    accepted_payment_ids = {
        payment_id for match in result.matches if match.confidence >= 0.60 for payment_id in match.payment_ids
    }
    invoice_rows = invoices.to_dict("records")
    invoice_by_number = {
        normalize_identifier(row.get("invoice_no")): row for row in invoice_rows
        if normalize_identifier(row.get("invoice_no"))
    }
    issues = []

    for invoice in invoice_rows:
        invoice_id = str(invoice["record_id"])
        invoice_date = _date(invoice.get("date"))
        if invoice_id not in accepted_invoice_ids and (invoice_date is None or invoice_date <= open_after):
            impact = money(invoice.get("total"))
            issues.append(issue(
                "missing_payment", "High", str(invoice.get("vendor_customer_name", "Unknown")),
                [invoice_id], f"Payment by {thresholds['payment_delay_max_days']} days",
                "No confident payment match", impact,
                {"invoice_date": str(invoice.get("date")), "open_item_exemption_after": open_after.isoformat()},
                "Check the bank statement and confirm whether the invoice remains outstanding.",
                confidence=0.85,
            ))

        if str(invoice.get("document_type", "purchase")).lower() == "purchase":
            invoice_no = normalize_identifier(invoice.get("invoice_no"))
            gstin = str(invoice.get("GSTIN", "")).upper()
            matching_filings = filings[
                filings.invoice_no.astype(str).map(normalize_identifier).eq(invoice_no)
                & filings.supplier_GSTIN.astype(str).str.upper().eq(gstin)
            ]
            if matching_filings.empty:
                issues.append(issue(
                    "missing_supplier_filing", "High", str(invoice.get("vendor_customer_name", "Unknown")),
                    [invoice_id], "Supplier invoice in GSTR-2B", "No matching filing",
                    money(invoice.get("CGST", 0)) + money(invoice.get("SGST", 0)) + money(invoice.get("IGST", 0)),
                    {"invoice_no": invoice.get("invoice_no"), "gstin": gstin},
                    "Confirm filing status with the supplier before claiming input tax credit.",
                ))

    ledger_refs = set(ledger.reference.astype(str).map(normalize_identifier))
    for invoice in invoice_rows:
        reference = normalize_identifier(invoice.get("invoice_no"))
        if reference and reference not in ledger_refs:
            amount = money(invoice.get("total"))
            invoice_date = _date(invoice.get("date"))
            possible = ledger[
                (ledger.debit.astype(float) + ledger.credit.astype(float)).abs().sub(float(amount)).abs() <= float(tolerance)
            ]
            date_close = any(
                abs((_date(posting_date) - invoice_date).days) <= 3
                for posting_date in possible.date
                if invoice_date and _date(posting_date)
            )
            if not date_close:
                issues.append(issue(
                    "missing_ledger_entry", "High", str(invoice.get("vendor_customer_name", "Unknown")),
                    [str(invoice["record_id"])], "Ledger posting exists", "No source-aligned entry",
                    amount, {"invoice_no": invoice.get("invoice_no")},
                    "Post or locate the missing accounts payable/receivable entry.",
                ))

    for payment in bank.to_dict("records"):
        payment_id = str(payment["txn_id"])
        if payment_id in accepted_payment_ids:
            continue
        text = normalize_identifier(f"{payment.get('reference', '')} {payment.get('narration', '')}")
        mentions_invoice = any(invoice_no in text for invoice_no in invoice_by_number)
        if not mentions_invoice:
            issues.append(issue(
                "payment_without_invoice", "Medium", str(payment.get("narration", "Unknown")),
                [payment_id], "Bank transaction linked to an invoice", "No invoice reference found",
                abs(money(payment.get("amount"))),
                {"reference": payment.get("reference"), "narration": payment.get("narration")},
                "Investigate the counterparty and attach the payment to its supporting document.",
            ))

    for ledger_row in ledger.to_dict("records"):
        reference = normalize_identifier(ledger_row.get("reference"))
        if not reference or reference not in invoice_by_number:
            amount = money(abs(ledger_row.get("debit", 0)) + abs(ledger_row.get("credit", 0)))
            issues.append(issue(
                "ledger_without_source", "Medium", str(ledger_row.get("account", "Unknown")),
                [str(ledger_row["entry_id"])], "Source invoice exists", str(ledger_row.get("reference", "")),
                amount, {"ledger_date": str(ledger_row.get("date"))},
                "Locate the originating invoice or correct the ledger reference.",
            ))

    matched_book_ids = {
        (normalize_identifier(row.get("invoice_no")), str(row.get("GSTIN", "")).upper())
        for row in invoice_rows
    }
    for filing in filings.to_dict("records"):
        if not bool(filing.get("appears_in_supplier_filing", True)):
            invoice_no = normalize_identifier(filing.get("invoice_no"))
            candidates = [
                row for row in invoice_rows
                if normalize_identifier(row.get("invoice_no")) == invoice_no
                and str(row.get("GSTIN", "")).upper() == str(filing.get("supplier_GSTIN", "")).upper()
            ]
            if candidates:
                book = candidates[0]
                issues.append(issue(
                    "supplier_not_filed", "High", str(book.get("vendor_customer_name", "Unknown")),
                    [str(book["record_id"]), str(filing["filing_id"])],
                    "Invoice present in supplier statement", "Supplier has not filed this invoice",
                    money(book.get("CGST", 0)) + money(book.get("SGST", 0)) + money(book.get("IGST", 0)),
                    {"invoice_no": filing.get("invoice_no"), "filing_period": filing.get("filing_period")},
                    "Follow up with the supplier and defer the related ITC until it is eligible.",
                ))
        filing_key = (
            normalize_identifier(filing.get("invoice_no")),
            str(filing.get("supplier_GSTIN", "")).upper(),
        )
        if filing_key not in matched_book_ids:
            issues.append(issue(
                "in_gstr2b_not_in_books", "High", str(filing.get("supplier_name", "Unknown")),
                [str(filing["filing_id"])], "Invoice recorded in books", "Supplier filing has no book match",
                money(filing.get("tax_amount")),
                {"invoice_no": filing.get("invoice_no"), "invoice_date": filing.get("invoice_date")},
                "Confirm the purchase and record it if valid, or dispute the supplier filing.",
            ))
    return issues
