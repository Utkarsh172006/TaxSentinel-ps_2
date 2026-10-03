from __future__ import annotations

from decimal import Decimal

import pandas as pd

from detect.duplicates import detect_duplicates
from detect.mismatch import detect_mismatches
from detect.missing import detect_missing_records
from detect.rules import gstin_checksum, is_valid_gstin
from detect.tax_verify import verify_tax


def _valid_gstin() -> str:
    prefix = "27ABCDE1234F1Z"
    return prefix + gstin_checksum(prefix)


def _invoice(gstin: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([{
        "record_id": "INVREC1",
        "invoice_no": "INV-001",
        "document_type": "purchase",
        "date": "2026-01-01",
        "vendor_customer_name": "Acme Private Limited",
        "GSTIN": gstin or _valid_gstin(),
        "party_id": "V001",
        "party_state": "Maharashtra",
        "HSN": 9999,
        "taxable_value": 100.0,
        "tax_rate_pct": 18,
        "CGST": 9.0,
        "SGST": 9.0,
        "IGST": 0.0,
        "total": 118.0,
        "is_new_vendor": False,
    }])


def _rates() -> pd.DataFrame:
    return pd.DataFrame([{
        "HSN": 9999,
        "standard_tax_rate_pct": 18,
        "valid_from": "2017-07-01",
        "valid_to": None,
        "description": "test rate",
    }])


def test_gstin_checksum_validation_accepts_and_rejects():
    valid = _valid_gstin()
    assert len(valid) == 15
    assert valid[13] == "Z"
    assert is_valid_gstin(valid)
    assert not is_valid_gstin(valid[:-1] + ("0" if valid[-1] != "0" else "1"))


def test_tax_verifier_flags_split_separately_and_not_clean_record():
    clean_issues = verify_tax(_invoice(), _rates())
    assert clean_issues == []
    bad_split = _invoice()
    bad_split.loc[0, ["CGST", "SGST", "IGST"]] = [0.0, 0.0, 18.0]
    issues = verify_tax(bad_split, _rates())
    assert [item.issue_type for item in issues] == ["wrong_tax_split"]


def test_tax_verifier_emits_invalid_gstin_issue():
    row = _invoice()
    row.loc[0, "GSTIN"] = "27ABCDE1234F1Z00"
    issues = verify_tax(row, _rates())
    assert "invalid_gstin" in {item.issue_type for item in issues}


def test_duplicate_detector_finds_suffix_near_duplicate():
    rows = pd.concat([
        _invoice(),
        _invoice().assign(record_id="DUP1", invoice_no="INV-001A", date="2026-01-03"),
    ], ignore_index=True)
    issues = detect_duplicates(rows)
    assert len(issues) == 1
    assert issues[0].issue_type == "duplicate_near"
    assert issues[0].rupee_impact == float(Decimal("118.00"))


def test_missing_payment_detector_exempts_recent_open_items():
    invoices = pd.DataFrame([
        {"record_id": "OLD", "invoice_no": "INV-OLD", "document_type": "sale",
         "date": "2026-08-01", "vendor_customer_name": "Acme Ltd", "total": 100.0},
        {"record_id": "OPEN", "invoice_no": "INV-OPEN", "document_type": "sale",
         "date": "2026-09-01", "vendor_customer_name": "Acme Ltd", "total": 200.0},
    ])
    bank = pd.DataFrame(columns=["txn_id", "date", "amount", "narration", "reference", "tds_withheld"])
    ledger = pd.DataFrame(columns=["entry_id", "date", "reference", "debit", "credit"])
    filings = pd.DataFrame(columns=["filing_id", "invoice_no", "supplier_GSTIN", "appears_in_supplier_filing"])
    issues = detect_missing_records(invoices, bank, ledger, filings)
    missing_payment_ids = {
        item.record_ids[0] for item in issues if item.issue_type == "missing_payment"
    }
    assert missing_payment_ids == {"OLD"}


def test_mismatch_detector_flags_payment_preceding_invoice_date():
    invoice = _invoice()
    bank = pd.DataFrame([{
        "txn_id": "TXN1",
        "date": "2025-12-28",
        "amount": 118.0,
        "tds_withheld": 0.0,
        "reference": "INV-001",
        "narration": "Acme payment INV-001",
    }])
    ledger = pd.DataFrame(columns=["entry_id", "date", "reference", "debit", "credit"])
    filings = pd.DataFrame(columns=[
        "filing_id", "invoice_no", "invoice_date", "supplier_GSTIN", "supplier_name",
        "tax_amount", "filing_period",
    ])

    issues = detect_mismatches(invoice, bank, ledger, filings)

    date_issues = [item for item in issues if item.issue_type == "payment_before_invoice"]
    assert len(date_issues) == 1
    assert date_issues[0].record_ids == ["INVREC1", "TXN1"]
