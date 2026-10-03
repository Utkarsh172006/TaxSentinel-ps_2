from __future__ import annotations

from datetime import date

import pandas as pd

from detect.common import issue
from detect.rules import gstin_checksum
from liability.liability import calculate_liability, estimate_interest_and_penalty


def _valid_gstin() -> str:
    prefix = "27ABCDE1234F1Z"
    return prefix + gstin_checksum(prefix)


def _tables():
    gstin = _valid_gstin()
    invoices = pd.DataFrame([
        {
            "record_id": "SALE1", "invoice_no": "SALE-1", "date": "2026-01-10",
            "document_type": "sale", "vendor_customer_name": "Customer Ltd",
            "HSN": 9999, "taxable_value": 100.0, "CGST": 9.0, "SGST": 9.0,
            "IGST": 0.0, "total": 118.0,
        },
        {
            "record_id": "BUY1", "invoice_no": "BUY-1", "date": "2026-01-12",
            "document_type": "purchase", "vendor_customer_name": "Supplier Ltd",
            "HSN": 8888, "taxable_value": 100.0, "CGST": 9.0, "SGST": 9.0,
            "IGST": 0.0, "total": 118.0, "GSTIN": gstin,
        },
    ])
    filings = pd.DataFrame([{
        "filing_id": "FIL1", "invoice_no": "BUY-1", "supplier_GSTIN": gstin,
        "appears_in_supplier_filing": False, "tax_amount": 18.0,
    }])
    return invoices, filings


def test_liability_reports_before_after_and_itc_at_risk():
    invoices, filings = _tables()
    result = calculate_liability(invoices, filings)
    january = result["periods"][0]
    assert january["net_liability_before"] == 0.0
    assert january["net_liability_after"] == 18.0
    assert january["itc_at_risk"] == 18.0


def test_resolving_a_supplier_filing_issue_recomputes_liability():
    invoices, filings = _tables()
    flagged = issue(
        "supplier_not_filed", "High", "Supplier Ltd", ["BUY1", "FIL1"],
        "Filed", "Not filed", 18.0, {}, "Follow up",
    )
    result = calculate_liability(invoices, filings, issues=[flagged], resolved_issue_ids={flagged.issue_id})
    january = result["periods"][0]
    assert january["net_liability_after"] == 0.0
    assert january["itc_at_risk"] == 0.0


def test_rate_what_if_recomputes_output_tax():
    invoices, filings = _tables()
    result = calculate_liability(invoices, filings, rate_overrides={"9999": 5})
    assert result["periods"][0]["output_tax"] == 5.0


def test_interest_estimator_is_effective_config_driven():
    result = estimate_interest_and_penalty(10000, date(2026, 1, 1), date(2026, 1, 31))
    assert result["days"] == 30
    assert result["interest"] == 147.95
    assert "Verify applicable" in result["notice"]
