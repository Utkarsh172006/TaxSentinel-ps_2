from __future__ import annotations

import pandas as pd
import pytest

from detect.common import issue
from ingest.clean import standardize_table, validation_report
from ingest.schema_map import detect_table_type, suggest_mapping
from reports.excel import build_excel_report
from reports.pdf import build_pdf_report
from review.store import (
    add_comment,
    initialize_store,
    list_audit_log,
    list_comments,
    list_feedback,
    save_feedback,
    sync_issues,
    transition_case,
)
from vendor_risk.score import score_vendors


def test_schema_mapping_and_standardization_retain_raw_values():
    source = pd.DataFrame([{
        "Invoice Number": " inv-o1 ",
        "Invoice Date": "2026-02-01",
        "Vendor": "Example Private Limited",
        "Supplier GSTIN": "27ABCDE1234F1Z5",
        "Taxable Amount": "1,000.00",
        "Invoice Total": "1,180.00",
    }])
    detected, _ = detect_table_type(source)
    assert detected == "invoices"
    mapping = suggest_mapping(source, detected)

    standardized, changes = standardize_table(source, detected, mapping)
    report = validation_report(standardized, detected)

    assert standardized.loc[0, "invoice_no_raw"] == " inv-o1 "
    assert standardized.loc[0, "invoice_no"] == "1NV01"
    assert standardized.loc[0, "taxable_value"] == 1000.0
    assert standardized.loc[0, "vendor_customer_name"] == "EXAMPLE"
    assert report["ready"]
    assert changes


def test_review_store_records_transition_comments_feedback_and_audit(tmp_path):
    path = tmp_path / "review.sqlite"
    item = issue("tax_mismatch", "High", "Supplier", ["INV1"], 100, 120, 20, {}, "Review")

    initialize_store(path)
    sync_issues([item], path)
    transition_case(item.issue_id, "Investigating", "Investigation started", path=path)
    transition_case(item.issue_id, "Explained", "Evidence reviewed", path=path)
    add_comment(item.issue_id, "Checked invoice copy.", "CA", path=path)
    save_feedback(item.issue_id, "accept", "Confirmed.", path=path)

    assert list_comments(item.issue_id, path)[0]["author"] == "CA"
    assert list_feedback(path)[0]["decision"] == "accept"
    assert [entry["action"] for entry in list_audit_log(path)] == [
        "Finding accepted", "Comment added", "Evidence reviewed", "Investigation started", "Detected",
    ]


def test_review_store_rejects_invalid_transition(tmp_path):
    path = tmp_path / "review.sqlite"
    item = issue("missing_payment", "High", "Supplier", ["INV1"], None, None, 0, {}, "Review")
    sync_issues([item], path)

    with pytest.raises(ValueError, match="Invalid case transition"):
        transition_case(item.issue_id, "Resolved", "Skipped workflow", path=path)


def test_vendor_risk_is_calculated_from_invoice_and_issue_severity():
    invoices = pd.DataFrame([{
        "record_id": "INV1",
        "vendor_customer_name": "Supplier",
        "document_type": "purchase",
        "total": 1000,
    }])
    finding = issue("tax_mismatch", "High", "Supplier", ["INV1"], 100, 120, 20, {}, "Review")

    result = score_vendors(invoices, [finding])

    assert result.loc[0, "risk_score"] == pytest.approx(100.0)
    assert result.loc[0, "weighted_issue_points"] == 3
    assert result.loc[0, "purchase_exposure"] == pytest.approx(1000.0)


def test_report_builders_return_real_downloadable_files():
    cases = [{"issue_id": "ISS-1", "issue_type": "tax_mismatch", "status": "Open"}]
    liability = {"periods": [], "totals": {"net_liability_after": 10}}
    vendors = pd.DataFrame([{"vendor": "Supplier", "risk_score": 25}])

    workbook = build_excel_report(cases, liability, vendors, [])
    pdf = build_pdf_report(cases, liability)

    assert workbook.startswith(b"PK")
    assert pdf.startswith(b"%PDF")
