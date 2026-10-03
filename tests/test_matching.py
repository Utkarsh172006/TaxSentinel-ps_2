from __future__ import annotations

import pandas as pd
import pytest

from matching.blocking import attach_training_labels, generate_pair_candidates
from matching.features import normalize_identifier, normalize_vendor_name
from matching.pipeline import reconcile
from matching.subset_sum import find_bounded_subset


def test_normalization_handles_ocr_variants_and_company_suffixes():
    assert normalize_identifier(" inv-O1-42 ") == "1NV0142"
    assert normalize_vendor_name("ACME Pvt. Limited") == "ACME"


def test_bounded_subset_sum_finds_two_or_more_installments():
    selected = find_bounded_subset(
        [6000, 4000, 1700],
        10000,
        min_items=2,
        max_items=3,
        tolerance_paise=0,
    )
    assert selected is not None
    assert len(selected) == 2
    assert sum([6000, 4000, 1700][index] for index in selected) == 10000


def test_candidate_features_are_separate_from_training_labels():
    invoices = pd.DataFrame([{
        "record_id": "INVREC1",
        "invoice_no": "INV-001",
        "date": "2026-01-01",
        "vendor_customer_name": "Acme Private Limited",
        "total": 100.0,
        "document_type": "purchase",
    }])
    payments = pd.DataFrame([{
        "txn_id": "TXN1",
        "date": "2026-01-05",
        "amount": 100.0,
        "narration": "Payment to Acme Pvt Ltd INV-001",
        "reference": "INV-001",
        "tds_withheld": 0.0,
    }])
    candidates = generate_pair_candidates(invoices, payments)
    assert len(candidates) == 1
    assert "label" not in candidates.columns
    links = pd.DataFrame([{
        "source_table": "bank_transactions",
        "source_id": "TXN1",
        "invoice_record_id": "INVREC1",
    }])
    labeled = attach_training_labels(candidates, links)
    assert labeled.label.tolist() == [1]


def test_pipeline_matches_exact_split_and_bulk_without_truth_columns():
    invoices = pd.DataFrame([
        {
            "record_id": "INVREC1", "invoice_no": "INV-001", "date": "2026-01-01",
            "vendor_customer_name": "Acme Private Limited", "GSTIN": "27AAAAA0000A1Z0",
            "total": 100.0, "document_type": "purchase",
        },
        {
            "record_id": "INVREC2", "invoice_no": "INV-002", "date": "2026-01-02",
            "vendor_customer_name": "Beta Limited", "GSTIN": "27BBBBB0000B1Z0",
            "total": 100.0, "document_type": "purchase",
        },
        {
            "record_id": "INVREC3", "invoice_no": "INV-003", "date": "2026-01-03",
            "vendor_customer_name": "Beta Limited", "GSTIN": "27BBBBB0000B1Z0",
            "total": 100.0, "document_type": "purchase",
        },
        {
            "record_id": "INVREC4", "invoice_no": "INV-004", "date": "2026-01-04",
            "vendor_customer_name": "Beta Limited", "GSTIN": "27BBBBB0000B1Z0",
            "total": 200.0, "document_type": "purchase",
        },
    ])
    payments = pd.DataFrame([
        {
            "txn_id": "TXN1", "date": "2026-01-03", "amount": 100.0,
            "narration": "Payment to Acme Pvt Ltd INV-001", "reference": "INV-001",
            "tds_withheld": 0.0,
        },
        {
            "txn_id": "TXN2", "date": "2026-01-04", "amount": 60.0,
            "narration": "Payment to Acme Pvt Ltd INV-002 part 1", "reference": "INV-002",
            "tds_withheld": 0.0,
        },
        {
            "txn_id": "TXN3", "date": "2026-01-08", "amount": 40.0,
            "narration": "Payment to Acme Pvt Ltd INV-002 part 2", "reference": "INV-002",
            "tds_withheld": 0.0,
        },
        {
            "txn_id": "TXN4", "date": "2026-01-10", "amount": 300.0,
            "narration": "Bulk payment to Beta Limited INV-003 INV-004", "reference": "BULK1",
            "tds_withheld": 0.0,
        },
    ])
    # The example has one exact match, a clean split and a separate bulk pair.
    result = reconcile(invoices, payments)
    assert any(match.tier == "exact" and match.invoice_ids == ("INVREC1",) for match in result.matches)
    assert any(match.tier == "many_to_many" and len(match.payment_ids) == 2 for match in result.matches)
    assert any(match.tier == "many_to_many" and len(match.invoice_ids) == 2 for match in result.matches)


def test_reconciliation_fails_closed_on_link_columns():
    invoices = pd.DataFrame([{
        "record_id": "INVREC1", "invoice_no": "INV-1", "date": "2026-01-01",
        "vendor_customer_name": "Acme Ltd", "total": 100.0,
        "linked_invoice_record_id": "INVREC1",
    }])
    payments = pd.DataFrame(columns=["txn_id", "date", "amount", "narration", "reference"])
    with pytest.raises(ValueError, match="link columns"):
        reconcile(invoices, payments)
