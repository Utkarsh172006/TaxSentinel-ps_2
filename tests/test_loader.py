from pathlib import Path

import pandas as pd
import pytest

from ingest.loader import assert_inference_safe, lint_dataset, strip_forbidden_columns


def test_strip_forbidden_columns_removes_link_columns():
    frame = pd.DataFrame({
        "invoice_no": ["INV-1"],
        "linked_invoice_record_id": ["INVREC1"],
        "linked_payment_id": ["TXN1"],
        "amount": [123.45],
    })
    cleaned = strip_forbidden_columns(frame)
    assert "linked_invoice_record_id" not in cleaned.columns
    assert "linked_payment_id" not in cleaned.columns
    assert list(cleaned.columns) == ["invoice_no", "amount"]


@pytest.mark.parametrize(
    "tables",
    [
        {"invoices.csv": pd.DataFrame({"invoice_no": ["INV-1"], "linked_invoice_record_id": ["INVREC1"]})},
        {"invoices.csv": pd.DataFrame({"invoice_no": ["INV-1"], "error_type": ["wrong_tax_rate"]})},
        {"injected_errors.csv": pd.DataFrame({"error_type": ["invalid_gstin"]})},
        {"match_challenges.csv": pd.DataFrame({"challenge_type": ["split_payment"]})},
        {"links.csv": pd.DataFrame({"invoice_record_id": ["INVREC1"]})},
    ],
)
def test_inference_rejects_ground_truth(tables):
    with pytest.raises(ValueError, match="forbidden"):
        assert_inference_safe(tables)


def test_lint_dataset_flags_forbidden_columns(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = pd.DataFrame({"invoice_no": ["INV-1"], "linked_invoice_record_id": ["INVREC1"]})
    df.to_csv(raw_dir / "sample.csv", index=False)
    findings = lint_dataset(raw_dir)
    assert "sample.csv" in findings


def test_lint_dataset_flags_answer_key_filename(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({"error_type": ["wrong_tax_rate"]}).to_csv(raw_dir / "injected_errors.csv", index=False)
    assert "injected_errors.csv" in lint_dataset(raw_dir)
