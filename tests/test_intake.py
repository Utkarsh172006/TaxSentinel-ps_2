from __future__ import annotations

import builtins

import pytest

from app.intake import read_uploaded_table


def test_csv_upload_works_without_pdfplumber(monkeypatch):
    original_import = builtins.__import__

    def import_without_pdfplumber(name, *args, **kwargs):
        if name == "pdfplumber":
            raise ModuleNotFoundError("No module named 'pdfplumber'", name="pdfplumber")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pdfplumber)

    frame, text = read_uploaded_table("records.csv", b"invoice_no,total\nINV-1,100\n")

    assert text is None
    assert frame is not None
    assert frame.loc[0, "invoice_no"] == "INV-1"


def test_pdf_upload_reports_optional_dependency_when_missing(monkeypatch):
    original_import = builtins.__import__

    def import_without_pdfplumber(name, *args, **kwargs):
        if name == "pdfplumber":
            raise ModuleNotFoundError("No module named 'pdfplumber'", name="pdfplumber")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pdfplumber)

    with pytest.raises(RuntimeError, match="PDF extraction requires pdfplumber"):
        read_uploaded_table("document.pdf", b"%PDF")
