from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

from ingest.ocr import extract_pdf_text


def read_uploaded_table(
    filename: str,
    payload: bytes,
    enable_ocr: bool = False,
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame | None, str | None]:
    """Read a CSV/Excel table; PDFs require the explicit OCR switch."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(payload)), None
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(payload), sheet_name=sheet_name), None
    if suffix == ".pdf":
        text = extract_pdf_text(payload, enable_ocr=enable_ocr)
        return None, text
    raise ValueError(f"Unsupported upload type: {suffix or 'unknown'}")


def mapping_validation_errors(
    table_type: str,
    mapping: dict[str, str | None],
    report: dict[str, Any],
) -> list[str]:
    errors = list(report["issues"])
    missing_mappings = [
        column for column in report["missing_required"]
        if not mapping.get(column)
    ]
    if missing_mappings:
        errors.append(f"Map the required fields: {', '.join(missing_mappings)}")
    return errors
