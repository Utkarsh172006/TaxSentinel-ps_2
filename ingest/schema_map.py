from __future__ import annotations

from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

TABLE_SCHEMAS: dict[str, dict[str, tuple[str, ...]]] = {
    "invoices": {
        "record_id": ("record id", "invoice record id", "row id"),
        "invoice_no": ("invoice number", "invoice no", "bill number", "document number"),
        "document_type": ("document type", "type", "purchase sale"),
        "date": ("invoice date", "document date", "date"),
        "vendor_customer_name": ("vendor", "supplier name", "party name", "customer name"),
        "GSTIN": ("gstin", "supplier gstin", "tax id"),
        "party_state": ("state", "vendor state", "place of supply"),
        "HSN": ("hsn", "hsn code", "sac"),
        "taxable_value": ("taxable value", "taxable amount", "base amount"),
        "tax_rate_pct": ("tax rate", "gst rate", "rate percent"),
        "CGST": ("cgst", "central gst"),
        "SGST": ("sgst", "state gst"),
        "IGST": ("igst", "integrated gst"),
        "total": ("invoice total", "gross amount", "total amount", "total"),
        "is_new_vendor": ("new vendor", "is new vendor"),
    },
    "bank_transactions": {
        "txn_id": ("transaction id", "txn id", "bank row id"),
        "date": ("transaction date", "payment date", "date"),
        "amount": ("amount", "transaction amount", "payment amount"),
        "narration": ("narration", "description", "details", "memo"),
        "reference": ("reference", "ref", "cheque number", "invoice reference"),
        "tds_withheld": ("tds", "tds withheld", "withholding"),
    },
    "ledger": {
        "entry_id": ("entry id", "ledger id", "journal id"),
        "date": ("posting date", "entry date", "date"),
        "account": ("account", "ledger account"),
        "debit": ("debit", "debit amount"),
        "credit": ("credit", "credit amount"),
        "reference": ("reference", "invoice reference", "document number"),
    },
    "supplier_filings": {
        "filing_id": ("filing id", "row id", "filing record id"),
        "supplier_GSTIN": ("supplier gstin", "gstin"),
        "supplier_name": ("supplier name", "vendor name", "legal name"),
        "invoice_no": ("invoice number", "invoice no", "document number"),
        "invoice_date": ("invoice date", "document date"),
        "taxable_value": ("taxable value", "taxable amount"),
        "tax_amount": ("tax amount", "gst amount", "total tax"),
        "appears_in_supplier_filing": ("appears in filing", "filed", "in gstr2b"),
        "filing_period": ("filing period", "return period", "tax period"),
    },
    "tax_rates": {
        "HSN": ("hsn", "hsn code", "sac"),
        "standard_tax_rate_pct": ("tax rate", "standard rate", "gst rate", "rate percent"),
        "valid_from": ("valid from", "effective from", "start date"),
        "valid_to": ("valid to", "effective to", "end date"),
    },
}


def _norm_header(value: Any) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def detect_table_type(frame: pd.DataFrame) -> tuple[str, dict[str, int]]:
    """Choose a likely business table using header overlap."""
    headers = {_norm_header(column) for column in frame.columns}
    scores = {}
    for table_type, schema in TABLE_SCHEMAS.items():
        possible = {
            _norm_header(alias)
            for aliases in schema.values()
            for alias in aliases
        }
        scores[table_type] = sum(
            any(fuzz.ratio(header, candidate) >= 86 for candidate in possible)
            for header in headers
        )
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        raise ValueError("Could not recognize this table type from its column headers.")
    return best, scores


def suggest_mapping(frame: pd.DataFrame, table_type: str) -> dict[str, str | None]:
    """Suggest canonical-to-source header mappings with fuzzy alias matching."""
    if table_type not in TABLE_SCHEMAS:
        raise ValueError(f"Unsupported table type: {table_type}")
    mapping: dict[str, str | None] = {}
    source_headers = list(frame.columns)
    for canonical, aliases in TABLE_SCHEMAS[table_type].items():
        candidates = [canonical, *aliases]
        best: tuple[int, str] | None = None
        for alias in candidates:
            result = process.extractOne(alias, source_headers, scorer=fuzz.WRatio)
            if result is not None and (best is None or result[1] > best[0]):
                best = (int(result[1]), str(result[0]))
        mapping[canonical] = best[1] if best and best[0] >= 72 else None
    return mapping
