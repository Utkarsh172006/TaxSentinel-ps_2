from __future__ import annotations

from decimal import InvalidOperation
from typing import Any

import pandas as pd

from detect.rules import money, normalized_name
from ingest.gstin import state_from_gstin, validate_gstin
from matching.features import normalize_identifier
from ingest.schema_map import TABLE_SCHEMAS

MONEY_FIELDS = {
    "taxable_value", "CGST", "SGST", "IGST", "total", "amount", "tds_withheld",
    "debit", "credit", "tax_amount",
}
DATE_FIELDS = {"date", "invoice_date", "valid_from", "valid_to"}
ID_FIELDS = {"invoice_no", "reference"}
GENERATED_IDS = {
    "invoices": ("record_id", "UPINV"),
    "bank_transactions": ("txn_id", "UPTXN"),
    "ledger": ("entry_id", "UPLED"),
    "supplier_filings": ("filing_id", "UPFIL"),
}


def _empty(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def standardize_table(
    frame: pd.DataFrame,
    table_type: str,
    mapping: dict[str, str | None],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Map and standardize uploaded fields while retaining raw values and change logs."""
    if table_type not in TABLE_SCHEMAS:
        raise ValueError(f"Unsupported table type: {table_type}")
    output = frame.copy()
    log: list[dict[str, Any]] = []
    for canonical, source in mapping.items():
        if source is None:
            continue
        if source not in frame.columns:
            raise ValueError(f"Mapped source column {source!r} does not exist.")
        output[f"{canonical}_raw"] = frame[source]
        output[canonical] = frame[source]

    for field in ID_FIELDS:
        if field in output.columns:
            before = output[field].tolist()
            after = [normalize_identifier(value) if not _empty(value) else value for value in before]
            output[field] = after
            _log_changes(log, table_type, field, before, after)

    for field in DATE_FIELDS:
        if field in output.columns:
            before = output[field].tolist()
            parsed = pd.to_datetime(output[field], errors="coerce")
            after = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None).tolist()
            output[field] = after
            _log_changes(log, table_type, field, before, after)

    for field in MONEY_FIELDS:
        if field in output.columns:
            before = output[field].tolist()
            after = []
            for value in before:
                if _empty(value):
                    after.append(None)
                    continue
                try:
                    after.append(float(money(str(value).replace(",", "").replace("₹", ""))))
                except (InvalidOperation, TypeError, ValueError):
                    after.append(None)
            output[field] = after
            _log_changes(log, table_type, field, before, after)

    gstin_field = "GSTIN" if "GSTIN" in output.columns else (
        "supplier_GSTIN" if "supplier_GSTIN" in output.columns else None
    )
    if gstin_field:
        raw_field = f"{gstin_field}_raw"
        output[raw_field] = output.get(raw_field, output[gstin_field])
        before = output[gstin_field].tolist()
        after = [str(value).strip().upper() if not _empty(value) else None for value in before]
        output[gstin_field] = after
        _log_changes(log, table_type, gstin_field, before, after)
        output["gstin_valid"] = [validate_gstin(value) for value in after]
        if gstin_field == "GSTIN" and "party_state" in output.columns:
            states = output["party_state"].tolist()
            derived_states = [state_from_gstin(value) for value in after]
            output["party_state"] = [
                state if not _empty(state) else derived_state
                for state, derived_state in zip(states, derived_states)
            ]
        elif gstin_field == "GSTIN" and any(state_from_gstin(value) is not None for value in after):
            output["party_state"] = [state_from_gstin(value) for value in after]

    for field in ("vendor_customer_name", "supplier_name"):
        if field in output.columns:
            before = output[field].tolist()
            after = [normalized_name(value) if not _empty(value) else None for value in before]
            output[field] = after
            _log_changes(log, table_type, field, before, after)

    if table_type in GENERATED_IDS:
        key, prefix = GENERATED_IDS[table_type]
        if key not in output.columns:
            output[key] = [f"{prefix}{index:06d}" for index in range(1, len(output) + 1)]
            for index, value in enumerate(output[key], start=1):
                log.append({
                    "table": table_type, "row": index, "column": key,
                    "before": None, "after": value,
                })
    return output, log


def _log_changes(
    log: list[dict[str, Any]],
    table_type: str,
    field: str,
    before: list[Any],
    after: list[Any],
) -> None:
    for row_number, (old, new) in enumerate(zip(before, after), start=1):
        old_text = None if _empty(old) else str(old)
        new_text = None if _empty(new) else str(new)
        if old_text != new_text:
            log.append({
                "table": table_type,
                "row": row_number,
                "column": field,
                "before": old_text,
                "after": new_text,
            })


def validation_report(frame: pd.DataFrame, table_type: str) -> dict[str, Any]:
    required = {
        "invoices": {"invoice_no", "date", "vendor_customer_name", "GSTIN", "taxable_value", "total"},
        "bank_transactions": {"date", "amount", "narration", "reference"},
        "ledger": {"date", "account", "debit", "credit", "reference"},
        "supplier_filings": {"supplier_GSTIN", "invoice_no", "invoice_date", "tax_amount", "filing_period"},
        "tax_rates": {"HSN", "standard_tax_rate_pct", "valid_from"},
    }
    needed = required.get(table_type, set())
    missing = sorted(needed - set(frame.columns))
    issues = []
    for field in DATE_FIELDS & set(frame.columns):
        values = pd.to_datetime(frame[field], errors="coerce")
        invalid_count = int(values.isna().sum())
        if invalid_count:
            issues.append(f"{field}: {invalid_count} missing or invalid dates")
    if "GSTIN" in frame.columns:
        invalid = int((~frame.gstin_valid.astype(bool)).sum()) if "gstin_valid" in frame.columns else 0
        if invalid:
            issues.append(f"GSTIN: {invalid} structurally/checksum-invalid values")
    if missing:
        issues.append(f"Missing required fields: {', '.join(missing)}")
    return {
        "table_type": table_type,
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing_required": missing,
        "issues": issues,
        "ready": not missing and len(frame) > 0,
    }
