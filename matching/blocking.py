from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from matching.features import normalize_identifier, normalize_vendor_name, pair_features


def generate_pair_candidates(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    date_window_days: int = 45,
    vendor_threshold: int = 55,
    max_per_invoice: int = 25,
) -> pd.DataFrame:
    """Generate a bounded candidate set from visible party, date and reference data."""
    if invoices.empty or payments.empty:
        return pd.DataFrame(columns=["invoice_record_id", "payment_id"])
    required_invoice = {"record_id", "invoice_no", "date", "vendor_customer_name"}
    required_payment = {"txn_id", "date", "narration", "reference", "amount"}
    if missing := required_invoice - set(invoices.columns):
        raise ValueError(f"Invoice table is missing required columns: {sorted(missing)}")
    if missing := required_payment - set(payments.columns):
        raise ValueError(f"Payment table is missing required columns: {sorted(missing)}")

    invoices = invoices.reset_index(drop=True)
    payments = payments.reset_index(drop=True)
    invoice_rows = invoices.to_dict("records")
    payment_rows = payments.to_dict("records")
    payment_dates = pd.to_datetime(payments["date"], errors="coerce")
    ordered = payment_dates.sort_values(kind="stable")
    sorted_positions = ordered.index.to_list()
    sorted_dates = ordered.to_numpy()
    payment_texts = [
        f"{normalize_identifier(row.get('reference'))} {normalize_identifier(row.get('narration'))}"
        for row in payment_rows
    ]
    payment_references = [normalize_identifier(row.get("reference")) for row in payment_rows]
    candidates: list[dict[str, Any]] = []

    for invoice in invoice_rows:
        invoice_date = pd.to_datetime(invoice.get("date"), errors="coerce")
        if pd.isna(invoice_date):
            continue
        low = (invoice_date - timedelta(days=date_window_days)).to_datetime64()
        high = (invoice_date + timedelta(days=date_window_days)).to_datetime64()
        left = int(sorted_dates.searchsorted(low, side="left"))
        right = int(sorted_dates.searchsorted(high, side="right"))
        invoice_key = normalize_identifier(invoice.get("invoice_no"))
        vendor_key = normalize_vendor_name(invoice.get("vendor_customer_name"))
        ranked: list[tuple[float, int, bool]] = []

        position_set = set(sorted_positions[left:right])
        if invoice_key:
            position_set.update(
                position for position, text in enumerate(payment_texts)
                if payment_references[position] == invoice_key or invoice_key in text
            )
        for payment_position in sorted(position_set):
            payment = payment_rows[payment_position]
            direct_reference = bool(
                invoice_key
                and (
                    payment_references[payment_position] == invoice_key
                    or invoice_key in payment_texts[payment_position]
                )
            )
            id_score = 1.0 if direct_reference else (
                max(
                    fuzz.ratio(invoice_key, payment_references[payment_position]),
                    fuzz.partial_ratio(invoice_key, payment_texts[payment_position]),
                ) / 100.0 if invoice_key else 0.0
            )
            vendor_score = 0.0 if direct_reference else (
                fuzz.token_set_ratio(
                    vendor_key,
                    normalize_vendor_name(payment.get("narration")),
                ) / 100.0 if vendor_key else 0.0
            )
            # Strong reference evidence bypasses a weak vendor-name block.
            if vendor_score * 100 < vendor_threshold and id_score < 0.88:
                continue
            payment_date = payment_dates.iloc[payment_position]
            date_gap = abs((payment_date - invoice_date).days) if not pd.isna(payment_date) else 9999
            total = abs(float(invoice.get("total", 0) or 0))
            amount = abs(float(payment.get("amount", 0) or 0))
            tds = abs(float(payment.get("tds_withheld", 0) or 0))
            amount_relative_diff = abs(total - amount - tds) / max(total, 1.0)
            date_score = max(0.0, 1.0 - date_gap / max(date_window_days, 1))
            amount_score = 1.0 / (1.0 + amount_relative_diff)
            score = 0.45 * id_score + 0.35 * vendor_score + 0.15 * amount_score + 0.05 * date_score
            ranked.append((score, payment_position, direct_reference))

        ranked.sort(key=lambda item: item[0], reverse=True)
        retained = ranked[:max_per_invoice]
        # Never discard directly reference-linked candidates due to a busy vendor.
        retained.extend(item for item in ranked[max_per_invoice:] if item[2])
        for _, position, _ in retained:
            payment = payment_rows[position]
            feature_values = pair_features(invoice, payment)
            candidates.append({
                "invoice_record_id": str(invoice["record_id"]),
                "payment_id": str(payment["txn_id"]),
                **feature_values,
            })

    if not candidates:
        return pd.DataFrame(columns=["invoice_record_id", "payment_id"])
    return pd.DataFrame(candidates).drop_duplicates(["invoice_record_id", "payment_id"])


def attach_training_labels(candidates: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Join training-only link labels to candidates after candidate features exist."""
    if candidates.empty:
        return candidates.assign(label=pd.Series(dtype="int64"))
    bank_links = links[links["source_table"].eq("bank_transactions")]
    positive_pairs: set[tuple[str, str]] = set()
    for row in bank_links.itertuples(index=False):
        if pd.isna(row.invoice_record_id):
            continue
        for invoice_id in str(row.invoice_record_id).split("|"):
            positive_pairs.add((invoice_id, str(row.source_id)))
    result = candidates.copy()
    result["label"] = [
        int((invoice_id, payment_id) in positive_pairs)
        for invoice_id, payment_id in zip(result.invoice_record_id, result.payment_id)
    ]
    return result
