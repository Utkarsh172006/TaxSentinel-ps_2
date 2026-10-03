from __future__ import annotations

from datetime import date
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

PAIR_FEATURES = (
    "invoice_id_similarity",
    "reference_similarity",
    "invoice_id_edit_distance",
    "normalized_id_equal",
    "vendor_name_similarity",
    "amount_abs_diff",
    "amount_relative_diff",
    "date_gap_days",
    "reference_contains_invoice",
    "direction_matches",
    "tds_amount",
    "same_gstin",
    "gstin_comparable",
)


def normalize_identifier(value: Any) -> str:
    """Normalize invoice identifiers while tolerating common OCR substitutions."""
    if value is None:
        return ""
    text = "".join(character for character in str(value).upper() if character.isalnum())
    return text.translate(str.maketrans({"O": "0", "I": "1"}))


def normalize_vendor_name(value: Any) -> str:
    """Normalize party names and remove common Indian legal suffixes."""
    if value is None:
        return ""
    import re

    text = str(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    words = text.split()
    suffixes = {"LTD", "LIMITED", "PVT", "PRIVATE", "LLP", "PLC", "INC"}
    while words and words[-1] in suffixes:
        words.pop()
    return " ".join(words)


def _as_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def pair_features(invoice: dict[str, Any], payment: dict[str, Any]) -> dict[str, float]:
    """Build deterministic invoice/payment features from detector-visible fields."""
    invoice_id = normalize_identifier(invoice.get("invoice_no"))
    reference = normalize_identifier(payment.get("reference"))
    narration = normalize_identifier(payment.get("narration"))
    text = f"{reference} {narration}"

    amount = abs(float(payment.get("amount", 0) or 0))
    tds = abs(float(payment.get("tds_withheld", 0) or 0))
    total = abs(float(invoice.get("total", 0) or 0))
    amount_diff = abs(total - amount - tds)
    denominator = max(total, 1.0)

    invoice_date = _as_date(invoice.get("date"))
    payment_date = _as_date(payment.get("date"))
    date_gap = abs((payment_date - invoice_date).days) if invoice_date and payment_date else 9999
    document_type = str(invoice.get("document_type", "purchase")).lower()
    direction = float((document_type == "purchase" and float(payment.get("amount", 0) or 0) >= 0)
                      or (document_type != "purchase" and float(payment.get("amount", 0) or 0) <= 0))
    invoice_gstin = str(invoice.get("GSTIN", "") or "").upper()
    payment_gstin = str(payment.get("GSTIN", payment.get("supplier_GSTIN", "")) or "").upper()
    gstin_comparable = bool(invoice_gstin and payment_gstin)

    return {
        "invoice_id_similarity": max(
            fuzz.ratio(invoice_id, normalize_identifier(payment.get("reference"))),
            fuzz.partial_ratio(invoice_id, text),
        ) / 100.0 if invoice_id else 0.0,
        "reference_similarity": fuzz.ratio(invoice_id, reference) / 100.0 if invoice_id else 0.0,
        "invoice_id_edit_distance": float(Levenshtein.distance(invoice_id, reference)) if invoice_id and reference else 999.0,
        "normalized_id_equal": float(bool(invoice_id and invoice_id == reference)),
        "vendor_name_similarity": fuzz.token_set_ratio(
            normalize_vendor_name(invoice.get("vendor_customer_name")),
            normalize_vendor_name(payment.get("narration")),
        ) / 100.0,
        "amount_abs_diff": amount_diff,
        "amount_relative_diff": amount_diff / denominator,
        "date_gap_days": float(date_gap),
        "reference_contains_invoice": float(bool(invoice_id and invoice_id in text)),
        "direction_matches": direction,
        "tds_amount": tds,
        "same_gstin": float(gstin_comparable and invoice_gstin == payment_gstin),
        "gstin_comparable": float(gstin_comparable),
    }
