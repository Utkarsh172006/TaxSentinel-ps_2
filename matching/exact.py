from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matching.features import normalize_identifier


@dataclass(frozen=True)
class Match:
    invoice_ids: tuple[str, ...]
    payment_ids: tuple[str, ...]
    tier: str
    confidence: float
    band: str
    features: dict[str, float]


def match_band(confidence: float) -> str:
    if confidence >= 0.90:
        return "auto"
    if confidence >= 0.60:
        return "review"
    return "unmatched"


def exact_match(invoice: dict[str, Any], payment: dict[str, Any], tolerance: float = 1.0) -> Match | None:
    """Return a Tier 1 match when normalized reference and amount agree."""
    invoice_id = normalize_identifier(invoice.get("invoice_no"))
    reference = normalize_identifier(payment.get("reference"))
    amount = abs(float(payment.get("amount", 0) or 0)) + abs(float(payment.get("tds_withheld", 0) or 0))
    total = abs(float(invoice.get("total", 0) or 0))
    if not invoice_id or invoice_id != reference or abs(total - amount) > tolerance:
        return None
    return Match(
        invoice_ids=(str(invoice["record_id"]),),
        payment_ids=(str(payment["txn_id"]),),
        tier="exact",
        confidence=0.99,
        band="auto",
        features={"normalized_reference_equal": 1.0, "amount_within_tolerance": 1.0},
    )
