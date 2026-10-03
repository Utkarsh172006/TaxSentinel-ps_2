from __future__ import annotations

from typing import Any

import pandas as pd

from matching.blocking import generate_pair_candidates
from matching.exact import Match, match_band
from matching.features import PAIR_FEATURES


def fuzzy_matches(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    classifier: Any | None = None,
    minimum_confidence: float = 0.60,
    candidates: pd.DataFrame | None = None,
) -> list[Match]:
    """Score blocked invoice/payment pairs and return reviewable match suggestions."""
    candidates = candidates if candidates is not None else generate_pair_candidates(invoices, payments)
    if candidates.empty:
        return []
    features = candidates.loc[:, PAIR_FEATURES]
    if classifier is not None:
        probabilities = classifier.predict_proba(features)[:, 1]
    else:
        # Deterministic fallback score for offline/first-run operation.
        probabilities = (
            0.30 * candidates.invoice_id_similarity
            + 0.22 * candidates.vendor_name_similarity
            + 0.18 * candidates.reference_similarity
            + 0.18 * (candidates.amount_relative_diff <= 0.01).astype(float)
            + 0.12 * (candidates.date_gap_days <= 45).astype(float)
        ).to_numpy()

    candidates = candidates.assign(confidence=probabilities)
    candidates = candidates[candidates.confidence >= minimum_confidence]
    candidates = candidates.sort_values("confidence", ascending=False)
    used_invoices: set[str] = set()
    used_payments: set[str] = set()
    results: list[Match] = []

    for row in candidates.itertuples(index=False):
        invoice_id, payment_id = str(row.invoice_record_id), str(row.payment_id)
        if invoice_id in used_invoices or payment_id in used_payments:
            continue
        confidence = float(row.confidence)
        results.append(Match(
            invoice_ids=(invoice_id,),
            payment_ids=(payment_id,),
            tier="fuzzy",
            confidence=confidence,
            band=match_band(confidence),
            features={feature: float(getattr(row, feature)) for feature in PAIR_FEATURES},
        ))
        used_invoices.add(invoice_id)
        used_payments.add(payment_id)
    return results
