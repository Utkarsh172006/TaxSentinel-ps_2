from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from ingest.loader import assert_inference_safe
from matching.blocking import generate_pair_candidates
from matching.exact import Match, exact_match
from matching.features import normalize_identifier, normalize_vendor_name, pair_features
from matching.fuzzy import fuzzy_matches
from matching.subset_sum import find_bounded_subset

TOLERANCE_RUPEES = 1.0


@dataclass
class ReconciliationResult:
    matches: list[Match] = field(default_factory=list)
    unmatched_invoice_ids: list[str] = field(default_factory=list)
    unmatched_payment_ids: list[str] = field(default_factory=list)
    ledger_links: dict[str, str] = field(default_factory=dict)
    filing_links: dict[str, str] = field(default_factory=dict)


def _cents(value: Any) -> int:
    return int(round(float(value or 0) * 100))


def _payment_value(row: dict[str, Any]) -> int:
    return abs(_cents(row.get("amount"))) + abs(_cents(row.get("tds_withheld")))


def _vendor_score(invoice: dict[str, Any], payment: dict[str, Any]) -> float:
    return fuzz.token_set_ratio(
        normalize_vendor_name(invoice.get("vendor_customer_name")),
        normalize_vendor_name(payment.get("narration")),
    ) / 100.0


def _invoice_mentions(invoice: dict[str, Any], payment: dict[str, Any]) -> bool:
    key = normalize_identifier(invoice.get("invoice_no"))
    if not key:
        return False
    reference = normalize_identifier(payment.get("reference"))
    narration = normalize_identifier(payment.get("narration"))
    return key == reference or key in narration


def _match_three_way(
    matches: list[Match],
    invoices: pd.DataFrame,
    ledger: pd.DataFrame | None,
    filings: pd.DataFrame | None,
) -> tuple[dict[str, str], dict[str, str]]:
    ledger_links: dict[str, str] = {}
    filing_links: dict[str, str] = {}
    if not matches:
        return ledger_links, filing_links

    invoice_lookup = invoices.set_index("record_id").to_dict("index")
    matched_invoice_ids = {invoice_id for match in matches for invoice_id in match.invoice_ids}
    by_reference: dict[str, list[str]] = {}
    for invoice_id in matched_invoice_ids:
        reference = normalize_identifier(invoice_lookup[invoice_id].get("invoice_no"))
        by_reference.setdefault(reference, []).append(invoice_id)

    if ledger is not None and not ledger.empty:
        for row in ledger.to_dict("records"):
            reference = normalize_identifier(row.get("reference"))
            amount = abs(float(row.get("debit", 0) or 0)) + abs(float(row.get("credit", 0) or 0))
            candidates = [
                invoice_id for invoice_id in by_reference.get(reference, [])
                if abs(abs(float(invoice_lookup[invoice_id].get("total", 0) or 0)) - amount) <= TOLERANCE_RUPEES
            ]
            if len(candidates) == 1:
                ledger_links[candidates[0]] = str(row["entry_id"])

    if filings is not None and not filings.empty:
        for row in filings.to_dict("records"):
            if not bool(row.get("appears_in_supplier_filing", True)):
                continue
            invoice_no = normalize_identifier(row.get("invoice_no"))
            gstin = str(row.get("supplier_GSTIN", "")).upper()
            candidates = [
                invoice_id for invoice_id in by_reference.get(invoice_no, [])
                if str(invoice_lookup[invoice_id].get("GSTIN", "")).upper() == gstin
            ]
            if len(candidates) == 1:
                filing_links[candidates[0]] = str(row["filing_id"])
    return ledger_links, filing_links


def reconcile(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    ledger: pd.DataFrame | None = None,
    filings: pd.DataFrame | None = None,
    classifier: Any | None = None,
) -> ReconciliationResult:
    """Run exact, bounded many-to-many and fuzzy matching without truth columns."""
    tables = {"invoices.csv": invoices, "bank_transactions.csv": payments}
    if ledger is not None:
        tables["ledger.csv"] = ledger
    if filings is not None:
        tables["supplier_filings.csv"] = filings
    assert_inference_safe(tables)

    inv_rows = invoices.reset_index(drop=True).to_dict("records")
    payment_rows = payments.reset_index(drop=True).to_dict("records")
    matches: list[Match] = []
    used_invoices: set[str] = set()
    used_payments: set[str] = set()
    payments_by_reference: dict[str, list[dict[str, Any]]] = {}
    for payment in payment_rows:
        reference_key = normalize_identifier(payment.get("reference"))
        payments_by_reference.setdefault(reference_key, []).append(payment)

    # Tier 1: exact normalized invoice reference plus full amount within tolerance.
    for invoice in inv_rows:
        exact_candidates = [
            payment for payment in payments_by_reference.get(
                normalize_identifier(invoice.get("invoice_no")), []
            )
            if str(payment["txn_id"]) not in used_payments
            and exact_match(invoice, payment, TOLERANCE_RUPEES) is not None
        ]
        if len(exact_candidates) == 1:
            match = exact_match(invoice, exact_candidates[0], TOLERANCE_RUPEES)
            if match is not None:
                matches.append(match)
                used_invoices.update(match.invoice_ids)
                used_payments.update(match.payment_ids)

    unresolved_invoices = [row for row in inv_rows if str(row["record_id"]) not in used_invoices]
    unresolved_payments = [row for row in payment_rows if str(row["txn_id"]) not in used_payments]
    unresolved_invoice_frame = invoices[~invoices.record_id.astype(str).isin(used_invoices)].reset_index(drop=True)
    unresolved_payment_frame = payments[~payments.txn_id.astype(str).isin(used_payments)].reset_index(drop=True)
    candidate_frame = generate_pair_candidates(unresolved_invoice_frame, unresolved_payment_frame)
    invoice_by_id = {str(row["record_id"]): row for row in unresolved_invoices}
    payment_by_id = {str(row["txn_id"]): row for row in unresolved_payments}
    candidates_by_invoice: dict[str, list[dict[str, Any]]] = {}
    candidates_by_payment: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_frame.to_dict("records"):
        candidates_by_invoice.setdefault(str(row["invoice_record_id"]), []).append(row)
        candidates_by_payment.setdefault(str(row["payment_id"]), []).append(row)

    # Tier 3: split invoices settled by a small set of installments.
    for invoice in unresolved_invoices:
        invoice_id = str(invoice["record_id"])
        pair_options = candidates_by_invoice.get(invoice_id, [])
        eligible_pairs = [
            row for row in pair_options
            if row["payment_id"] not in used_payments
            and (
                _invoice_mentions(invoice, payment_by_id[row["payment_id"]])
                or (
                    "PART" in normalize_identifier(payment_by_id[row["payment_id"]].get("narration"))
                    and row["vendor_name_similarity"] >= 0.70
                    and row["date_gap_days"] <= 45
                )
            )
        ]
        eligible_pairs.sort(
            key=lambda row: (
                row["reference_contains_invoice"] == 0,
                row["date_gap_days"],
            )
        )
        candidates = [payment_by_id[row["payment_id"]] for row in eligible_pairs[:18]]
        subset = find_bounded_subset(
            [_payment_value(row) for row in candidates],
            _cents(invoice.get("total")),
            min_items=2,
            max_items=3,
            tolerance_paise=100,
            max_nodes=10000,
        )
        if subset is not None:
            selected = [candidates[position] for position in subset]
            payment_ids = tuple(str(row["txn_id"]) for row in selected)
            matches.append(Match(
                invoice_ids=(invoice_id,),
                payment_ids=payment_ids,
                tier="many_to_many",
                confidence=0.97,
                band="auto",
                features={"payment_count": float(len(selected)), "amount_tolerance_paise": 100.0},
            ))
            used_invoices.add(invoice_id)
            used_payments.update(payment_ids)

    # Tier 3: one bulk payment covering 2-6 invoices.
    for payment in unresolved_payments:
        payment_id = str(payment["txn_id"])
        if payment_id in used_payments:
            continue
        eligible_pairs = [
            row for row in candidates_by_payment.get(payment_id, [])
            if row["invoice_record_id"] not in used_invoices
            and (
                _invoice_mentions(invoice_by_id[row["invoice_record_id"]], payment)
                or (
                    "BULK" in normalize_identifier(payment.get("narration"))
                    and row["vendor_name_similarity"] >= 0.70
                    and row["date_gap_days"] <= 45
                )
            )
        ]
        # Prefer invoices explicitly named in the payment narration, then nearest dates.
        eligible_pairs.sort(
            key=lambda row: (
                row["reference_contains_invoice"] == 0,
                row["date_gap_days"],
            )
        )
        eligible = [invoice_by_id[row["invoice_record_id"]] for row in eligible_pairs[:20]]
        subset = find_bounded_subset(
            [_cents(row.get("total")) for row in eligible],
            _payment_value(payment),
            min_items=2,
            max_items=6,
            tolerance_paise=100,
            max_nodes=30000,
        )
        if subset is not None:
            selected = [eligible[position] for position in subset]
            invoice_ids = tuple(str(row["record_id"]) for row in selected)
            matches.append(Match(
                invoice_ids=invoice_ids,
                payment_ids=(payment_id,),
                tier="many_to_many",
                confidence=0.97,
                band="auto",
                features={"invoice_count": float(len(selected)), "amount_tolerance_paise": 100.0},
            ))
            used_invoices.update(invoice_ids)
            used_payments.add(payment_id)

    remaining_invoices = invoices[~invoices.record_id.astype(str).isin(used_invoices)].reset_index(drop=True)
    remaining_payments = payments[~payments.txn_id.astype(str).isin(used_payments)].reset_index(drop=True)
    remaining_candidate_frame = candidate_frame[
        ~candidate_frame.invoice_record_id.astype(str).isin(used_invoices)
        & ~candidate_frame.payment_id.astype(str).isin(used_payments)
    ]
    for match in fuzzy_matches(
        remaining_invoices,
        remaining_payments,
        classifier,
        candidates=remaining_candidate_frame,
    ):
        matches.append(match)
        if match.band == "auto":
            used_invoices.update(match.invoice_ids)
            used_payments.update(match.payment_ids)

    ledger_links, filing_links = _match_three_way(matches, invoices, ledger, filings)
    return ReconciliationResult(
        matches=matches,
        unmatched_invoice_ids=[
            str(record_id) for record_id in invoices.record_id
            if str(record_id) not in used_invoices
        ],
        unmatched_payment_ids=[
            str(txn_id) for txn_id in payments.txn_id
            if str(txn_id) not in used_payments
        ],
        ledger_links=ledger_links,
        filing_links=filing_links,
    )
