from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from data.splits import EVALUATION_DIR, SPLITS_DIR
from detect.issue import Issue
from ingest.loader import load_detector_table
from liability.liability import calculate_liability
from matching.features import normalize_identifier
from matching.pipeline import Match, ReconciliationResult
from app.service import run_reconciliation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "eval" / "results.json"
ANOMALY_TYPES = {
    "anomaly_approval_threshold",
    "anomaly_round_number",
    "anomaly_unusual_spike",
    "new_vendor_huge_invoice",
}

ERROR_GROUPS: dict[str, tuple[str, ...]] = {
    "payment_amount": (
        "amount_mismatch_large", "amount_mismatch_small", "bank_amount_mismatch",
        "rounding_diff", "short_payment_tds",
    ),
    "anomaly": (
        "anomaly_approval_threshold", "anomaly_round_number", "anomaly_unusual_spike",
        "new_vendor_huge_invoice",
    ),
    "date_and_period": ("date_shift", "ledger_date_gap", "wrong_tax_period", "payment_before_invoice"),
    "invoice_duplicates": ("duplicate_exact", "duplicate_near"),
    "duplicate_payments": ("duplicate_payment",),
    "filing_tax": ("gstr2b_tax_mismatch",),
    "unbooked_filing": ("in_gstr2b_not_in_books",),
    "gstin": ("invalid_gstin",),
    "invoice_identifier": ("invoice_id_variant",),
    "unreferenced_ledger": ("ledger_without_source",),
    "tax_calculation": ("miscalculated_tax",),
    "missing_ledger": ("missing_ledger_entry",),
    "missing_payment": ("missing_payment",),
    "unreferenced_payment": ("payment_without_invoice",),
    "supplier_filing": ("supplier_not_filed",),
    "vendor_name": ("vendor_name_variant",),
    "tax_rate": ("wrong_tax_rate",),
    "tax_split": ("wrong_tax_split",),
}
ISSUE_GROUPS: dict[str, set[str]] = {
    "payment_amount": {"short_payment", "overpayment", "ledger_amount_mismatch", "invoice_total_mismatch"},
    "anomaly": {
        "anomaly_approval_threshold", "anomaly_round_number", "anomaly_unusual_spike",
        "new_vendor_huge_invoice", "anomaly_behavioral",
    },
    "date_and_period": {"ledger_date_gap", "wrong_tax_period", "payment_before_invoice"},
    "invoice_duplicates": {"duplicate_exact", "duplicate_near"},
    "duplicate_payments": {"duplicate_payment"},
    "filing_tax": {"gstr2b_tax_mismatch"},
    "unbooked_filing": {"in_gstr2b_not_in_books"},
    "gstin": {"invalid_gstin"},
    "invoice_identifier": {"invoice_id_mismatch"},
    "unreferenced_ledger": {"ledger_without_source"},
    "tax_calculation": {"miscalculated_tax"},
    "missing_ledger": {"missing_ledger_entry"},
    "missing_payment": {"missing_payment"},
    "unreferenced_payment": {"payment_without_invoice"},
    "supplier_filing": {"supplier_not_filed", "missing_supplier_filing"},
    "vendor_name": {"vendor_name_mismatch"},
    "tax_rate": {"wrong_tax_rate"},
    "tax_split": {"wrong_tax_split"},
}
ERROR_CAUSES = {
    "payment_before_invoice": "Detection depends on a bank candidate surviving invoice/reference blocking; unmatched references are not date-compared.",
    "vendor_name_variant": "The mismatch rule targets invoice-to-filing names; equivalent company suffix/case variants are normalized and may not be surfaced.",
    "anomaly_approval_threshold": "The amount rule is conservative, while the broader vendor and Isolation Forest signals add unrelated anomaly candidates.",
    "anomaly_unusual_spike": "The robust-z and Isolation Forest signals are broad and miss spikes when vendor history is sparse or skewed.",
    "anomaly_round_number": "Round-number detection is suppressed for vendors whose history already contains many round totals.",
    "new_vendor_huge_invoice": "The detector depends on the source `is_new_vendor` flag and configured threshold matching the injected high invoice.",
    "wrong_tax_split": "Test B uses legacy conventions that differ from the Maharashtra CGST/SGST-versus-IGST rule.",
    "invalid_gstin": "Test B GSTIN values follow different conventions; the strict 15-character/checksum rule is intentionally reported separately.",
}


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _event_keys(row: dict[str, Any]) -> set[str]:
    return {key for key in (_safe_text(row.get("record_id")), _safe_text(row.get("row_id"))) if key}


def _issue_matches(issue: Issue, keys: set[str]) -> bool:
    return bool(keys.intersection(map(str, issue.record_ids)))


def score_issues(issues: list[Issue], truth: pd.DataFrame) -> dict[str, Any]:
    """Score detector events against labels, keeping ambiguous labels in one error family."""
    by_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth.to_dict("records"):
        by_error[_safe_text(row.get("error_type"))].append(row)

    per_type: dict[str, dict[str, Any]] = {}
    group_metrics = {}
    mapped_issue_ids: set[str] = set()
    total_tp = total_fp = total_fn = 0
    all_scored_errors = {name for group in ERROR_GROUPS.values() for name in group}
    all_truth_types = set(by_error)

    for group_name, error_types in ERROR_GROUPS.items():
        labels = [
            label for error_type in error_types
            for label in by_error.get(error_type, [])
        ]
        issue_types = ISSUE_GROUPS[group_name]
        predictions = [item for item in issues if item.issue_type in issue_types]
        mapped_issue_ids.update(item.issue_id for item in predictions)
        available = set(range(len(predictions)))
        matched_labels: dict[str, int] = defaultdict(int)
        tp = fn = 0
        for label in labels:
            keys = _event_keys(label)
            candidates = [
                index for index in available
                if _issue_matches(predictions[index], keys)
            ]
            exact_type = _safe_text(label.get("error_type"))
            candidates.sort(key=lambda index: predictions[index].issue_type != exact_type)
            if not candidates:
                fn += 1
                continue
            available.remove(candidates[0])
            matched_labels[exact_type] += 1
            tp += 1
        fp = len(available)
        group_prf = _prf(tp, fp, fn)
        group_metrics[group_name] = {
            **group_prf,
            "error_types": list(error_types),
            "detector_issue_types": sorted(issue_types),
            "labeled_errors": len(labels),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for error_type in error_types:
            count = len(by_error.get(error_type, []))
            type_tp = matched_labels[error_type]
            type_fn = count - type_tp
            precision = group_prf["precision"]
            recall = type_tp / count if count else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_type[error_type] = {
                "tp": type_tp,
                "fp": fp,
                "fn": type_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "labeled_errors": count,
                "precision_scope": f"shared {group_name} detector issue family",
            }
            if error_type in ERROR_CAUSES:
                per_type[error_type]["cause"] = ERROR_CAUSES[error_type]

    unscored_labels = all_truth_types - all_scored_errors
    total_fn += sum(len(by_error[name]) for name in unscored_labels)
    unscored_predictions = [item for item in issues if item.issue_id not in mapped_issue_ids]
    total_fp += len(unscored_predictions)
    for error_type in unscored_labels:
        metric = _prf(0, 0, len(by_error[error_type]))
        metric["labeled_errors"] = len(by_error[error_type])
        metric["cause"] = "No evaluator mapping or detector issue type is defined."
        per_type[error_type] = metric
    overall = _prf(total_tp, total_fp, total_fn)
    weak = sorted(
        per_type.items(),
        key=lambda item: (
            item[1]["f1"],
            item[1]["recall"],
            item[0],
        ),
    )[:5]
    return {
        "overall": overall,
        "grouped_metrics": group_metrics,
        "per_error_type": per_type,
        "weakest_error_types": [
            {
                "error_type": name,
                "f1": metric["f1"],
                "recall": metric["recall"],
                "cause": metric.get("cause") or ERROR_CAUSES.get(name),
            }
            for name, metric in weak
        ],
        "targets": {
            "precision_at_least_0_85": overall["precision"] >= 0.85,
            "recall_at_least_0_90": overall["recall"] >= 0.90,
            "both_met": overall["precision"] >= 0.85 and overall["recall"] >= 0.90,
        },
    }


def _model_pairs(result: ReconciliationResult) -> set[tuple[str, str]]:
    return {
        (str(invoice_id), str(payment_id))
        for match in result.matches
        for invoice_id in match.invoice_ids
        for payment_id in match.payment_ids
        if match.confidence >= 0.60
    }


def _truth_pairs(links: pd.DataFrame) -> set[tuple[str, str]]:
    bank_links = links[links.source_table.astype(str).eq("bank_transactions")]
    return {
        (str(row.invoice_record_id), str(row.source_id))
        for row in bank_links.itertuples()
        if _safe_text(row.invoice_record_id) and _safe_text(row.source_id)
    }


def _exact_baseline_pairs(invoices: pd.DataFrame, bank: pd.DataFrame) -> set[tuple[str, str]]:
    """Naive one-to-one invoice-number and total-only baseline; no fuzzy or grouped matching."""
    available = bank.to_dict("records")
    used: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for invoice in invoices.to_dict("records"):
        invoice_no = normalize_identifier(invoice.get("invoice_no"))
        if not invoice_no:
            continue
        invoice_total = abs(float(invoice.get("total", 0) or 0))
        candidates = []
        for payment in available:
            payment_id = str(payment.get("txn_id"))
            reference = normalize_identifier(payment.get("reference"))
            if payment_id in used or reference != invoice_no:
                continue
            payment_total = abs(float(payment.get("amount", 0) or 0)) + abs(
                float(payment.get("tds_withheld", 0) or 0)
            )
            difference = abs(invoice_total - payment_total)
            if difference <= 1.0:
                candidates.append((difference, payment_id))
        if candidates:
            _, payment_id = min(candidates)
            pairs.add((str(invoice["record_id"]), payment_id))
            used.add(payment_id)
    return pairs


def _pair_metrics(predicted: set[tuple[str, str]], actual: set[tuple[str, str]]) -> dict[str, Any]:
    tp = len(predicted & actual)
    return _prf(tp, len(predicted - actual), len(actual - predicted))


def _match_threshold_curve(
    matches: list[Match],
    actual: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    points = []
    thresholds = [round(0.60 + index * 0.01, 2) for index in range(40)]
    for threshold in thresholds:
        predicted = {
            (str(invoice_id), str(payment_id))
            for match in matches
            if match.confidence >= threshold
            for invoice_id in match.invoice_ids
            for payment_id in match.payment_ids
        }
        points.append({"threshold": threshold, **_pair_metrics(predicted, actual)})
    return points


def _many_to_many_metrics(
    matches: list[Match],
    challenges: pd.DataFrame,
    links: pd.DataFrame,
) -> dict[str, Any]:
    actual_groups: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for row in challenges.to_dict("records"):
        kind = _safe_text(row.get("challenge_type"))
        if kind not in {"bulk_payment", "split_payment"}:
            continue
        invoice_ids = tuple(sorted(filter(None, _safe_text(row.get("record_id")).split("|"))))
        if kind == "bulk_payment":
            payment_ids = (_safe_text(row.get("row_id")),)
        else:
            payment_ids = tuple(sorted(
                str(link.source_id)
                for link in links.itertuples()
                if str(link.source_table) == "bank_transactions"
                and str(link.invoice_record_id) in set(invoice_ids)
            ))
        if invoice_ids and payment_ids and all(payment_ids):
            actual_groups.add((invoice_ids, payment_ids))

    predicted_groups = {
        (tuple(sorted(map(str, match.invoice_ids))), tuple(sorted(map(str, match.payment_ids))))
        for match in matches
        if match.tier == "many_to_many" and match.confidence >= 0.60
    }
    tp = len(actual_groups & predicted_groups)
    fp = len(predicted_groups - actual_groups)
    fn = len(actual_groups - predicted_groups)
    return {
        **_prf(tp, fp, fn),
        "exact_group_accuracy": tp / len(actual_groups) if actual_groups else 0.0,
        "expected_groups": len(actual_groups),
        "predicted_groups": len(predicted_groups),
    }


def _anomaly_precision_at_k(issues: list[Issue], truth: pd.DataFrame) -> dict[str, Any]:
    relevant_ids = {
        _safe_text(row.get("record_id"))
        for row in truth.to_dict("records")
        if _safe_text(row.get("error_type")) in ANOMALY_TYPES and _safe_text(row.get("record_id"))
    }
    ranked = sorted(
        (
            issue for issue in issues
            if issue.issue_type in ISSUE_GROUPS["anomaly"]
        ),
        key=lambda issue: (issue.confidence, issue.rupee_impact),
        reverse=True,
    )
    k = len(relevant_ids)
    top = ranked[:k]
    relevant = sum(bool(relevant_ids.intersection(map(str, item.record_ids))) for item in top)
    return {
        "k": k,
        "predictions_available": len(ranked),
        "relevant_in_top_k": relevant,
        "precision_at_k": relevant / k if k else 0.0,
    }


def _liability_error(
    result_liability: dict[str, Any],
    clean_invoices: pd.DataFrame,
    clean_filings: pd.DataFrame,
) -> dict[str, Any]:
    expected = calculate_liability(clean_invoices, clean_filings, issues=[])["totals"]
    observed = result_liability["totals"]
    output = {}
    for metric_name, key in (
        ("net_liability", "net_liability_after"),
        ("itc_at_risk", "itc_at_risk"),
    ):
        actual = float(expected[key])
        predicted = float(observed[key])
        difference = abs(predicted - actual)
        output[metric_name] = {
            "expected_rupees": actual,
            "predicted_rupees": predicted,
            "absolute_error_rupees": difference,
            "absolute_error_pct": difference / abs(actual) * 100 if actual else None,
        }
    return output


def evaluate_split(name: str, directory: Path) -> dict[str, Any]:
    raw = directory / "raw"
    tables = {
        "invoices": load_detector_table("invoices.csv", raw),
        "bank_transactions": load_detector_table("bank_transactions.csv", raw),
        "ledger": load_detector_table("ledger.csv", raw),
        "supplier_filings": load_detector_table("supplier_filings.csv", raw),
        "tax_rates": load_detector_table("tax_rates.csv", raw),
    }
    result = run_reconciliation(tables)

    # Read scoring artifacts only after all inference and issue generation has finished.
    truth = pd.read_csv(directory / "truth" / "injected_errors.csv", dtype=str)
    challenges = pd.read_csv(directory / "truth" / "match_challenges.csv", dtype=str)
    links = pd.read_csv(directory / "truth" / "links.csv", dtype=str)
    issue_metrics = score_issues(result.issues, truth)
    model_pairs = _model_pairs(result.reconciliation)
    actual_pairs = _truth_pairs(links)
    baseline_pairs = _exact_baseline_pairs(tables["invoices"], tables["bank_transactions"])
    reference_dir = directory / "reference"
    clean_invoice_path = reference_dir / "invoices_clean.csv"
    clean_filings_path = reference_dir / "supplier_filings_clean.csv"
    if not clean_invoice_path.is_file() or not clean_filings_path.is_file():
        raise FileNotFoundError(
            f"Clean evaluation references missing for {name}; rerun `python tasks.py data`."
        )
    clean_invoices = pd.read_csv(clean_invoice_path)
    clean_filings = pd.read_csv(clean_filings_path)
    clean_invoices = clean_invoices.drop(
        columns=[column for column in clean_invoices if column.lower().startswith("linked_")],
        errors="ignore",
    )
    clean_filings = clean_filings.drop(
        columns=[column for column in clean_filings if column.lower().startswith("linked_")],
        errors="ignore",
    )
    anomaly_precision = _anomaly_precision_at_k(result.issues, truth)

    return {
        "issue_detection": issue_metrics,
        "matching": {
            "model": _pair_metrics(model_pairs, actual_pairs),
            "naive_exact_invoice_number_and_amount": _pair_metrics(baseline_pairs, actual_pairs),
            "model_match_groups": len(result.reconciliation.matches),
            "model_matched_invoice_count": len({
                invoice_id
                for match in result.reconciliation.matches
                if match.confidence >= 0.60
                for invoice_id in match.invoice_ids
            }),
            "validation_threshold_curve": (
                _match_threshold_curve(result.reconciliation.matches, actual_pairs)
                if name == "validation" else []
            ),
        },
        "many_to_many": _many_to_many_metrics(result.reconciliation.matches, challenges, links),
        "anomaly_precision_at_k": anomaly_precision,
        "liability_error": _liability_error(result.liability, clean_invoices, clean_filings),
        "counts": {
            "invoice_rows": len(tables["invoices"]),
            "bank_rows": len(tables["bank_transactions"]),
            "truth_error_rows": len(truth),
            "detected_issue_rows": len(result.issues),
        },
        "test_b_convention_issue_types_reported_separately": (
            ["wrong_tax_split", "invalid_gstin"] if name == "test_b" else []
        ),
    }


def run_evaluation() -> dict[str, Any]:
    """Run fixed validation and held-out evaluations, then save a report for the app."""
    splits = {
        "validation": SPLITS_DIR / "validation" / "seed-14",
        "test_a": EVALUATION_DIR / "test_a",
        "test_b": EVALUATION_DIR / "test_b",
    }
    for name, directory in splits.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} split unavailable; run `python tasks.py data` first.")
    results = {
        "protocol": {
            "training_data": ["seed-11", "seed-12", "seed-13"],
            "validation_data": ["seed-14"],
            "heldout_evaluation": ["test_a", "test_b"],
            "test_sets_used_for_training_or_tuning": False,
            "liability_reference": "clean source tables isolated under each split/reference/",
            "issue_scoring": "mapped injected error types and source row identifiers",
            "matching_baseline": "exact normalized invoice number and amount only",
        },
        "datasets": {},
    }
    for name, directory in splits.items():
        print(f"Evaluating {name}...")
        results["datasets"][name] = evaluate_split(name, directory)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved results to {RESULTS_PATH}")
    return results
