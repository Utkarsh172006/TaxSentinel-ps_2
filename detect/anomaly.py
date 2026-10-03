from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest

from detect.common import guard_inference, issue
from detect.rules import load_thresholds, money
from matching.features import normalize_identifier, normalize_vendor_name

BENFORD_PROBABILITIES = np.array([
    math.log10(1 + 1 / digit) for digit in range(1, 10)
])


def _first_digit(value: Any) -> int | None:
    try:
        amount = abs(float(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or amount == 0:
        return None
    return int(f"{amount:.12g}".lstrip("0").lstrip(".")[0])


def _robust_z(values: pd.Series) -> pd.Series:
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return 0.6745 * (values - median) / mad


def _payment_delays(invoices: pd.DataFrame, bank: pd.DataFrame) -> dict[str, int]:
    by_reference = {
        normalize_identifier(row.get("invoice_no")): row
        for row in invoices.to_dict("records")
        if normalize_identifier(row.get("invoice_no"))
    }
    delays: dict[str, int] = {}
    for payment in bank.to_dict("records"):
        reference = normalize_identifier(payment.get("reference"))
        invoice = by_reference.get(reference)
        if not invoice:
            continue
        try:
            delay = (
                pd.Timestamp(payment["date"]) - pd.Timestamp(invoice["date"])
            ).days
        except (KeyError, TypeError, ValueError):
            continue
        if delay >= 0:
            delays[str(invoice["record_id"])] = delay
    return delays


def detect_anomalies(
    invoices: pd.DataFrame,
    bank: pd.DataFrame | None = None,
) -> tuple[list, dict[str, dict[str, Any]]]:
    """Combine Isolation Forest, vendor robust statistics, Benford and rules."""
    guard_inference(invoices=invoices, bank=bank)
    if invoices.empty:
        return [], {}
    thresholds = load_thresholds()["anomaly"]
    frame = invoices.copy().reset_index(drop=True)
    frame["_total"] = pd.to_numeric(frame["total"], errors="coerce").abs().fillna(0.0)
    frame["_log_amount"] = np.log1p(frame["_total"])
    vendor_column = frame.get("vendor_customer_name", pd.Series(["Unknown"] * len(frame)))
    frame["_vendor_key"] = vendor_column.map(normalize_vendor_name)
    frame["_robust_z"] = frame.groupby("_vendor_key", group_keys=False)["_log_amount"].apply(_robust_z)

    feature_matrix = frame[["_log_amount"]].to_numpy()
    if len(frame) >= 40:
        forest = IsolationForest(
            contamination=float(thresholds["isolation_forest_contamination"]),
            random_state=2026,
            n_estimators=150,
        )
        frame["_iforest"] = forest.fit_predict(feature_matrix)
    else:
        frame["_iforest"] = 1

    round_threshold = Decimal("1000")
    vendor_round_rate = frame.assign(
        _round=frame["_total"].map(lambda amount: int(amount >= 10000 and amount % 1000 == 0))
    ).groupby("_vendor_key")["_round"].mean()
    weekly = pd.to_datetime(frame["date"], errors="coerce").dt.to_period("W")
    invoice_week_counts = frame.assign(_week=weekly).groupby(["_vendor_key", "_week"])["record_id"].transform("count")
    delays = _payment_delays(invoices, bank) if bank is not None else {}

    reasons_by_record: dict[str, dict[str, Any]] = {}
    for position, row in frame.iterrows():
        reasons: list[str] = []
        amount = float(row["_total"])
        robust_z = float(row["_robust_z"])
        vendor_key = str(row["_vendor_key"])
        vendor_round = float(vendor_round_rate.get(vendor_key, 0.0))
        if int(row["_iforest"]) == -1:
            reasons.append("Isolation Forest classified the amount profile as unusual.")
        if abs(robust_z) >= float(thresholds["vendor_robust_z_threshold"]):
            reasons.append(f"Vendor robust amount z-score is {robust_z:.2f}.")
        if vendor_round < 0.10 and amount >= 10000 and amount % float(round_threshold) == 0:
            reasons.append("Round-number amount is unusual for this vendor.")
        if bool(row.get("is_new_vendor", False)) and amount >= float(thresholds["new_vendor_huge_invoice_rupees"]):
            reasons.append("New vendor invoice exceeds the configured large-invoice threshold.")
        approval = float(thresholds["approval_threshold_rupees"])
        window = float(thresholds["approval_threshold_window_rupees"])
        if approval - window <= amount < approval:
            reasons.append("Invoice falls just below the configured approval threshold.")
        if int(invoice_week_counts.iloc[position]) >= 4:
            reasons.append("Vendor invoice count is unusually high in this week.")
        delay = delays.get(str(row["record_id"]))
        if delay is not None and delay > 43:
            reasons.append(f"Payment delay is {delay} days, beyond the expected timing window.")
        if reasons:
            typical = float(frame.loc[frame["_vendor_key"] == vendor_key, "_total"].median())
            impact = abs(money(amount) - money(typical))
            reasons_by_record[str(row["record_id"])] = {
                "reasons": reasons,
                "robust_z": robust_z,
                "amount": round(amount, 2),
                "vendor_median": round(typical, 2),
                "payment_delay_days": delay,
                "isolation_forest_flag": int(row["_iforest"]) == -1,
                "rupee_impact": float(impact),
            }

    # Benford tests are only meaningful for vendors with at least 300 invoices.
    for vendor_key, group in frame.groupby("_vendor_key"):
        if len(group) < 300:
            continue
        digits = [digit for digit in group["_total"].map(_first_digit) if digit is not None and 1 <= digit <= 9]
        if len(digits) < 300:
            continue
        observed = np.bincount(digits, minlength=10)[1:10]
        expected = BENFORD_PROBABILITIES * len(digits)
        statistic, p_value = chisquare(observed, expected)
        effect_size = float(statistic / len(digits))
        if p_value < 0.01 and effect_size >= 0.10:
            for record_id in group.record_id.astype(str):
                entry = reasons_by_record.setdefault(record_id, {
                    "reasons": [],
                    "robust_z": 0.0,
                    "amount": float(group.loc[group.record_id.astype(str).eq(record_id), "_total"].iloc[0]),
                    "vendor_median": float(group["_total"].median()),
                    "payment_delay_days": None,
                    "isolation_forest_flag": False,
                    "rupee_impact": 0.0,
                })
                entry["reasons"].append(
                    f"Vendor Benford test is unusual (n={len(digits)}, p={p_value:.4g}, effect={effect_size:.3f})."
                )
                entry["benford"] = {"n": len(digits), "chi_square": float(statistic), "p_value": float(p_value), "effect_size": effect_size}

    issues = []
    invoice_lookup = frame.set_index("record_id").to_dict("index")
    for record_id, evidence in reasons_by_record.items():
        row = invoice_lookup[record_id]
        severity = "High" if abs(float(evidence["robust_z"])) >= 6 or evidence.get("benford") else "Medium"
        reasons = evidence["reasons"]
        if any("approval threshold" in reason for reason in reasons):
            issue_type = "anomaly_approval_threshold"
        elif any("Round-number amount" in reason for reason in reasons):
            issue_type = "anomaly_round_number"
        elif any("New vendor invoice" in reason for reason in reasons):
            issue_type = "new_vendor_huge_invoice"
        elif any(
            "Isolation Forest" in reason
            or "robust amount z-score" in reason
            or "Benford test" in reason
            for reason in reasons
        ):
            issue_type = "anomaly_unusual_spike"
        else:
            issue_type = "anomaly_behavioral"
        issues.append(issue(
            issue_type,
            severity,
            str(row.get("vendor_customer_name", "Unknown")),
            [record_id],
            f"Vendor median {evidence['vendor_median']:.2f}",
            f"Observed amount {evidence['amount']:.2f}",
            evidence["rupee_impact"],
            evidence,
            "Review source documents, approval history and payment timing before posting.",
            confidence=0.85 if evidence.get("benford") else 0.75,
        ))
    return issues, reasons_by_record


def benford_summary(invoices: pd.DataFrame, minimum_sample_size: int = 300) -> list[dict[str, Any]]:
    """Return per-vendor Benford statistics for sufficiently large samples."""
    summaries = []
    for vendor, group in invoices.groupby("vendor_customer_name", dropna=False):
        digits = [digit for digit in group.total.map(_first_digit) if digit is not None and 1 <= digit <= 9]
        if len(digits) < minimum_sample_size:
            continue
        observed = np.bincount(digits, minlength=10)[1:10]
        statistic, p_value = chisquare(observed, BENFORD_PROBABILITIES * len(digits))
        summaries.append({
            "vendor": str(vendor),
            "sample_size": len(digits),
            "chi_square": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(statistic / len(digits)),
            "observed_first_digits": observed.tolist(),
        })
    return summaries
