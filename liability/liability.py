from __future__ import annotations

import json
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd

from detect.common import guard_inference
from detect.rules import money
from liability.itc import calculate_itc_risk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PENALTIES_PATH = PROJECT_ROOT / "config" / "penalties.json"


def _tax_for_invoice(invoice: dict[str, Any], rate_overrides: dict[str, float] | None) -> Decimal:
    hsn = str(invoice.get("HSN", "")).lstrip("0")
    overrides = {str(key).lstrip("0"): value for key, value in (rate_overrides or {}).items()}
    if hsn in overrides:
        taxable = money(invoice.get("taxable_value"))
        rate = Decimal(str(overrides[hsn]))
        return money(taxable * rate / Decimal("100"))
    return sum(
        (money(invoice.get(column, 0)) for column in ("CGST", "SGST", "IGST")),
        Decimal("0"),
    )


def calculate_liability(
    invoices: pd.DataFrame,
    filings: pd.DataFrame,
    issues: list | None = None,
    resolved_issue_ids: set[str] | None = None,
    rate_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute monthly output tax, eligible ITC, ITC at risk and net liability."""
    guard_inference(invoices=invoices, filings=filings)
    _, risk_rows = calculate_itc_risk(
        invoices,
        filings,
        issues=issues,
        resolved_issue_ids=resolved_issue_ids,
    )
    at_risk_ids = {row["record_id"] for row in risk_rows}
    monthly: dict[str, dict[str, Decimal]] = {}

    for invoice in invoices.to_dict("records"):
        period = str(invoice.get("date", ""))[:7]
        if len(period) != 7:
            continue
        row = monthly.setdefault(period, {
            "output_tax": Decimal("0"),
            "booked_purchase_itc": Decimal("0"),
            "eligible_itc": Decimal("0"),
            "itc_at_risk": Decimal("0"),
        })
        tax = _tax_for_invoice(invoice, rate_overrides)
        if str(invoice.get("document_type", "purchase")).lower() == "purchase":
            row["booked_purchase_itc"] += tax
            if str(invoice["record_id"]) in at_risk_ids:
                row["itc_at_risk"] += tax
            else:
                row["eligible_itc"] += tax
        else:
            row["output_tax"] += tax

    period_rows = []
    for period, row in sorted(monthly.items()):
        net_before = row["output_tax"] - row["booked_purchase_itc"]
        net_after = row["output_tax"] - row["eligible_itc"]
        period_rows.append({
            "period": period,
            **{key: float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) for key, value in row.items()},
            "net_liability_before": float(net_before.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            "net_liability_after": float(net_after.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        })

    total = {
        key: sum((row[key] for row in monthly.values()), Decimal("0"))
        for key in ("output_tax", "booked_purchase_itc", "eligible_itc", "itc_at_risk")
    }
    total["net_liability_before"] = total["output_tax"] - total["booked_purchase_itc"]
    total["net_liability_after"] = total["output_tax"] - total["eligible_itc"]
    invoice_by_id = {str(row["record_id"]): row for row in invoices.to_dict("records")}
    risk_rows = [
        {
            **risk,
            "itc_at_risk": float(_tax_for_invoice(
                invoice_by_id[risk["record_id"]],
                rate_overrides,
            )),
        }
        for risk in risk_rows
    ]
    return {
        "periods": period_rows,
        "totals": {
            key: float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            for key, value in total.items()
        },
        "itc_at_risk_by_invoice": risk_rows,
        "notice": "Not tax advice. Verify rates and eligibility against official GST Council/CBIC notifications.",
    }


def estimate_interest_and_penalty(
    tax_amount: Decimal | float,
    from_date: date,
    to_date: date,
) -> dict[str, float | int | str]:
    """Estimate interest and configured penalty using the effective rule in JSON."""
    if to_date < from_date:
        raise ValueError("to_date must not be earlier than from_date")
    config = json.loads(PENALTIES_PATH.read_text(encoding="utf-8"))
    rule = next(
        (
            row for row in config["effective_rates"]
            if date.fromisoformat(row["valid_from"]) <= from_date
            and (row["valid_to"] is None or from_date <= date.fromisoformat(row["valid_to"]))
        ),
        None,
    )
    if rule is None:
        raise ValueError(f"No effective interest/penalty rule on {from_date.isoformat()}")
    days = (to_date - from_date).days
    principal = money(tax_amount)
    interest = money(
        principal * Decimal(str(rule["interest_rate_annual_pct"])) / Decimal("100")
        * Decimal(days) / Decimal("365")
    )
    penalty = money(principal * Decimal(str(rule["penalty_rate_pct"])) / Decimal("100"))
    return {
        "days": days,
        "interest": float(interest),
        "penalty": float(penalty),
        "total_estimate": float(interest + penalty),
        "interest_rate_annual_pct": float(rule["interest_rate_annual_pct"]),
        "penalty_rate_pct": float(rule["penalty_rate_pct"]),
        "notice": config["notice"],
    }
