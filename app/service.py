from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from detect.issue import Issue
from detect.pipeline import detect_all
from ingest.loader import load_detector_table
from liability.liability import calculate_liability
from matching.exact import Match
from matching.model import load_pair_model
from matching.pipeline import ReconciliationResult, reconcile
from vendor_risk.score import score_vendors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_FILES = {
    "invoices": "invoices.csv",
    "bank_transactions": "bank_transactions.csv",
    "ledger": "ledger.csv",
    "supplier_filings": "supplier_filings.csv",
    "tax_rates": "tax_rates.csv",
}


@dataclass
class PipelineResult:
    tables: dict[str, pd.DataFrame]
    reconciliation: ReconciliationResult
    issues: list[Issue]
    liability: dict[str, Any]
    vendors: pd.DataFrame


def load_demo_tables(split: str = "test_a") -> dict[str, pd.DataFrame]:
    """Load detector-only data from one designated evaluation split."""
    if split not in {"validation", "test_a", "test_b"}:
        raise ValueError("Demo data split must be validation, test_a, or test_b.")
    if split == "validation":
        base = PROJECT_ROOT / "data" / "splits" / "validation" / "seed-14" / "raw"
    else:
        base = PROJECT_ROOT / "data" / "evaluation" / split / "raw"
    if not base.is_dir():
        raise FileNotFoundError("Demo dataset is not prepared. Run `python tasks.py data` first.")
    return {
        key: load_detector_table(filename, base)
        for key, filename in TABLE_FILES.items()
    }


def run_reconciliation(
    tables: dict[str, pd.DataFrame],
    on_stage: Callable[[str, int], None] | None = None,
) -> PipelineResult:
    required = set(TABLE_FILES)
    missing = required - tables.keys()
    if missing:
        raise ValueError(f"Missing required tables for reconciliation: {sorted(missing)}")
    invoice = tables["invoices"]
    bank = tables["bank_transactions"]
    ledger = tables["ledger"]
    filings = tables["supplier_filings"]
    rates = tables["tax_rates"]
    model = load_pair_model()
    if on_stage:
        on_stage("Building match candidates", 10)
    matching = reconcile(invoice, bank, ledger, filings, classifier=model)
    if on_stage:
        on_stage("Detecting reconciliation issues", 45)
    issues = detect_all(invoice, bank, ledger, filings, rates, classifier=model)
    if on_stage:
        on_stage("Calculating liability and ITC risk", 75)
    liability = calculate_liability(invoice, filings, issues)
    if on_stage:
        on_stage("Computing vendor risk and reports", 90)
    vendors = score_vendors(invoice, issues)
    if on_stage:
        on_stage("Review queue ready", 100)
    return PipelineResult(
        tables=tables,
        reconciliation=matching,
        issues=issues,
        liability=liability,
        vendors=vendors,
    )
