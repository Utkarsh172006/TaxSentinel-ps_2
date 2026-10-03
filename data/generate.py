from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from config.settings import DATA_DIR, FORBIDDEN_COLUMNS, RAW_DIR, SOURCE_DIR, TRUTH_DIR, HELDOUT_DIR, ZIP_PATH


def _source_root() -> Path:
    candidates = [SOURCE_DIR / "gst_reconciliation_dataset", SOURCE_DIR]
    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return SOURCE_DIR


def validate_archive_members(zip_path: Path) -> list[str]:
    """List archive members while refusing any path traversal candidates."""
    members: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if info.filename.startswith("/") or ".." in pure.parts or pure.is_absolute():
                raise ValueError(f"Unsafe zip member: {info.filename!r}")
            members.append(info.filename)
    return members


def extract_source_archive(zip_path: Path, dest_dir: Path) -> None:
    """Safely extract the supplied archive if the source directory is still empty."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if any(dest_dir.iterdir()):
        return
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (dest_dir / info.filename).resolve()
            if dest_dir.resolve() not in target.parents and target != dest_dir.resolve():
                raise ValueError(f"Archive entry escapes target: {info.filename!r}")
            archive.extract(info, dest_dir)


def strip_forbidden_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in sorted(FORBIDDEN_COLUMNS):
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    return frame


def build_quality_summary() -> list[str]:
    """Profile the supplied dataset and return a 15-line summary."""
    source_path = _source_root()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing extracted source directory: {source_path}")
    files = [p for p in sorted(source_path.glob("*.csv"))]
    summaries: list[str] = []
    summaries.append(f"Archive-safe extraction: {len(validate_archive_members(ZIP_PATH))} members validated")
    summaries.append(f"Detected CSV files: {len(files)}")
    for csv_path in files[:5]:
        frame = pd.read_csv(csv_path)
        summaries.append(f"{csv_path.name}: {len(frame)} rows, {len(frame.columns)} columns")
    # Sample base stats from the invoices table.
    invoices = pd.read_csv(source_path / "invoices.csv")
    bank = pd.read_csv(source_path / "bank_transactions.csv")
    ledger = pd.read_csv(source_path / "ledger.csv")
    filings = pd.read_csv(source_path / "supplier_filings.csv")
    error_key = pd.read_csv(source_path / "injected_errors.csv", encoding="cp1252")
    summaries.append(f"Invoice date range: {invoices['date'].min()} to {invoices['date'].max()}")
    summaries.append(f"Bank link leakage: {'linked_invoice_record_id' in bank.columns} (true if present)")
    summaries.append(f"Ledger link leakage: {'linked_invoice_record_id' in ledger.columns} (true if present)")
    summaries.append(f"Filing link leakage: {'linked_invoice_record_id' in filings.columns} (true if present)")
    summaries.append(f"Injected error rows: {len(error_key)}; error types: {error_key['error_type'].nunique()}")
    summaries.append(f"Many-to-many challenge rows: {pd.read_csv(source_path / 'match_challenges.csv').shape[0]}")
    summaries.append(f"Split/bulk challenge labels exist: {pd.read_csv(source_path / 'match_challenges.csv')['challenge_type'].str.contains('split|bulk', case=False, na=False).any()}")
    summaries.append(f"Invalid GSTIN rows exist: {error_key['error_type'].str.contains('invalid_gstin', case=False).any()}")
    summaries.append(f"Anomaly codes present: {error_key['error_type'].str.contains('anomaly', case=False).any()}")
    summaries.append(f"Vendor-variant rows present: {error_key['error_type'].str.contains('vendor_name_variant|invoice_id_variant', case=False).any()}")
    summaries.append(f"Tax-rate change signals present: {('HSN' in pd.read_csv(source_path / 'tax_rates.csv').columns) and pd.read_csv(source_path / 'tax_rates.csv')['standard_tax_rate_pct'].nunique() > 1}")
    return summaries


def build_raw_tables() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source_root = _source_root()
    for csv_name in [
        "invoices.csv",
        "bank_transactions.csv",
        "ledger.csv",
        "supplier_filings.csv",
        "vendor_master.csv",
        "tax_rates.csv",
    ]:
        frame = pd.read_csv(source_root / csv_name)
        raw_frame = strip_forbidden_columns(frame)
        if "narration" in raw_frame.columns:
            raw_frame["narration"] = raw_frame["narration"].str.replace(r"INV-(\d+)", r"INV-\1", regex=True)
        raw_frame.to_csv(RAW_DIR / csv_name, index=False)


def build_truth_tables() -> None:
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    source_root = _source_root()
    for csv_name in ["injected_errors.csv", "match_challenges.csv"]:
        frame = pd.read_csv(source_root / csv_name, encoding="cp1252")
        frame.to_csv(TRUTH_DIR / csv_name, index=False)
    # Keep a compact links table for the training and evaluation assembly.
    bank = pd.read_csv(source_root / "bank_transactions.csv")
    ledger = pd.read_csv(source_root / "ledger.csv")
    filings = pd.read_csv(source_root / "supplier_filings.csv")
    links = pd.concat(
        [
            bank[["txn_id", "linked_invoice_record_id"]].rename(columns={"txn_id": "source_id", "linked_invoice_record_id": "invoice_record_id"}),
            ledger[["entry_id", "linked_invoice_record_id"]].rename(columns={"entry_id": "source_id", "linked_invoice_record_id": "invoice_record_id"}),
            filings[["filing_id", "linked_invoice_record_id"]].rename(columns={"filing_id": "source_id", "linked_invoice_record_id": "invoice_record_id"}),
        ],
        ignore_index=True,
    )
    links.to_csv(TRUTH_DIR / "links.csv", index=False)


def build_heldout_dataset() -> None:
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    base = RAW_DIR
    for csv_name in sorted(base.glob("*.csv")):
        frame = pd.read_csv(csv_name)
        if "date" in frame.columns:
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["date"] = frame["date"] + pd.to_timedelta(5, unit="D")
            frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        if "narration" in frame.columns:
            frame["narration"] = frame["narration"].astype(str).str.replace("INV-", "INV-")
        frame.to_csv(HELDOUT_DIR / csv_name.name, index=False)


def prepare_project_data() -> list[str]:
    """Create the safe source copy, generator outputs and a short data-quality report."""
    zip_path = ZIP_PATH
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing archive: {zip_path}")
    extract_source_archive(zip_path, SOURCE_DIR)
    build_raw_tables()
    build_truth_tables()
    build_heldout_dataset()
    summaries = build_quality_summary()
    for line in summaries[:15]:
        print(line)
    return summaries


if __name__ == "__main__":
    prepare_project_data()
