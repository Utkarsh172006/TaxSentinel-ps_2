from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from data.seeded_generator import generate, write_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
EVALUATION_DIR = DATA_DIR / "evaluation"
TABLES = (
    "invoices.csv",
    "bank_transactions.csv",
    "ledger.csv",
    "supplier_filings.csv",
    "vendor_master.csv",
    "tax_rates.csv",
)
TRUTH_TABLES = ("injected_errors.csv", "match_challenges.csv")
REFERENCE_TABLES = (
    "invoices_clean.csv",
    "bank_transactions_clean.csv",
    "ledger_clean.csv",
    "supplier_filings_clean.csv",
)
LINK_COLUMN_PREFIX = "linked_"


def _archive_members(archive: zipfile.ZipFile) -> dict[str, bytes]:
    """Read only safe archive members, without extracting paths to disk."""
    contents: dict[str, bytes] = {}
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in info.filename
            or ":" in info.filename
        ):
            raise ValueError(f"Unsafe archive member path: {info.filename!r}")
        if not info.is_dir():
            contents[info.filename] = archive.read(info)
    return contents


def _dataset_files(contents: dict[str, bytes]) -> dict[str, bytes]:
    """Select the supported dataset CSVs by basename from a zip member map."""
    required = set(TABLES) | set(TRUTH_TABLES)
    wanted = required | set(REFERENCE_TABLES)
    selected: dict[str, bytes] = {}
    for member, payload in contents.items():
        basename = PurePosixPath(member).name
        if basename in wanted:
            if basename in selected:
                raise ValueError(f"Archive contains multiple files named {basename!r}")
            selected[basename] = payload
    missing = required - selected.keys()
    if missing:
        raise ValueError(f"Dataset archive is missing required files: {', '.join(sorted(missing))}")
    return selected


def _read_archive(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return _dataset_files(_archive_members(archive))


def _read_bundled_test_a(bundle_path: Path) -> dict[str, bytes]:
    """Read the user-supplied Test A zip nested inside files.zip."""
    with zipfile.ZipFile(bundle_path) as bundle:
        nested = next(
            (item for item in bundle.infolist() if PurePosixPath(item.filename).name == "reconciliation_dataset_new.zip"),
            None,
        )
        if nested is None:
            raise FileNotFoundError("files.zip does not contain reconciliation_dataset_new.zip")
        payload = bundle.read(nested)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return _dataset_files(_archive_members(archive))


def _read_source_folder(folder: Path) -> dict[str, bytes]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Test B source dataset is unavailable: {folder}")
    contents = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    selected = _dataset_files(contents)
    return selected


def _read_csvs(files: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, payload in files.items():
        # Supplied synthetic CSVs include a rupee symbol in some text columns.
        frames[name] = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    return frames


def _write_split(
    frames: dict[str, pd.DataFrame],
    destination: Path,
    metadata: dict[str, object] | None = None,
) -> None:
    raw_dir = destination / "raw"
    truth_dir = destination / "truth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    for name in TABLES:
        frame = frames[name]
        forbidden = [column for column in frame.columns if column.startswith(LINK_COLUMN_PREFIX)]
        frame.drop(columns=forbidden, errors="ignore").to_csv(raw_dir / name, index=False)
    for name in TRUTH_TABLES:
        frames[name].to_csv(truth_dir / name, index=False)
    reference_dir = destination / "reference"
    reference_allowed = (metadata or {}).get("split") != "train"
    present_references = [
        name for name in REFERENCE_TABLES
        if reference_allowed and name in frames
    ]
    if present_references:
        reference_dir.mkdir(parents=True, exist_ok=True)
        for name in present_references:
            clean_frame = frames[name]
            forbidden = [column for column in clean_frame.columns if column.startswith(LINK_COLUMN_PREFIX)]
            clean_frame.drop(columns=forbidden, errors="ignore").to_csv(reference_dir / name, index=False)
    elif not reference_allowed and reference_dir.is_dir():
        for name in REFERENCE_TABLES:
            (reference_dir / name).unlink(missing_ok=True)
        try:
            reference_dir.rmdir()
        except OSError:
            pass

    links: list[pd.DataFrame] = []
    for table_name, key in (
        ("bank_transactions.csv", "txn_id"),
        ("ledger.csv", "entry_id"),
        ("supplier_filings.csv", "filing_id"),
    ):
        frame = frames[table_name]
        link_columns = [column for column in frame.columns if column.startswith(LINK_COLUMN_PREFIX)]
        if "linked_invoice_record_id" in link_columns:
            links.append(
                frame[[key, "linked_invoice_record_id"]].rename(
                    columns={key: "source_id", "linked_invoice_record_id": "invoice_record_id"}
                ).assign(source_table=table_name.removesuffix(".csv"))
            )
    if links:
        pd.concat(links, ignore_index=True).to_csv(truth_dir / "links.csv", index=False)

    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "raw_tables": list(TABLES),
                "truth_tables": list(TRUTH_TABLES) + (["links.csv"] if links else []),
                "detector_inputs_exclude_truth": True,
                "reference_tables": present_references,
                "reference_data_evaluation_only": reference_allowed,
                **(metadata or {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def prepare_data_splits() -> dict[str, Path]:
    """Build train seeds 11-13, validation seed 14, and isolated Test A/B splits."""
    generated: dict[str, Path] = {}
    for seed in (11, 12, 13):
        split_name = f"train/seed-{seed}"
        output_dir = SPLITS_DIR / "train" / f"seed-{seed}" / "generated"
        generated_folder = write_all(generate(seed), output_dir, seed)
        source_frames = _read_csvs(_dataset_files({
            path.name: path.read_bytes()
            for path in generated_folder.iterdir()
            if path.suffix.lower() == ".csv"
        }))
        target = SPLITS_DIR / split_name
        _write_split(source_frames, target, {"split": "train", "seed": seed, "evaluation_only": False})
        generated[split_name] = target

    val_name = "validation/seed-14"
    val_output = SPLITS_DIR / "validation" / "seed-14" / "generated"
    val_folder = write_all(generate(14), val_output, 14)
    val_frames = _read_csvs(_dataset_files({
        path.name: path.read_bytes()
        for path in val_folder.iterdir()
        if path.suffix.lower() == ".csv"
    }))
    val_target = SPLITS_DIR / val_name
    _write_split(val_frames, val_target, {"split": "validation", "seed": 14, "evaluation_only": False})
    generated[val_name] = val_target

    workspace_root = PROJECT_ROOT.parent
    test_a_path = workspace_root / "reconciliation_dataset_new.zip"
    bundle_path = workspace_root / "files.zip"
    if test_a_path.exists():
        test_a = _read_archive(test_a_path)
    elif bundle_path.exists():
        test_a = _read_bundled_test_a(bundle_path)
    else:
        raise FileNotFoundError(
            "Test A data missing. Provide reconciliation_dataset_new.zip or the supplied files.zip bundle."
        )
    test_a_target = EVALUATION_DIR / "test_a"
    _write_split(_read_csvs(test_a), test_a_target, {"split": "test_a", "evaluation_only": True})
    generated["evaluation/test_a"] = test_a_target

    test_b_path = workspace_root / "reconciliation_dataset.zip"
    test_b_source = DATA_DIR / "source" / "gst_reconciliation_dataset"
    if test_b_path.exists():
        test_b = _read_archive(test_b_path)
    else:
        test_b = _read_source_folder(test_b_source)
    test_b_target = EVALUATION_DIR / "test_b"
    _write_split(_read_csvs(test_b), test_b_target, {"split": "test_b", "evaluation_only": True})
    generated["evaluation/test_b"] = test_b_target
    return generated
