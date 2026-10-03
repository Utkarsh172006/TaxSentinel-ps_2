from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import FORBIDDEN_COLUMNS


FORBIDDEN_TABLE_NAMES = {"injected_errors.csv", "match_challenges.csv", "links.csv"}
FORBIDDEN_LABEL_COLUMNS = {"error_type", "challenge_type"}


def assert_inference_safe(tables: dict[str, pd.DataFrame]) -> None:
    """Fail closed if answer-key data or ground-truth links enter inference."""
    for name, frame in tables.items():
        if Path(name).name.lower() in FORBIDDEN_TABLE_NAMES:
            raise ValueError(f"Ground-truth table is forbidden at inference: {name}")
        linked_columns = [column for column in frame.columns if column.lower().startswith("linked_")]
        if linked_columns:
            raise ValueError(f"Ground-truth link columns are forbidden at inference: {linked_columns}")
        label_columns = [column for column in frame.columns if column.lower() in FORBIDDEN_LABEL_COLUMNS]
        if label_columns:
            raise ValueError(f"Ground-truth label columns are forbidden at inference: {label_columns}")


def strip_forbidden_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove hidden link columns before any matching or issue detection."""
    forbidden = [
        column for column in frame.columns
        if column in FORBIDDEN_COLUMNS or column.lower().startswith("linked_")
    ]
    return frame.drop(columns=forbidden, errors="ignore")


def load_detector_table(name: str, directory: Path | str | None = None) -> pd.DataFrame:
    """Load a detector input table from the raw data directory."""
    base_dir = Path(directory) if directory is not None else Path("data/raw")
    path = base_dir / name
    if path.name.lower() in FORBIDDEN_TABLE_NAMES:
        raise ValueError(f"Ground-truth table is forbidden at inference: {path.name}")
    frame = pd.read_csv(path)
    assert_inference_safe({path.name: frame})
    return frame


def lint_dataset(raw_dir: Path | str) -> dict[str, list[str]]:
    """Return a simple validation report for raw detector inputs."""
    raw_dir = Path(raw_dir)
    findings: dict[str, list[str]] = {}
    for csv_path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        issues = []
        for column in df.columns:
            if column in FORBIDDEN_COLUMNS or column.lower().startswith("linked_"):
                issues.append(f"forbidden column in {csv_path.name}: {column}")
            if column.lower() in FORBIDDEN_LABEL_COLUMNS:
                issues.append(f"forbidden label column in {csv_path.name}: {column}")
        if csv_path.name.lower() in FORBIDDEN_TABLE_NAMES:
            issues.append(f"forbidden truth table in inference directory: {csv_path.name}")
        if issues:
            findings[csv_path.name] = issues
    return findings
