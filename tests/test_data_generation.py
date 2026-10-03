from __future__ import annotations

import json

import pandas as pd

from data.splits import prepare_data_splits


def test_seeded_and_heldout_splits_are_separate_and_leakage_safe():
    splits = prepare_data_splits()
    assert set(splits) == {
        "train/seed-11",
        "train/seed-12",
        "train/seed-13",
        "validation/seed-14",
        "evaluation/test_a",
        "evaluation/test_b",
    }

    for name, directory in splits.items():
        invoices = pd.read_csv(directory / "raw" / "invoices.csv")
        assert not any(column.lower().startswith("linked_") for column in invoices.columns)
        assert (directory / "truth" / "injected_errors.csv").is_file()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["detector_inputs_exclude_truth"] is True
        assert (directory / "truth" / "links.csv").is_file()
        reference_allowed = not name.startswith("train/")
        assert manifest["reference_data_evaluation_only"] is reference_allowed
        if reference_allowed:
            assert (directory / "reference" / "invoices_clean.csv").is_file()
            clean_invoices = pd.read_csv(directory / "reference" / "invoices_clean.csv")
            assert not any(column.lower().startswith("linked_") for column in clean_invoices.columns)
        else:
            assert not (directory / "reference").exists()
        for raw_table in (directory / "raw").glob("*.csv"):
            columns = pd.read_csv(raw_table, nrows=0).columns
            assert not any(column.lower().startswith("linked_") for column in columns)

    for seed in (11, 12, 13):
        manifest = json.loads(
            (splits[f"train/seed-{seed}"] / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["seed"] == seed
        assert manifest["evaluation_only"] is False
    for split in ("test_a", "test_b"):
        manifest = json.loads(
            (splits[f"evaluation/{split}"] / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["evaluation_only"] is True
    seed11 = pd.read_csv(splits["train/seed-11"] / "raw" / "invoices.csv")
    seed14 = pd.read_csv(splits["validation/seed-14"] / "raw" / "invoices.csv")
    assert not seed11.equals(seed14)
