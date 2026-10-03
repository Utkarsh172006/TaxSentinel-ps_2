from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ingest.loader import assert_inference_safe, load_detector_table
from matching.blocking import attach_training_labels, generate_pair_candidates
from matching.features import PAIR_FEATURES, normalize_vendor_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = PROJECT_ROOT / "data" / "splits" / "train"
MODEL_ROOT = PROJECT_ROOT / "models"
AUTO_THRESHOLD = 0.90


def _load_train_candidates() -> pd.DataFrame:
    """Build labeled candidates only from the three designated training seeds."""
    records: list[pd.DataFrame] = []
    for seed in (11, 12, 13):
        split_dir = TRAIN_ROOT / f"seed-{seed}"
        invoices = load_detector_table("invoices.csv", split_dir / "raw")
        payments = load_detector_table("bank_transactions.csv", split_dir / "raw")
        assert_inference_safe({"invoices.csv": invoices, "bank_transactions.csv": payments})
        links = pd.read_csv(split_dir / "truth" / "links.csv")
        candidates = generate_pair_candidates(invoices, payments)
        labeled = attach_training_labels(candidates, links)
        vendor_groups = invoices.set_index("record_id")["vendor_customer_name"].map(normalize_vendor_name)
        labeled["vendor_group"] = labeled.invoice_record_id.map(vendor_groups).fillna("UNKNOWN")
        labeled["seed"] = seed
        records.append(labeled)
    combined = pd.concat(records, ignore_index=True)
    if combined.empty or combined.label.nunique() < 2:
        raise ValueError("Training seeds did not produce both positive and negative candidate pairs.")
    return combined


def _new_lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=-1,
        class_weight="balanced",
        random_state=2026,
        verbosity=-1,
        n_jobs=1,
    )


def _group_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5):
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two distinct vendors are required for grouped cross-validation.")
    splitter = GroupKFold(n_splits=min(n_splits, unique_groups))
    return list(splitter.split(X, y, groups))


def _cross_validated_metrics(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    calibrated: bool,
) -> dict[str, float]:
    outer_splits = _group_splits(X, y, groups)
    probabilities = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in outer_splits:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        if calibrated:
            inner_groups = groups.iloc[train_idx]
            inner_splits = _group_splits(X_train, y_train, inner_groups, n_splits=3)
            model = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=inner_splits)
        else:
            model = estimator
        model.fit(X_train, y_train)
        probabilities[test_idx] = model.predict_proba(X_test)[:, 1]
    predicted = probabilities >= AUTO_THRESHOLD
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "precision_at_auto_threshold": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_auto_threshold": float(recall_score(y, predicted, zero_division=0)),
        "auto_threshold": AUTO_THRESHOLD,
    }


def train_pair_model() -> dict[str, Any]:
    """Train grouped-CV logistic and calibrated LightGBM pair models on seeds 11-13."""
    data = _load_train_candidates()
    X = data.loc[:, PAIR_FEATURES].astype(float)
    y = data["label"].astype(int)
    groups = data["vendor_group"].astype(str)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values found in matching features.")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026),
    )
    lightgbm = _new_lightgbm()
    baseline_metrics = _cross_validated_metrics(logistic, X, y, groups, calibrated=False)
    lightgbm_metrics = _cross_validated_metrics(lightgbm, X, y, groups, calibrated=True)

    calibration_splits = _group_splits(X, y, groups, n_splits=5)
    calibrated_model = CalibratedClassifierCV(
        estimator=lightgbm,
        method="sigmoid",
        cv=calibration_splits,
    )
    calibrated_model.fit(X, y)

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "pair_classifier.joblib"
    joblib.dump(calibrated_model, model_path)
    model_card = {
        "model": "LightGBM + sigmoid calibration",
        "training_seeds": [11, 12, 13],
        "validation_seed": 14,
        "heldout_sets_excluded": ["test_a", "test_b"],
        "training_rows": int(len(data)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int((y == 0).sum()),
        "vendor_groups": int(groups.nunique()),
        "features": list(PAIR_FEATURES),
        "cross_validation": {
            "splitter": "GroupKFold by normalized vendor",
            "logistic_regression_baseline": baseline_metrics,
            "calibrated_lightgbm": lightgbm_metrics,
        },
        "confidence_bands": {"auto": ">=0.90", "review": ">=0.60 and <0.90", "unmatched": "<0.60"},
    }
    (MODEL_ROOT / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    print(json.dumps(model_card, indent=2))
    return model_card


def load_pair_model(path: Path | None = None) -> Any | None:
    """Load the calibrated model when present, otherwise use the offline rule fallback."""
    model_path = path or MODEL_ROOT / "pair_classifier.joblib"
    return joblib.load(model_path) if model_path.exists() else None


def train_demo_model() -> dict[str, Any]:
    """Backward-compatible task entry point for pair-model training."""
    return train_pair_model()
