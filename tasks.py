from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from data.splits import prepare_data_splits
from eval.runner import run_evaluation
from matching.model import train_demo_model


def run_tests() -> int:
    root = Path(__file__).resolve().parent
    return subprocess.call([sys.executable, "-m", "pytest", str(root / "tests")])


def run_app() -> int:
    app_path = Path(__file__).resolve().parent / "app" / "Home.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main() -> int:
    parser = argparse.ArgumentParser(description="TaxSentinel commands")
    parser.add_argument("command", choices=["data", "train", "eval", "app", "test"])
    args = parser.parse_args()

    if args.command == "data":
        splits = prepare_data_splits()
        for name, directory in splits.items():
            print(f"{name}: {directory}")
        return 0
    if args.command == "train":
        train_demo_model()
        return 0
    if args.command == "eval":
        run_evaluation()
        return 0
    if args.command == "app":
        return run_app()
    if args.command == "test":
        return run_tests()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
