from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
RAW_DIR = DATA_DIR / "raw"
TRUTH_DIR = DATA_DIR / "truth"
HELDOUT_DIR = DATA_DIR / "heldout"
ZIP_PATH = PROJECT_ROOT.parent / "reconciliation_dataset.zip"
FORBIDDEN_COLUMNS = {"linked_invoice_record_id"}
