from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = PROJECT_ROOT / "config" / "thresholds.json"
GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))


def decimal_value(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid monetary value: {value!r}") from exc


def money(value: Any) -> Decimal:
    return decimal_value(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def normalized_tax_id(value: Any) -> str:
    return "".join(character for character in str(value or "").upper() if character.isalnum())


def gstin_checksum(gstin_prefix: str) -> str:
    if len(gstin_prefix) != 14 or any(character not in GSTIN_ALPHABET for character in gstin_prefix):
        return ""
    total = 0
    for index, character in enumerate(gstin_prefix):
        product = GSTIN_ALPHABET.index(character) * (1 if index % 2 == 0 else 2)
        total += product // 36 + product % 36
    return GSTIN_ALPHABET[(36 - total % 36) % 36]


def is_valid_gstin(value: Any) -> bool:
    gstin = normalized_tax_id(value)
    return (
        len(gstin) == 15
        and gstin[13] == str(load_thresholds()["gstin_checksum_character"])
        and all(character in GSTIN_ALPHABET for character in gstin)
        and gstin[14] == gstin_checksum(gstin[:14])
    )


def normalized_name(value: Any) -> str:
    import re

    name = re.sub(r"[^A-Z0-9 ]+", " ", str(value or "").upper().replace("&", " AND "))
    suffixes = {"LTD", "LIMITED", "PVT", "PRIVATE", "LLP", "PLC", "INC"}
    words = name.split()
    while words and words[-1] in suffixes:
        words.pop()
    return " ".join(words)


def name_similarity(left: Any, right: Any) -> float:
    return float(fuzz.token_set_ratio(normalized_name(left), normalized_name(right)))
