from __future__ import annotations

import re
import unicodedata


MISSING_TOKENS = {
    "",
    "n.d.",
    "nd",
    "n/d",
    "na",
    "nan",
    "none",
    "null",
}


MONTH_MAP_ES = {
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
}


MOJIBAKE_FIXES = {
    "Producci\ufffdn": "Producción",
    "seg\ufffdn": "según",
    "Hu\ufffdnuco": "Huánuco",
    "Jun\ufffdn": "Junín",
    "Esta\ufffdo": "Estaño",
    "Apur\ufffdmac": "Apurímac",
}


SPACES_RE = re.compile(r"\s+")


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def clean_text(value: str) -> str:
    text = value.strip()
    text = SPACES_RE.sub(" ", text)
    return text


def fix_mojibake(value: str) -> str:
    text = value
    for bad, good in MOJIBAKE_FIXES.items():
        text = text.replace(bad, good)
    # Remove unresolved replacement symbols if any remain.
    text = text.replace("\ufffd", "")
    return text


def normalize_text(value: str) -> str:
    return clean_text(fix_mojibake(value))
