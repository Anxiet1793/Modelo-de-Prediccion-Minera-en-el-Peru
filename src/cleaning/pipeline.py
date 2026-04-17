from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning.rules import MISSING_TOKENS, MONTH_MAP_ES, normalize_text, strip_accents


SERIES_PATTERN = re.compile(r"^(.+?) - (.+?) - (.+?) \(([^)]+)\)$")
PERIOD_PATTERN = re.compile(r"^([A-Za-z]{3})(\d{2})$")


@dataclass
class CleaningOutputs:
    full_output_path: Path
    base_output_path: Path
    report_path: Path


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase2_cleaning")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_period(period_raw: str) -> pd.Timestamp | pd.NaT:
    if period_raw is None:
        return pd.NaT

    token = normalize_text(str(period_raw))
    match = PERIOD_PATTERN.match(token)
    if not match:
        return pd.NaT

    month_txt = strip_accents(match.group(1).lower())
    year_short = int(match.group(2))
    month = MONTH_MAP_ES.get(month_txt)
    if month is None:
        return pd.NaT

    year = 2000 + year_short
    return pd.Timestamp(year=year, month=month, day=1)


def parse_series_metadata(description_raw: str) -> tuple[str | None, str | None, str | None]:
    if description_raw is None:
        return None, None, None

    description = normalize_text(str(description_raw))
    match = SERIES_PATTERN.match(description)
    if not match:
        clean_ascii = strip_accents(description)
        match = SERIES_PATTERN.match(clean_ascii)
        if not match:
            return None, None, None

    # Expected form:
    # Produccion de productos mineros segun departamentos - Mineral - Departamento (unidad)
    mineral = normalize_text(match.group(2))
    departamento = normalize_text(match.group(3))
    unidad = normalize_text(match.group(4)).lower()
    return mineral, departamento, unidad


def parse_numeric(value_raw: object) -> tuple[float, bool, bool]:
    """
    Returns: (value, is_missing, is_invalid_numeric).
    """
    if pd.isna(value_raw):
        return np.nan, True, False

    token = normalize_text(str(value_raw))
    token_norm = strip_accents(token).lower()
    if token_norm in MISSING_TOKENS:
        return np.nan, True, False

    token_norm = token_norm.replace(",", "")
    try:
        return float(token_norm), False, False
    except ValueError:
        return np.nan, False, True


def load_excel_monthly_to_long(excel_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name="Mensuales", header=None)

    series_codes = [normalize_text(str(code)) for code in raw.iloc[0, 1:].tolist()]
    series_descriptions = [normalize_text(str(desc)) for desc in raw.iloc[1, 1:].tolist()]
    code_to_description = dict(zip(series_codes, series_descriptions))

    monthly = raw.iloc[2:, :].copy()
    monthly = monthly.rename(columns={0: "period_raw"})
    monthly_data = monthly.iloc[:, 1:]
    monthly_data.columns = series_codes

    long_df = monthly_data.assign(period_raw=monthly["period_raw"].values).melt(
        id_vars=["period_raw"],
        var_name="series_code",
        value_name="value_raw",
    )
    long_df["series_description_raw"] = long_df["series_code"].map(code_to_description)
    return long_df


def run_cleaning(
    excel_path: Path,
    output_dir: Path,
    report_path: Path,
    log_path: Path,
) -> CleaningOutputs:
    logger = build_logger(log_path)
    logger.info("Starting Phase 2 cleaning")
    logger.info("Reading source file: %s", excel_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    long_df = load_excel_monthly_to_long(excel_path)
    logger.info("Raw long rows: %s", len(long_df))

    long_df["source_dataset"] = excel_path.name
    long_df["series_description"] = long_df["series_description_raw"].astype(str).map(normalize_text)

    parsed = long_df["series_description_raw"].map(parse_series_metadata)
    long_df[["mineral", "departamento", "unidad"]] = pd.DataFrame(parsed.tolist(), index=long_df.index)

    long_df["period_raw"] = long_df["period_raw"].astype(str).map(normalize_text)
    long_df["period_date"] = long_df["period_raw"].map(parse_period)
    long_df["year"] = long_df["period_date"].dt.year
    long_df["month"] = long_df["period_date"].dt.month

    numeric_parsed = long_df["value_raw"].map(parse_numeric)
    numeric_df = pd.DataFrame(numeric_parsed.tolist(), columns=["value", "is_missing", "is_invalid_numeric"])
    long_df = pd.concat([long_df, numeric_df], axis=1)

    before_dedup = len(long_df)
    long_df = long_df.drop_duplicates(subset=["period_date", "series_code"], keep="first")
    duplicates_removed = before_dedup - len(long_df)

    # Base dataset: remove semantic missing values.
    base_df = long_df[~long_df["is_missing"]].copy()

    ordered_columns = [
        "source_dataset",
        "period_raw",
        "period_date",
        "year",
        "month",
        "series_code",
        "series_description",
        "mineral",
        "departamento",
        "unidad",
        "value",
        "is_missing",
        "is_invalid_numeric",
        "value_raw",
    ]
    long_df = long_df[ordered_columns]
    base_df = base_df[ordered_columns]

    full_output = output_dir / "mineria_mensual_long_clean_full.csv"
    base_output = output_dir / "mineria_mensual_long_clean_base.csv"
    long_df.to_csv(full_output, index=False, encoding="utf-8-sig")
    base_df.to_csv(base_output, index=False, encoding="utf-8-sig")

    report = {
        "phase": "Fase 2 - Limpieza automatizada",
        "source_file": str(excel_path),
        "rows_raw_long": int(before_dedup),
        "rows_after_dedup": int(len(long_df)),
        "duplicates_removed": int(duplicates_removed),
        "rows_missing_semantic": int(long_df["is_missing"].sum()),
        "rows_invalid_numeric": int(long_df["is_invalid_numeric"].sum()),
        "rows_base_dataset": int(len(base_df)),
        "null_period_date": int(long_df["period_date"].isna().sum()),
        "null_mineral": int(long_df["mineral"].isna().sum()),
        "null_departamento": int(long_df["departamento"].isna().sum()),
        "unique_periods": int(long_df["period_date"].nunique(dropna=True)),
        "unique_series_codes": int(long_df["series_code"].nunique(dropna=True)),
        "unique_minerals": sorted([x for x in long_df["mineral"].dropna().unique().tolist()]),
        "output_full": str(full_output),
        "output_base": str(base_output),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved full cleaned dataset: %s", full_output)
    logger.info("Saved base cleaned dataset: %s", base_output)
    logger.info("Saved cleaning report: %s", report_path)
    logger.info("Phase 2 cleaning finished")

    return CleaningOutputs(
        full_output_path=full_output,
        base_output_path=base_output,
        report_path=report_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 automated cleaning.")
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=Path("data/raw/DatosProduccionMinera.xlsx"),
        help="Path to canonical Excel source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
        help="Output directory for cleaned datasets.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase2_cleaning_report.json"),
        help="Output path for quality report.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase2_cleaning.log"),
        help="Path to cleaning log file.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    run_cleaning(
        excel_path=args.excel_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        log_path=args.log_path,
    )
    return 0
