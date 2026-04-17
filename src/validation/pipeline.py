from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
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

MANDATORY_NON_NULL_COLUMNS = [
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
]

ALLOWED_UNITS = {"tm.f", "kg.f", "grs.f"}


@dataclass
class RuleResult:
    rule_id: str
    description: str
    status: str
    detail: str


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase3_validation")
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


def _rule(rule_id: str, description: str, passed: bool, detail: str) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        description=description,
        status="PASS" if passed else "FAIL",
        detail=detail,
    )


def validate_dataset(df: pd.DataFrame) -> list[RuleResult]:
    results: list[RuleResult] = []

    # R01: Schema columns.
    expected_set = set(EXPECTED_COLUMNS)
    current_set = set(df.columns.tolist())
    missing_cols = sorted(list(expected_set - current_set))
    extra_cols = sorted(list(current_set - expected_set))
    schema_ok = not missing_cols and not extra_cols
    detail_schema = f"missing={missing_cols} extra={extra_cols}"
    results.append(_rule("R01", "Columns match expected schema", schema_ok, detail_schema))

    # R02: Dataset non-empty.
    non_empty_ok = len(df) > 0
    results.append(_rule("R02", "Dataset has at least one row", non_empty_ok, f"rows={len(df)}"))

    # Remaining rules require required columns to exist.
    if not expected_set.issubset(current_set):
        return results

    # R03: Mandatory columns non-null.
    null_counts = {col: int(df[col].isna().sum()) for col in MANDATORY_NON_NULL_COLUMNS}
    mandatory_ok = all(count == 0 for count in null_counts.values())
    results.append(_rule("R03", "Mandatory columns have no nulls", mandatory_ok, json.dumps(null_counts, ensure_ascii=False)))

    # R04: period_date parseable.
    parsed_period = pd.to_datetime(df["period_date"], errors="coerce")
    invalid_period_count = int(parsed_period.isna().sum())
    results.append(_rule("R04", "period_date is valid datetime", invalid_period_count == 0, f"invalid_period_date={invalid_period_count}"))

    # R05: year/month consistent with period_date.
    year_num = pd.to_numeric(df["year"], errors="coerce")
    month_num = pd.to_numeric(df["month"], errors="coerce")
    ym_consistency = (
        parsed_period.notna()
        & year_num.notna()
        & month_num.notna()
        & (parsed_period.dt.year == year_num.astype(int))
        & (parsed_period.dt.month == month_num.astype(int))
    )
    ym_bad = int((~ym_consistency).sum())
    results.append(_rule("R05", "year/month are consistent with period_date", ym_bad == 0, f"inconsistent_rows={ym_bad}"))

    # R06: month in [1, 12].
    month_valid = month_num.between(1, 12, inclusive="both")
    month_bad = int((~month_valid).sum())
    results.append(_rule("R06", "month value is between 1 and 12", month_bad == 0, f"invalid_month_rows={month_bad}"))

    # R07: value numeric and non-negative.
    value_num = pd.to_numeric(df["value"], errors="coerce")
    value_nan = int(value_num.isna().sum())
    value_negative = int((value_num < 0).sum())
    value_ok = value_nan == 0 and value_negative == 0
    results.append(_rule("R07", "value is numeric and non-negative", value_ok, f"nan_values={value_nan} negative_values={value_negative}"))

    # R08: Unique key period_date + series_code.
    duplicates = int(df.duplicated(subset=["period_date", "series_code"]).sum())
    results.append(_rule("R08", "Primary key (period_date, series_code) is unique", duplicates == 0, f"duplicate_keys={duplicates}"))

    # R09: Unit domain.
    units_found = set(df["unidad"].dropna().astype(str).unique().tolist())
    invalid_units = sorted(list(units_found - ALLOWED_UNITS))
    results.append(_rule("R09", "unidad belongs to allowed domain", len(invalid_units) == 0, f"invalid_units={invalid_units}"))

    # R10: Base dataset policy checks.
    missing_true = int(df["is_missing"].astype(str).str.lower().eq("true").sum())
    invalid_numeric_true = int(df["is_invalid_numeric"].astype(str).str.lower().eq("true").sum())
    base_policy_ok = missing_true == 0 and invalid_numeric_true == 0
    results.append(
        _rule(
            "R10",
            "Base dataset has no rows flagged as missing or invalid numeric",
            base_policy_ok,
            f"is_missing_true={missing_true} is_invalid_numeric_true={invalid_numeric_true}",
        )
    )

    return results


def run_validation(
    input_path: Path,
    report_path: Path,
    failures_path: Path,
    log_path: Path,
) -> int:
    logger = build_logger(log_path)
    logger.info("Starting Phase 3 schema and quality validation")
    logger.info("Input dataset: %s", input_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    rules = validate_dataset(df)
    overall_pass = all(rule.status == "PASS" for rule in rules)

    failures_df = pd.DataFrame([asdict(rule) for rule in rules if rule.status == "FAIL"])
    failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")

    report = {
        "phase": "Fase 3 - Validacion de esquema y calidad",
        "input_dataset": str(input_path),
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "rules": [asdict(rule) for rule in rules],
        "failures_output": str(failures_path),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Validation overall status: %s", report["overall_status"])
    logger.info("Saved validation report: %s", report_path)
    logger.info("Saved validation failures: %s", failures_path)
    logger.info("Phase 3 validation finished")
    return 0 if overall_pass else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 schema and quality validation.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/interim/mineria_mensual_long_clean_base.csv"),
        help="Path to cleaned base dataset from Phase 2.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase3_validation_report.json"),
        help="Path to JSON validation report.",
    )
    parser.add_argument(
        "--failures-path",
        type=Path,
        default=Path("reports/phase3_validation_failures.csv"),
        help="Path to CSV with failed rules.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase3_validation.log"),
        help="Path to validation log file.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    return run_validation(
        input_path=args.input_path,
        report_path=args.report_path,
        failures_path=args.failures_path,
        log_path=args.log_path,
    )
