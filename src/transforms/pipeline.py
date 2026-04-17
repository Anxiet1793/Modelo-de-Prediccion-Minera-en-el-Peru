from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


MONTH_NAME_ES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase4_transforms")
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


def build_variable_classification() -> pd.DataFrame:
    rows = [
        ("record_id", "technical_key", "Llave tecnica por periodo y serie."),
        ("source_dataset", "technical", "Archivo fuente canonico utilizado."),
        ("period_raw", "temporal_raw", "Token temporal original en formato Abr24."),
        ("period_date", "temporal", "Fecha normalizada al primer dia de mes."),
        ("period_yyyymm", "temporal", "Representacion YYYY-MM para ordenamiento."),
        ("year", "temporal", "Anio calendario."),
        ("month", "temporal", "Mes numerico 1..12."),
        ("month_name_es", "temporal", "Nombre de mes en espanol."),
        ("quarter", "temporal", "Trimestre calendario 1..4."),
        ("semester", "temporal", "Semestre calendario 1..2."),
        ("series_code", "dimension_identifier", "Codigo oficial de serie estadistica."),
        ("series_description", "dimension_descriptor", "Descripcion oficial de la serie."),
        ("mineral", "dimension", "Producto mineral."),
        ("departamento", "dimension", "Ambito geografico departamental o Total."),
        ("unidad", "dimension", "Unidad de medida original, no convertida en Fase 4."),
        ("is_total_departamento", "flag", "Marca registros con departamento igual a Total."),
        ("area_geografica", "dimension", "Clasificacion Nacional o Departamental."),
        ("metric_level", "dimension", "Detalle departamental o total oficial."),
        ("value", "metric", "Valor numerico de produccion."),
        ("value_rounded_6", "metric", "Valor redondeado a 6 decimales para consistencia tecnica."),
        ("is_missing", "flag", "Marca de faltante semantico heredada de limpieza."),
        ("is_invalid_numeric", "flag", "Marca de conversion numerica invalida heredada de limpieza."),
        ("value_raw", "technical_raw", "Valor textual original antes de parseo numerico."),
    ]
    return pd.DataFrame(rows, columns=["column_name", "role", "description"])


def run_transforms(
    input_path: Path,
    output_dir: Path,
    report_path: Path,
    classification_path: Path,
    log_path: Path,
) -> None:
    logger = build_logger(log_path)
    logger.info("Starting Phase 4 transformation and classification")
    logger.info("Input dataset: %s", input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    classification_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")

    # Enriched analytical table without changing measurement units.
    tidy = df.copy()
    tidy["period_yyyymm"] = tidy["period_date"].dt.strftime("%Y-%m")
    tidy["month_name_es"] = tidy["month"].map(MONTH_NAME_ES)
    tidy["quarter"] = tidy["period_date"].dt.quarter
    tidy["semester"] = (((tidy["month"] - 1) // 6) + 1).astype(int)
    tidy["is_total_departamento"] = tidy["departamento"].astype(str).str.lower().eq("total")
    tidy["area_geografica"] = np.where(tidy["is_total_departamento"], "Nacional", "Departamental")
    tidy["metric_level"] = np.where(tidy["is_total_departamento"], "total_oficial", "detalle_departamental")
    tidy["value_rounded_6"] = pd.to_numeric(tidy["value"], errors="coerce").round(6)
    tidy["record_id"] = tidy["period_yyyymm"].astype(str) + "|" + tidy["series_code"].astype(str)

    tidy = tidy.sort_values(["period_date", "mineral", "departamento", "series_code"]).reset_index(drop=True)

    # Aggregation 1: sum of departmental detail and comparison against official total.
    detail = tidy[~tidy["is_total_departamento"]].copy()
    total = tidy[tidy["is_total_departamento"]].copy()

    agg_detail_mineral = (
        detail.groupby(
            ["period_date", "period_raw", "period_yyyymm", "year", "month", "quarter", "semester", "mineral", "unidad"],
            as_index=False,
        )
        .agg(
            dept_sum_value=("value", "sum"),
            n_departamentos=("departamento", "nunique"),
        )
    )

    agg_total_mineral = (
        total.groupby(
            ["period_date", "period_raw", "period_yyyymm", "year", "month", "quarter", "semester", "mineral", "unidad"],
            as_index=False,
        )
        .agg(official_total_value=("value", "first"))
    )

    agg_mineral_period = agg_detail_mineral.merge(
        agg_total_mineral,
        on=["period_date", "period_raw", "period_yyyymm", "year", "month", "quarter", "semester", "mineral", "unidad"],
        how="left",
    )
    agg_mineral_period["abs_gap_vs_total"] = agg_mineral_period["official_total_value"] - agg_mineral_period["dept_sum_value"]
    agg_mineral_period["pct_gap_vs_total"] = np.where(
        agg_mineral_period["official_total_value"].abs() > 0,
        (agg_mineral_period["abs_gap_vs_total"] / agg_mineral_period["official_total_value"]) * 100,
        np.nan,
    )

    # Aggregation 2: department-period summary by unit (no cross-unit mixing).
    agg_departamento_period = (
        detail.groupby(
            ["period_date", "period_raw", "period_yyyymm", "year", "month", "quarter", "semester", "departamento", "unidad"],
            as_index=False,
        )
        .agg(
            dept_unit_sum_value=("value", "sum"),
            n_minerales=("mineral", "nunique"),
        )
    )

    # Outputs.
    tidy_path = output_dir / "mineria_mensual_tidy.csv"
    mineral_path = output_dir / "mineria_agg_mineral_period.csv"
    dept_path = output_dir / "mineria_agg_departamento_period.csv"

    tidy.to_csv(tidy_path, index=False, encoding="utf-8-sig")
    agg_mineral_period.to_csv(mineral_path, index=False, encoding="utf-8-sig")
    agg_departamento_period.to_csv(dept_path, index=False, encoding="utf-8-sig")

    classification_df = build_variable_classification()
    classification_df.to_csv(classification_path, index=False, encoding="utf-8-sig")

    report = {
        "phase": "Fase 4 - Transformacion y clasificacion",
        "input_dataset": str(input_path),
        "rows_input": int(len(df)),
        "rows_tidy": int(len(tidy)),
        "rows_agg_mineral_period": int(len(agg_mineral_period)),
        "rows_agg_departamento_period": int(len(agg_departamento_period)),
        "unique_periods": int(tidy["period_date"].nunique()),
        "unique_minerales": int(tidy["mineral"].nunique()),
        "unique_departamentos": int(tidy["departamento"].nunique()),
        "units_detected": sorted(tidy["unidad"].dropna().astype(str).unique().tolist()),
        "added_columns_tidy": [
            "period_yyyymm",
            "month_name_es",
            "quarter",
            "semester",
            "is_total_departamento",
            "area_geografica",
            "metric_level",
            "value_rounded_6",
            "record_id",
        ],
        "output_tidy": str(tidy_path),
        "output_agg_mineral_period": str(mineral_path),
        "output_agg_departamento_period": str(dept_path),
        "output_variable_classification": str(classification_path),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved tidy dataset: %s", tidy_path)
    logger.info("Saved mineral aggregation: %s", mineral_path)
    logger.info("Saved department aggregation: %s", dept_path)
    logger.info("Saved variable classification: %s", classification_path)
    logger.info("Saved transform report: %s", report_path)
    logger.info("Phase 4 transformation finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 4 transformation and classification.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/interim/mineria_mensual_long_clean_base.csv"),
        help="Input base dataset from Phase 2/3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed datasets.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase4_transform_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--classification-path",
        type=Path,
        default=Path("reports/phase4_variable_classification.csv"),
        help="Output CSV for variable classification.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase4_transforms.log"),
        help="Output log path.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    run_transforms(
        input_path=args.input_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        classification_path=args.classification_path,
        log_path=args.log_path,
    )
    return 0
