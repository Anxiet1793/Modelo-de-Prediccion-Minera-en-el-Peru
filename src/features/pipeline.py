from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase7_features")
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


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["month_sin"] = np.sin((2 * np.pi * output["month"]) / 12)
    output["month_cos"] = np.cos((2 * np.pi * output["month"]) / 12)
    output["quarter_sin"] = np.sin((2 * np.pi * output["quarter"]) / 4)
    output["quarter_cos"] = np.cos((2 * np.pi * output["quarter"]) / 4)
    output["is_year_start"] = output["month"].eq(1)
    output["is_year_end"] = output["month"].eq(12)
    return output


def add_group_time_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "target_value",
) -> pd.DataFrame:
    output = df.sort_values(group_cols + ["period_date"]).copy()
    grouped = output.groupby(group_cols, dropna=False)

    output["series_index"] = grouped.cumcount().astype(int)

    for lag in [1, 2, 3, 6, 12]:
        output[f"lag_{lag}"] = grouped[target_col].shift(lag)

    output["diff_lag_1"] = output[target_col] - output["lag_1"]
    output["diff_lag_12"] = output[target_col] - output["lag_12"]

    output["mom_pct"] = np.where(output["lag_1"] > 0, (output[target_col] / output["lag_1"]) - 1.0, np.nan)
    output["yoy_pct"] = np.where(output["lag_12"] > 0, (output[target_col] / output["lag_12"]) - 1.0, np.nan)

    for window in [3, 6, 12]:
        output[f"roll_mean_{window}"] = grouped[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        output[f"roll_std_{window}"] = grouped[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
        )
        output[f"roll_min_{window}"] = grouped[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).min()
        )
        output[f"roll_max_{window}"] = grouped[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).max()
        )

    for horizon in [1, 2, 3]:
        output[f"target_t_plus_{horizon}"] = grouped[target_col].shift(-horizon)

    output["ready_for_model_h1"] = output["lag_12"].notna() & output["target_t_plus_1"].notna()
    output["ready_for_model_h3"] = output["lag_12"].notna() & output["target_t_plus_3"].notna()
    return output


def build_national_mineral_dataset(tidy: pd.DataFrame) -> pd.DataFrame:
    total = tidy[tidy["is_total_departamento"]].copy()
    grouped = (
        total.groupby(
            ["period_date", "period_yyyymm", "year", "month", "quarter", "semester", "mineral", "unidad"],
            as_index=False,
        )
        .agg(target_value=("value", "first"))
        .sort_values(["mineral", "unidad", "period_date"])
    )
    grouped["series_id"] = grouped["mineral"].astype(str) + "|" + grouped["unidad"].astype(str)
    return grouped


def build_departamento_mineral_dataset(tidy: pd.DataFrame) -> pd.DataFrame:
    detail = tidy[~tidy["is_total_departamento"]].copy()
    grouped = (
        detail.groupby(
            [
                "period_date",
                "period_yyyymm",
                "year",
                "month",
                "quarter",
                "semester",
                "mineral",
                "departamento",
                "unidad",
            ],
            as_index=False,
        )
        .agg(target_value=("value", "sum"))
        .sort_values(["mineral", "departamento", "unidad", "period_date"])
    )
    grouped["series_id"] = (
        grouped["mineral"].astype(str)
        + "|"
        + grouped["departamento"].astype(str)
        + "|"
        + grouped["unidad"].astype(str)
    )
    return grouped


def build_feature_dictionary() -> pd.DataFrame:
    rows = [
        ("period_date", "temporal", "Fecha normalizada del periodo mensual."),
        ("period_yyyymm", "temporal", "Representacion de periodo YYYY-MM."),
        ("series_id", "identifier", "Llave de serie temporal."),
        ("target_value", "target", "Valor objetivo de produccion mensual."),
        ("month_sin", "calendar_feature", "Componente seno del mes para estacionalidad."),
        ("month_cos", "calendar_feature", "Componente coseno del mes para estacionalidad."),
        ("quarter_sin", "calendar_feature", "Componente seno del trimestre."),
        ("quarter_cos", "calendar_feature", "Componente coseno del trimestre."),
        ("is_year_start", "calendar_feature", "Marca enero."),
        ("is_year_end", "calendar_feature", "Marca diciembre."),
        ("series_index", "time_index", "Posicion temporal incremental por serie."),
        ("lag_1", "lag_feature", "Valor de la serie en t-1."),
        ("lag_2", "lag_feature", "Valor de la serie en t-2."),
        ("lag_3", "lag_feature", "Valor de la serie en t-3."),
        ("lag_6", "lag_feature", "Valor de la serie en t-6."),
        ("lag_12", "lag_feature", "Valor de la serie en t-12."),
        ("diff_lag_1", "difference_feature", "Diferencia target - lag_1."),
        ("diff_lag_12", "difference_feature", "Diferencia target - lag_12."),
        ("mom_pct", "growth_feature", "Variacion mensual relativa contra t-1."),
        ("yoy_pct", "growth_feature", "Variacion interanual relativa contra t-12."),
        ("roll_mean_3", "rolling_feature", "Promedio movil de 3 meses en ventana historica."),
        ("roll_mean_6", "rolling_feature", "Promedio movil de 6 meses en ventana historica."),
        ("roll_mean_12", "rolling_feature", "Promedio movil de 12 meses en ventana historica."),
        ("roll_std_3", "rolling_feature", "Desviacion estandar movil de 3 meses."),
        ("roll_std_6", "rolling_feature", "Desviacion estandar movil de 6 meses."),
        ("roll_std_12", "rolling_feature", "Desviacion estandar movil de 12 meses."),
        ("roll_min_3", "rolling_feature", "Minimo movil de 3 meses."),
        ("roll_min_6", "rolling_feature", "Minimo movil de 6 meses."),
        ("roll_min_12", "rolling_feature", "Minimo movil de 12 meses."),
        ("roll_max_3", "rolling_feature", "Maximo movil de 3 meses."),
        ("roll_max_6", "rolling_feature", "Maximo movil de 6 meses."),
        ("roll_max_12", "rolling_feature", "Maximo movil de 12 meses."),
        ("target_t_plus_1", "future_target", "Target desplazado a t+1."),
        ("target_t_plus_2", "future_target", "Target desplazado a t+2."),
        ("target_t_plus_3", "future_target", "Target desplazado a t+3."),
        ("ready_for_model_h1", "training_flag", "Fila util para entrenamiento de horizonte 1."),
        ("ready_for_model_h3", "training_flag", "Fila util para entrenamiento de horizonte 3."),
    ]
    return pd.DataFrame(rows, columns=["feature_name", "feature_type", "description"])


def run_feature_engineering(
    tidy_path: Path,
    output_dir: Path,
    report_path: Path,
    dictionary_path: Path,
    log_path: Path,
) -> None:
    logger = build_logger(log_path)
    logger.info("Starting Phase 7 feature engineering")
    logger.info("Input tidy dataset: %s", tidy_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)

    tidy = pd.read_csv(tidy_path)
    tidy["period_date"] = pd.to_datetime(tidy["period_date"], errors="coerce")
    tidy["is_total_departamento"] = tidy["is_total_departamento"].astype(str).str.lower().eq("true")

    national = build_national_mineral_dataset(tidy)
    national = add_calendar_features(national)
    national = add_group_time_features(national, group_cols=["mineral", "unidad"], target_col="target_value")

    dept_mineral = build_departamento_mineral_dataset(tidy)
    dept_mineral = add_calendar_features(dept_mineral)
    dept_mineral = add_group_time_features(
        dept_mineral,
        group_cols=["mineral", "departamento", "unidad"],
        target_col="target_value",
    )

    national_path = output_dir / "mineria_features_nacional_mineral.csv"
    dept_path = output_dir / "mineria_features_departamento_mineral.csv"

    national.to_csv(national_path, index=False, encoding="utf-8-sig")
    dept_mineral.to_csv(dept_path, index=False, encoding="utf-8-sig")

    dictionary_df = build_feature_dictionary()
    dictionary_df.to_csv(dictionary_path, index=False, encoding="utf-8-sig")

    report = {
        "phase": "Fase 7 - Feature engineering",
        "input_tidy_path": str(tidy_path),
        "rows_input_tidy": int(len(tidy)),
        "outputs": {
            "national_features_path": str(national_path),
            "department_mineral_features_path": str(dept_path),
            "feature_dictionary_path": str(dictionary_path),
        },
        "national": {
            "rows": int(len(national)),
            "n_series": int(national["series_id"].nunique()),
            "ready_h1_rows": int(national["ready_for_model_h1"].sum()),
            "ready_h3_rows": int(national["ready_for_model_h3"].sum()),
            "period_min": national["period_date"].min().strftime("%Y-%m-%d") if not national.empty else None,
            "period_max": national["period_date"].max().strftime("%Y-%m-%d") if not national.empty else None,
        },
        "department_mineral": {
            "rows": int(len(dept_mineral)),
            "n_series": int(dept_mineral["series_id"].nunique()),
            "ready_h1_rows": int(dept_mineral["ready_for_model_h1"].sum()),
            "ready_h3_rows": int(dept_mineral["ready_for_model_h3"].sum()),
            "period_min": dept_mineral["period_date"].min().strftime("%Y-%m-%d") if not dept_mineral.empty else None,
            "period_max": dept_mineral["period_date"].max().strftime("%Y-%m-%d") if not dept_mineral.empty else None,
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved national features: %s", national_path)
    logger.info("Saved department-mineral features: %s", dept_path)
    logger.info("Saved feature dictionary: %s", dictionary_path)
    logger.info("Saved feature report: %s", report_path)
    logger.info("Phase 7 feature engineering finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 7 feature engineering.")
    parser.add_argument(
        "--tidy-path",
        type=Path,
        default=Path("data/processed/mineria_mensual_tidy.csv"),
        help="Input tidy dataset from Phase 4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for feature tables.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase7_feature_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--dictionary-path",
        type=Path,
        default=Path("reports/phase7_feature_dictionary.csv"),
        help="Output CSV feature dictionary path.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase7_features.log"),
        help="Output log path.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    run_feature_engineering(
        tidy_path=args.tidy_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        dictionary_path=args.dictionary_path,
        log_path=args.log_path,
    )
    return 0
