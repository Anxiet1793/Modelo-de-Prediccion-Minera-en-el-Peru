from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from src.models.pipeline import load_feature_table, run_model_forecast


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase8_operational")
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


def select_best_models_by_level(leaderboard_path: Path) -> dict[str, str]:
    leaderboard = pd.read_csv(leaderboard_path)
    required_cols = {"level", "model", "mae_mean"}
    if not required_cols.issubset(set(leaderboard.columns)):
        missing = sorted(list(required_cols.difference(set(leaderboard.columns))))
        raise ValueError(f"Missing required leaderboard columns: {missing}")

    selected: dict[str, str] = {}
    for level, group in leaderboard.groupby("level", sort=False):
        top = group.sort_values(["mae_mean", "rmse_mean" if "rmse_mean" in group.columns else "mae_mean"]).iloc[0]
        selected[str(level)] = str(top["model"])

    return selected


def select_best_models_by_series(
    metrics_by_series_path: Path,
    mode: Literal["series_mae", "series_mae_dynamic"],
    dynamic_tolerance_pct: float,
) -> dict[tuple[str, str], str]:
    metrics = pd.read_csv(metrics_by_series_path)
    required_cols = {"level", "series_id", "model", "mae", "rmse"}
    if not required_cols.issubset(set(metrics.columns)):
        missing = sorted(list(required_cols.difference(set(metrics.columns))))
        raise ValueError(f"Missing required metrics columns: {missing}")

    metrics = metrics.copy()
    metrics["mae"] = pd.to_numeric(metrics["mae"], errors="coerce")
    metrics["rmse"] = pd.to_numeric(metrics["rmse"], errors="coerce")
    metrics["mae"] = metrics["mae"].fillna(float("inf"))
    metrics["rmse"] = metrics["rmse"].fillna(float("inf"))

    series_selection: dict[tuple[str, str], str] = {}
    dynamic_models = {"sarima", "prophet"}
    flat_models = {"naive", "moving_average"}

    for (level, series_id), group in metrics.groupby(["level", "series_id"], sort=False):
        ranked = group.sort_values(["mae", "rmse"]).reset_index(drop=True)
        best = ranked.iloc[0]
        selected_model = str(best["model"])

        if mode == "series_mae_dynamic" and selected_model in flat_models:
            dynamic_ranked = ranked[ranked["model"].isin(dynamic_models)].sort_values(["mae", "rmse"]).reset_index(
                drop=True
            )
            if not dynamic_ranked.empty:
                best_dynamic = dynamic_ranked.iloc[0]
                dynamic_mae = float(best_dynamic["mae"])
                best_mae = float(best["mae"])
                max_acceptable = best_mae * (1.0 + dynamic_tolerance_pct / 100.0)
                if dynamic_mae <= max_acceptable:
                    selected_model = str(best_dynamic["model"])

        series_selection[(str(level), str(series_id))] = selected_model

    return series_selection


def build_operational_forecasts_for_level(
    df: pd.DataFrame,
    level: str,
    model_name_fallback: str,
    series_model_map: dict[tuple[str, str], str],
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    min_train_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for series_id, group in df.groupby("series_id", sort=False):
        group = group.sort_values("period_date").reset_index(drop=True)
        if len(group) < min_train_size:
            skipped_rows.append(
                {
                    "level": level,
                    "series_id": series_id,
                    "model": model_name_fallback,
                    "reason": f"insufficient_history(len={len(group)})",
                }
            )
            continue

        last_period = pd.to_datetime(group["period_date"].max())
        all_future_dates = pd.date_range(last_period + pd.offsets.MonthBegin(1), forecast_end, freq="MS")
        if len(all_future_dates) == 0:
            skipped_rows.append(
                {
                    "level": level,
                    "series_id": series_id,
                    "model": model_name_fallback,
                    "reason": "no_future_window",
                }
            )
            continue

        selected_model = series_model_map.get((level, str(series_id)), model_name_fallback)

        try:
            y_pred = run_model_forecast(model_name=selected_model, train_df=group, horizon=len(all_future_dates))
        except Exception as error:
            if selected_model != model_name_fallback:
                try:
                    y_pred = run_model_forecast(
                        model_name=model_name_fallback,
                        train_df=group,
                        horizon=len(all_future_dates),
                    )
                    selected_model = model_name_fallback
                except Exception as fallback_error:
                    skipped_rows.append(
                        {
                            "level": level,
                            "series_id": series_id,
                            "model": selected_model,
                            "reason": f"primary={error}; fallback={fallback_error}",
                        }
                    )
                    continue
            else:
                skipped_rows.append(
                    {
                        "level": level,
                        "series_id": series_id,
                        "model": selected_model,
                        "reason": str(error),
                    }
                )
                continue

        for i, future_date in enumerate(all_future_dates):
            if future_date < forecast_start:
                continue
            forecast_rows.append(
                {
                    "level": level,
                    "series_id": series_id,
                    "mineral": str(group.iloc[0]["mineral"]),
                    "departamento": str(group.iloc[0]["departamento"]),
                    "unidad": str(group.iloc[0]["unidad"]),
                    "selected_model": selected_model,
                    "last_observed_period": last_period.strftime("%Y-%m-%d"),
                    "forecast_period_date": future_date.strftime("%Y-%m-%d"),
                    "forecast_period_yyyymm": future_date.strftime("%Y-%m"),
                    "forecast_value": float(y_pred[i]),
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    return forecast_df, skipped_df


def build_executive_table(operational_df: pd.DataFrame) -> pd.DataFrame:
    if operational_df.empty:
        return pd.DataFrame()

    pivot = operational_df.pivot_table(
        index=[
            "level",
            "series_id",
            "mineral",
            "departamento",
            "unidad",
            "selected_model",
            "last_observed_period",
        ],
        columns="forecast_period_yyyymm",
        values="forecast_value",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None

    fixed_months = pd.date_range("2026-02-01", "2026-12-01", freq="MS").strftime("%Y-%m").tolist()
    for month_col in fixed_months:
        if month_col not in pivot.columns:
            pivot[month_col] = pd.NA

    ordered_cols = [
        "level",
        "series_id",
        "mineral",
        "departamento",
        "unidad",
        "selected_model",
        "last_observed_period",
        *fixed_months,
    ]
    pivot = pivot[ordered_cols].sort_values(["level", "series_id"]).reset_index(drop=True)
    return pivot


def run_operational_forecast(
    leaderboard_path: Path,
    metrics_by_series_path: Path,
    national_features_path: Path,
    department_features_path: Path,
    operational_long_path: Path,
    operational_executive_path: Path,
    skipped_path: Path,
    report_path: Path,
    log_path: Path,
    min_train_size: int,
    selection_mode: Literal["level_mae", "series_mae", "series_mae_dynamic"],
    dynamic_tolerance_pct: float,
) -> None:
    logger = build_logger(log_path)
    logger.info("Starting Phase 8 operational forecast generation")

    selected_models_level = select_best_models_by_level(leaderboard_path)
    if "nacional_mineral" not in selected_models_level or "departamento_mineral" not in selected_models_level:
        raise ValueError("Leaderboard must contain both levels: nacional_mineral and departamento_mineral")

    if selection_mode == "level_mae":
        selected_models_series: dict[tuple[str, str], str] = {}
    elif selection_mode in {"series_mae", "series_mae_dynamic"}:
        selected_models_series = select_best_models_by_series(
            metrics_by_series_path=metrics_by_series_path,
            mode=selection_mode,
            dynamic_tolerance_pct=dynamic_tolerance_pct,
        )
    else:
        raise ValueError(f"Unsupported selection mode: {selection_mode}")

    logger.info("Selection mode: %s", selection_mode)
    logger.info("Fallback models by level: %s", selected_models_level)
    if selection_mode != "level_mae":
        logger.info("Series-level selected model count: %s", len(selected_models_series))

    national_df = load_feature_table(national_features_path, level="nacional_mineral")
    department_df = load_feature_table(department_features_path, level="departamento_mineral")

    forecast_start = pd.Timestamp("2026-02-01")
    forecast_end = pd.Timestamp("2026-12-01")

    nat_forecast, nat_skipped = build_operational_forecasts_for_level(
        national_df,
        level="nacional_mineral",
        model_name_fallback=selected_models_level["nacional_mineral"],
        series_model_map=selected_models_series,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        min_train_size=min_train_size,
    )
    dep_forecast, dep_skipped = build_operational_forecasts_for_level(
        department_df,
        level="departamento_mineral",
        model_name_fallback=selected_models_level["departamento_mineral"],
        series_model_map=selected_models_series,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        min_train_size=min_train_size,
    )

    operational_df = pd.concat([nat_forecast, dep_forecast], ignore_index=True)
    skipped_df = pd.concat([nat_skipped, dep_skipped], ignore_index=True)

    operational_df = operational_df.sort_values(["level", "series_id", "forecast_period_date"]).reset_index(drop=True)
    executive_df = build_executive_table(operational_df)

    operational_long_path.parent.mkdir(parents=True, exist_ok=True)
    operational_executive_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    operational_df.to_csv(operational_long_path, index=False, encoding="utf-8-sig")
    executive_df.to_csv(operational_executive_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    report = {
        "phase": "Fase 8 - Operational forecast freeze",
        "selection_rule": {
            "mode": selection_mode,
            "fallback": "best mae per level from phase8_model_leaderboard.csv",
            "series_source": str(metrics_by_series_path),
            "dynamic_tolerance_pct": dynamic_tolerance_pct,
        },
        "forecast_window": {
            "start": forecast_start.strftime("%Y-%m-%d"),
            "end": forecast_end.strftime("%Y-%m-%d"),
            "months": len(pd.date_range(forecast_start, forecast_end, freq="MS")),
        },
        "selected_models_level": selected_models_level,
        "rows": {
            "operational_long": int(len(operational_df)),
            "operational_executive": int(len(executive_df)),
            "skipped": int(len(skipped_df)),
        },
        "model_usage": (
            operational_df.groupby(["level", "selected_model"], as_index=False)
            .agg(series_count=("series_id", "nunique"))
            .to_dict(orient="records")
            if not operational_df.empty
            else []
        ),
        "series_coverage": {
            "nacional_mineral": int(operational_df[operational_df["level"] == "nacional_mineral"]["series_id"].nunique())
            if not operational_df.empty
            else 0,
            "departamento_mineral": int(
                operational_df[operational_df["level"] == "departamento_mineral"]["series_id"].nunique()
            )
            if not operational_df.empty
            else 0,
        },
        "outputs": {
            "operational_long_path": str(operational_long_path),
            "operational_executive_path": str(operational_executive_path),
            "skipped_path": str(skipped_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved operational long table: %s", operational_long_path)
    logger.info("Saved executive table: %s", operational_executive_path)
    logger.info("Saved skipped table: %s", skipped_path)
    logger.info("Saved operational report: %s", report_path)
    logger.info("Phase 8 operational forecast generation finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze best model per level and build operational 2026 forecasts.")
    parser.add_argument(
        "--leaderboard-path",
        type=Path,
        default=Path("reports/phase8_model_leaderboard.csv"),
        help="Leaderboard generated by phase 8 baseline evaluation.",
    )
    parser.add_argument(
        "--metrics-by-series-path",
        type=Path,
        default=Path("reports/phase8_metrics_by_series.csv"),
        help="Metrics by series generated by phase 8 baseline evaluation.",
    )
    parser.add_argument(
        "--national-features-path",
        type=Path,
        default=Path("data/processed/mineria_features_nacional_mineral.csv"),
        help="National features input path.",
    )
    parser.add_argument(
        "--department-features-path",
        type=Path,
        default=Path("data/processed/mineria_features_departamento_mineral.csv"),
        help="Department-mineral features input path.",
    )
    parser.add_argument(
        "--operational-long-path",
        type=Path,
        default=Path("data/processed/mineria_phase8_operational_forecast_2026.csv"),
        help="Output path for long-format operational forecast.",
    )
    parser.add_argument(
        "--operational-executive-path",
        type=Path,
        default=Path("reports/phase8_operational_executive_forecast_2026.csv"),
        help="Output path for executive pivot table.",
    )
    parser.add_argument(
        "--skipped-path",
        type=Path,
        default=Path("reports/phase8_operational_skipped_series.csv"),
        help="Output path for skipped series.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase8_operational_report.json"),
        help="Output path for operational report.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase8_operational.log"),
        help="Output path for operational log.",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=24,
        help="Minimum history required by series.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["level_mae", "series_mae", "series_mae_dynamic"],
        default="series_mae_dynamic",
        help="Model selection strategy for operational forecast.",
    )
    parser.add_argument(
        "--dynamic-tolerance-pct",
        type=float,
        default=10.0,
        help="In series_mae_dynamic, allows dynamic model if mae is within this pct of best mae.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    run_operational_forecast(
        leaderboard_path=args.leaderboard_path,
        metrics_by_series_path=args.metrics_by_series_path,
        national_features_path=args.national_features_path,
        department_features_path=args.department_features_path,
        operational_long_path=args.operational_long_path,
        operational_executive_path=args.operational_executive_path,
        skipped_path=args.skipped_path,
        report_path=args.report_path,
        log_path=args.log_path,
        min_train_size=args.min_train_size,
        selection_mode=args.selection_mode,
        dynamic_tolerance_pct=args.dynamic_tolerance_pct,
    )
    return 0
