from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet

    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase8_forecasting")
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


def load_feature_table(path: Path, level: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"period_date", "series_id", "target_value", "mineral", "unidad"}
    if not required.issubset(set(df.columns)):
        missing = sorted(list(required.difference(set(df.columns))))
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["level"] = level

    if "departamento" not in df.columns:
        df["departamento"] = "Total"

    df = df.dropna(subset=["period_date", "series_id", "target_value"]).copy()
    df = df.sort_values(["series_id", "period_date"]).reset_index(drop=True)
    return df


def split_train_test(series_df: pd.DataFrame, horizon: int, min_train_size: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if len(series_df) < (horizon + min_train_size):
        return None

    train_df = series_df.iloc[:-horizon].copy()
    test_df = series_df.iloc[-horizon:].copy()
    if train_df.empty or test_df.empty:
        return None
    return train_df, test_df


def future_periods(last_period: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
    return list(pd.date_range(last_period + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"))


def forecast_naive(train_values: pd.Series, horizon: int) -> np.ndarray:
    last_value = float(train_values.iloc[-1])
    return np.array([last_value] * horizon, dtype=float)


def forecast_moving_average(train_values: pd.Series, horizon: int, window: int = 3) -> np.ndarray:
    tail = train_values.tail(window)
    value = float(tail.mean())
    return np.array([value] * horizon, dtype=float)


def forecast_sarima(train_values: pd.Series, horizon: int) -> np.ndarray:
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels is not available")

    if len(train_values) < 24:
        raise RuntimeError("series too short for SARIMA baseline")

    model = SARIMAX(
        train_values.astype(float).values,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=horizon).predicted_mean
    return np.asarray(forecast, dtype=float)


def forecast_prophet(train_dates: pd.Series, train_values: pd.Series, horizon: int) -> np.ndarray:
    if not HAS_PROPHET:
        raise RuntimeError("prophet is not available")

    prophet_input = pd.DataFrame({"ds": pd.to_datetime(train_dates), "y": train_values.astype(float).values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_input)

    future = model.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
    pred = model.predict(future)
    return pred["yhat"].astype(float).values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    abs_error = np.abs(y_true - y_pred)
    sq_error = (y_true - y_pred) ** 2

    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(sq_error)))

    valid_mape = y_true != 0
    if np.any(valid_mape):
        mape = float(np.mean(np.abs((y_true[valid_mape] - y_pred[valid_mape]) / y_true[valid_mape])) * 100)
    else:
        mape = float("nan")

    denom = np.abs(y_true) + np.abs(y_pred)
    valid_smape = denom != 0
    if np.any(valid_smape):
        smape = float(np.mean((2 * np.abs(y_true[valid_smape] - y_pred[valid_smape]) / denom[valid_smape])) * 100)
    else:
        smape = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
    }


def run_model_forecast(
    model_name: str,
    train_df: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    values = train_df["target_value"]

    if model_name == "naive":
        return forecast_naive(values, horizon)
    if model_name == "moving_average":
        return forecast_moving_average(values, horizon=horizon, window=3)
    if model_name == "sarima":
        return forecast_sarima(values, horizon)
    if model_name == "prophet":
        return forecast_prophet(train_df["period_date"], values, horizon)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_table(
    df: pd.DataFrame,
    level: str,
    horizon: int,
    min_train_size: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = ["naive", "moving_average", "sarima", "prophet"]

    prediction_rows: list[dict] = []
    series_metric_rows: list[dict] = []
    future_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for series_id, group in df.groupby("series_id", sort=False):
        group = group.sort_values("period_date").reset_index(drop=True)

        split = split_train_test(group, horizon=horizon, min_train_size=min_train_size)
        if split is None:
            skipped_rows.append(
                {
                    "level": level,
                    "series_id": series_id,
                    "model": "all",
                    "reason": f"insufficient_history(len={len(group)})",
                }
            )
            continue

        train_df, test_df = split
        y_true = test_df["target_value"].astype(float).values

        for model_name in models:
            try:
                y_pred = run_model_forecast(model_name=model_name, train_df=train_df, horizon=horizon)
                if len(y_pred) != horizon:
                    raise RuntimeError(f"invalid_forecast_length={len(y_pred)}")
            except Exception as error:
                skipped_rows.append(
                    {
                        "level": level,
                        "series_id": series_id,
                        "model": model_name,
                        "reason": str(error),
                    }
                )
                continue

            metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
            context = {
                "level": level,
                "series_id": series_id,
                "mineral": str(group.iloc[0]["mineral"]),
                "departamento": str(group.iloc[0]["departamento"]),
                "unidad": str(group.iloc[0]["unidad"]),
                "model": model_name,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_last_period": train_df["period_date"].max().strftime("%Y-%m-%d"),
            }

            series_metric_rows.append({**context, **metrics})

            for i in range(horizon):
                prediction_rows.append(
                    {
                        **context,
                        "horizon_step": i + 1,
                        "period_date": test_df.iloc[i]["period_date"].strftime("%Y-%m-%d"),
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                        "abs_error": float(abs(y_true[i] - y_pred[i])),
                    }
                )

            # Forecast next 3 months using full history for practical use.
            try:
                y_future = run_model_forecast(model_name=model_name, train_df=group, horizon=horizon)
                next_dates = future_periods(group["period_date"].max(), horizon=horizon)
                for i in range(horizon):
                    future_rows.append(
                        {
                            "level": level,
                            "series_id": series_id,
                            "mineral": str(group.iloc[0]["mineral"]),
                            "departamento": str(group.iloc[0]["departamento"]),
                            "unidad": str(group.iloc[0]["unidad"]),
                            "model": model_name,
                            "horizon_step": i + 1,
                            "forecast_period_date": next_dates[i].strftime("%Y-%m-%d"),
                            "forecast_value": float(y_future[i]),
                            "train_last_period": group["period_date"].max().strftime("%Y-%m-%d"),
                        }
                    )
            except Exception as error:
                skipped_rows.append(
                    {
                        "level": level,
                        "series_id": series_id,
                        "model": f"{model_name}_future",
                        "reason": str(error),
                    }
                )

    pred_df = pd.DataFrame(prediction_rows)
    series_metrics_df = pd.DataFrame(series_metric_rows)
    future_df = pd.DataFrame(future_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    logger.info(
        "Level=%s finished: series=%s predictions=%s series_metrics=%s skipped=%s",
        level,
        df["series_id"].nunique(),
        len(pred_df),
        len(series_metrics_df),
        len(skipped_df),
    )

    return pred_df, series_metrics_df, future_df, skipped_df


def build_overall_metrics(series_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if series_metrics_df.empty:
        return pd.DataFrame(
            columns=["level", "model", "n_series", "mae_mean", "rmse_mean", "mape_mean", "smape_mean"]
        )

    grouped = (
        series_metrics_df.groupby(["level", "model"], as_index=False)
        .agg(
            n_series=("series_id", "nunique"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            smape_mean=("smape", "mean"),
        )
        .sort_values(["level", "mae_mean", "rmse_mean"])
        .reset_index(drop=True)
    )
    return grouped


def build_best_model_by_level(overall_df: pd.DataFrame) -> list[dict]:
    if overall_df.empty:
        return []

    best_rows = []
    for level, group in overall_df.groupby("level", sort=False):
        top = group.sort_values(["mae_mean", "rmse_mean"]).iloc[0]
        best_rows.append(
            {
                "level": level,
                "best_model_by_mae": str(top["model"]),
                "mae_mean": float(top["mae_mean"]),
                "rmse_mean": float(top["rmse_mean"]),
                "mape_mean": float(top["mape_mean"]),
                "smape_mean": float(top["smape_mean"]),
            }
        )
    return best_rows


def run_forecasting(
    national_features_path: Path,
    department_features_path: Path,
    predictions_path: Path,
    future_forecasts_path: Path,
    metrics_by_series_path: Path,
    leaderboard_path: Path,
    skipped_path: Path,
    report_path: Path,
    log_path: Path,
    horizon: int,
    min_train_size: int,
) -> None:
    logger = build_logger(log_path)
    logger.info("Starting Phase 8 forecasting baselines")

    national_df = load_feature_table(national_features_path, level="nacional_mineral")
    department_df = load_feature_table(department_features_path, level="departamento_mineral")

    nat_pred, nat_series_metrics, nat_future, nat_skipped = evaluate_table(
        national_df,
        level="nacional_mineral",
        horizon=horizon,
        min_train_size=min_train_size,
        logger=logger,
    )
    dep_pred, dep_series_metrics, dep_future, dep_skipped = evaluate_table(
        department_df,
        level="departamento_mineral",
        horizon=horizon,
        min_train_size=min_train_size,
        logger=logger,
    )

    pred_df = pd.concat([nat_pred, dep_pred], ignore_index=True)
    series_metrics_df = pd.concat([nat_series_metrics, dep_series_metrics], ignore_index=True)
    future_df = pd.concat([nat_future, dep_future], ignore_index=True)
    skipped_df = pd.concat([nat_skipped, dep_skipped], ignore_index=True)

    overall_df = build_overall_metrics(series_metrics_df)
    best_by_level = build_best_model_by_level(overall_df)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_by_series_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    future_df.to_csv(future_forecasts_path, index=False, encoding="utf-8-sig")
    series_metrics_df.to_csv(metrics_by_series_path, index=False, encoding="utf-8-sig")
    overall_df.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    report = {
        "phase": "Fase 8 - Forecasting baselines",
        "settings": {
            "horizon": horizon,
            "min_train_size": min_train_size,
            "models_requested": ["naive", "moving_average", "sarima", "prophet"],
            "dependency_status": {
                "statsmodels_available": HAS_STATSMODELS,
                "prophet_available": HAS_PROPHET,
            },
        },
        "inputs": {
            "national_features_path": str(national_features_path),
            "department_features_path": str(department_features_path),
        },
        "rows": {
            "predictions": int(len(pred_df)),
            "future_forecasts": int(len(future_df)),
            "series_metrics": int(len(series_metrics_df)),
            "leaderboard_rows": int(len(overall_df)),
            "skipped_rows": int(len(skipped_df)),
        },
        "series_coverage": {
            "national_series": int(national_df["series_id"].nunique()),
            "department_series": int(department_df["series_id"].nunique()),
        },
        "best_model_by_level": best_by_level,
        "outputs": {
            "predictions_path": str(predictions_path),
            "future_forecasts_path": str(future_forecasts_path),
            "metrics_by_series_path": str(metrics_by_series_path),
            "leaderboard_path": str(leaderboard_path),
            "skipped_path": str(skipped_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved predictions: %s", predictions_path)
    logger.info("Saved future forecasts: %s", future_forecasts_path)
    logger.info("Saved metrics by series: %s", metrics_by_series_path)
    logger.info("Saved leaderboard: %s", leaderboard_path)
    logger.info("Saved skipped report: %s", skipped_path)
    logger.info("Saved phase report: %s", report_path)
    logger.info("Phase 8 forecasting baselines finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 8 baseline forecasting.")
    parser.add_argument(
        "--national-features-path",
        type=Path,
        default=Path("data/processed/mineria_features_nacional_mineral.csv"),
        help="Features file for national mineral level.",
    )
    parser.add_argument(
        "--department-features-path",
        type=Path,
        default=Path("data/processed/mineria_features_departamento_mineral.csv"),
        help="Features file for department-mineral level.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("data/processed/mineria_phase8_eval_predictions.csv"),
        help="Output CSV for evaluation predictions.",
    )
    parser.add_argument(
        "--future-forecasts-path",
        type=Path,
        default=Path("data/processed/mineria_phase8_future_forecasts.csv"),
        help="Output CSV for next 3-month forecasts.",
    )
    parser.add_argument(
        "--metrics-by-series-path",
        type=Path,
        default=Path("reports/phase8_metrics_by_series.csv"),
        help="Output CSV for metrics by model and series.",
    )
    parser.add_argument(
        "--leaderboard-path",
        type=Path,
        default=Path("reports/phase8_model_leaderboard.csv"),
        help="Output CSV for model leaderboard by level.",
    )
    parser.add_argument(
        "--skipped-path",
        type=Path,
        default=Path("reports/phase8_skipped_series.csv"),
        help="Output CSV with skipped model-series pairs and reasons.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase8_forecast_report.json"),
        help="Output JSON summary report.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase8_forecasting.log"),
        help="Log file path.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forecast horizon in months for evaluation and future forecasts.",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=24,
        help="Minimum number of training observations required per series.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    run_forecasting(
        national_features_path=args.national_features_path,
        department_features_path=args.department_features_path,
        predictions_path=args.predictions_path,
        future_forecasts_path=args.future_forecasts_path,
        metrics_by_series_path=args.metrics_by_series_path,
        leaderboard_path=args.leaderboard_path,
        skipped_path=args.skipped_path,
        report_path=args.report_path,
        log_path=args.log_path,
        horizon=args.horizon,
        min_train_size=args.min_train_size,
    )
    return 0
