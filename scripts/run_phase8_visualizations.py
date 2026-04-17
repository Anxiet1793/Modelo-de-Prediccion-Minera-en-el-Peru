from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.visualization import (
    configure_report_style,
    plot_category_lines,
    plot_historical_vs_forecast,
    plot_residuals,
    plot_trend,
)


def _load_features(path: Path, level: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"period_date", "target_value", "series_id", "mineral", "unidad"}
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(set(df.columns))))
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if "departamento" not in df.columns:
        df["departamento"] = "Total"

    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df = df.dropna(subset=["period_date", "target_value", "series_id"]).copy()
    df["level"] = level
    return df


def _load_operational(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "level",
        "series_id",
        "mineral",
        "departamento",
        "unidad",
        "selected_model",
        "forecast_period_date",
        "forecast_value",
    }
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(set(df.columns))))
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["forecast_period_date"] = pd.to_datetime(df["forecast_period_date"], errors="coerce")
    df["forecast_value"] = pd.to_numeric(df["forecast_value"], errors="coerce")
    return df.dropna(subset=["forecast_period_date", "forecast_value", "series_id"]).copy()


def _load_evaluation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"level", "series_id", "model", "period_date", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(set(df.columns))))
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    return df.dropna(subset=["period_date", "y_true", "y_pred"]).copy()


def _pick_series(operational_df: pd.DataFrame, requested_series: str | None, level: str) -> str:
    candidates = operational_df[operational_df["level"] == level].copy()
    if candidates.empty:
        raise ValueError(f"No operational forecasts found for level={level}")

    if requested_series:
        if (candidates["series_id"] == requested_series).any():
            return requested_series
        raise ValueError(f"Requested series_id not found in operational forecasts: {requested_series}")

    ranking = (
        candidates.groupby("series_id", as_index=False)
        .agg(total_forecast=("forecast_value", "sum"))
        .sort_values("total_forecast", ascending=False)
    )
    return str(ranking.iloc[0]["series_id"])


def _series_context(operational_series_df: pd.DataFrame) -> tuple[str, str, str, str]:
    first_row = operational_series_df.iloc[0]
    return (
        str(first_row["selected_model"]),
        str(first_row["mineral"]),
        str(first_row["departamento"]),
        str(first_row["unidad"]),
    )


def run_visualizations(
    eval_predictions_path: Path,
    operational_forecast_path: Path,
    national_features_path: Path,
    department_features_path: Path,
    output_dir: Path,
    level: str,
    series_id: str | None,
    top_n_categories: int,
    trend_window: int,
    report_path: Path,
) -> None:
    configure_report_style()

    eval_df = _load_evaluation(eval_predictions_path)
    operational_df = _load_operational(operational_forecast_path)
    national_features = _load_features(national_features_path, level="nacional_mineral")
    department_features = _load_features(department_features_path, level="departamento_mineral")

    features_df = department_features if level == "departamento_mineral" else national_features
    selected_series_id = _pick_series(operational_df, requested_series=series_id, level=level)

    operational_series = operational_df[
        (operational_df["level"] == level) & (operational_df["series_id"] == selected_series_id)
    ].sort_values("forecast_period_date")
    if operational_series.empty:
        raise ValueError(f"No operational forecast rows found for series_id={selected_series_id} and level={level}")

    selected_model, mineral, departamento, unidad = _series_context(operational_series)

    historical_series = features_df[features_df["series_id"] == selected_series_id].sort_values("period_date")
    if historical_series.empty:
        raise ValueError(f"No historical feature rows found for series_id={selected_series_id}")

    main_historical_df = historical_series.rename(columns={"period_date": "date", "target_value": "actual"})[
        ["date", "actual"]
    ]
    main_forecast_df = operational_series.rename(
        columns={"forecast_period_date": "date", "forecast_value": "predicted"}
    )[["date", "predicted"]]

    eval_series_model = eval_df[
        (eval_df["level"] == level) & (eval_df["series_id"] == selected_series_id) & (eval_df["model"] == selected_model)
    ].sort_values("period_date")

    if eval_series_model.empty:
        fallback = (
            eval_df[(eval_df["level"] == level) & (eval_df["series_id"] == selected_series_id)]
            .groupby("model", as_index=False)
            .agg(mae=("y_true", lambda x: float("nan")))
        )
        raise ValueError(
            "No evaluation rows found for selected series and model. "
            f"series={selected_series_id}, model={selected_model}, fallback_rows={len(fallback)}"
        )

    eval_plot_df = eval_series_model.rename(
        columns={"period_date": "date", "y_true": "actual", "y_pred": "predicted"}
    )[["date", "actual", "predicted"]]

    if level == "departamento_mineral":
        category_source = operational_df[operational_df["departamento"] == departamento].copy()
        category_col = "mineral"
        category_title = f"Forecast 2026 por mineral | {departamento} (Top {top_n_categories})"
    else:
        category_source = operational_df[operational_df["level"] == "nacional_mineral"].copy()
        category_col = "mineral"
        category_title = f"Forecast 2026 nacional por mineral (Top {top_n_categories})"

    category_rank = (
        category_source.groupby(category_col, as_index=False)
        .agg(total_value=("forecast_value", "sum"))
        .sort_values("total_value", ascending=False)
        .head(top_n_categories)
    )
    top_categories = category_rank[category_col].astype(str).tolist()

    category_plot_df = (
        category_source[category_source[category_col].isin(top_categories)]
        .rename(columns={"forecast_period_date": "date", "forecast_value": "value", category_col: "category"})
        [["date", "category", "value"]]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    main_chart_path = output_dir / "01_historico_vs_prediccion.png"
    error_chart_path = output_dir / "02_residuales_backtest.png"
    trend_chart_path = output_dir / "03_tendencia_suavizada.png"
    category_chart_path = output_dir / "04_forecast_por_categoria.png"

    plot_historical_vs_forecast(
        historical_df=main_historical_df,
        forecast_df=main_forecast_df,
        output_path=main_chart_path,
        title=(
            f"Produccion historica vs prediccion | {mineral} | {departamento} "
            f"| Modelo: {selected_model}"
        ),
        y_label=f"Produccion ({unidad})",
    )
    plot_residuals(
        evaluation_df=eval_plot_df,
        output_path=error_chart_path,
        title=f"Residuales de backtest | {selected_series_id} | Modelo: {selected_model}",
    )
    plot_trend(
        evaluation_df=eval_plot_df,
        output_path=trend_chart_path,
        title=(
            f"Tendencia suavizada (rolling={trend_window}) | {selected_series_id} "
            f"| Modelo: {selected_model}"
        ),
        window=trend_window,
    )
    plot_category_lines(
        category_df=category_plot_df,
        categories=top_categories,
        output_path=category_chart_path,
        title=category_title,
        y_label="Produccion forecast",
    )

    report = {
        "level": level,
        "series_id": selected_series_id,
        "selected_model": selected_model,
        "mineral": mineral,
        "departamento": departamento,
        "unidad": unidad,
        "trend_window": trend_window,
        "top_categories": top_categories,
        "charts": {
            "historical_vs_forecast": str(main_chart_path),
            "residuals": str(error_chart_path),
            "trend": str(trend_chart_path),
            "category": str(category_chart_path),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate professional matplotlib charts for phase 8 forecasting.")
    parser.add_argument(
        "--eval-predictions-path",
        type=Path,
        default=Path("data/processed/mineria_phase8_eval_predictions.csv"),
        help="Evaluation predictions file with y_true and y_pred.",
    )
    parser.add_argument(
        "--operational-forecast-path",
        type=Path,
        default=Path("data/processed/mineria_phase8_operational_forecast_2026.csv"),
        help="Operational forecast file.",
    )
    parser.add_argument(
        "--national-features-path",
        type=Path,
        default=Path("data/processed/mineria_features_nacional_mineral.csv"),
        help="Historical national-level features file.",
    )
    parser.add_argument(
        "--department-features-path",
        type=Path,
        default=Path("data/processed/mineria_features_departamento_mineral.csv"),
        help="Historical department-level features file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures/phase8_visualizations"),
        help="Directory to store generated charts.",
    )
    parser.add_argument(
        "--level",
        choices=["nacional_mineral", "departamento_mineral"],
        default="departamento_mineral",
        help="Series level to visualize.",
    )
    parser.add_argument(
        "--series-id",
        type=str,
        default="Cobre|Ancash|tm.f",
        help="Target series id. If omitted, top forecast series is selected.",
    )
    parser.add_argument(
        "--top-n-categories",
        type=int,
        default=6,
        help="Top categories for category chart.",
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=3,
        help="Rolling window for trend chart.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase8_visualization_report.json"),
        help="JSON report output path.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    run_visualizations(
        eval_predictions_path=args.eval_predictions_path,
        operational_forecast_path=args.operational_forecast_path,
        national_features_path=args.national_features_path,
        department_features_path=args.department_features_path,
        output_dir=args.output_dir,
        level=args.level,
        series_id=args.series_id,
        top_n_categories=args.top_n_categories,
        trend_window=args.trend_window,
        report_path=args.report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_from_cli())
