from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase6_eda")
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


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def classify_gap(abs_pct_gap: float | None, ok_max_abs: float, warning_max_abs: float) -> str:
    if abs_pct_gap is None or pd.isna(abs_pct_gap):
        return "critical"
    if abs_pct_gap <= ok_max_abs:
        return "ok"
    if abs_pct_gap <= warning_max_abs:
        return "warning"
    return "critical"


def compute_iqr_outlier_count(df: pd.DataFrame, group_cols: list[str], value_col: str) -> int:
    if df.empty:
        return 0

    outlier_count = 0
    grouped = df.groupby(group_cols, dropna=False)

    for _, group_df in grouped:
        sample = group_df[value_col]
        if sample.empty:
            continue
        q1 = sample.quantile(0.25)
        q3 = sample.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count += int(((sample < lower_bound) | (sample > upper_bound)).sum())

    return outlier_count


def save_plot_trend_official_tmf(official: pd.DataFrame, output_path: Path) -> bool:
    tmf = official[official["unidad"] == "tm.f"].copy()
    if tmf.empty:
        return False

    trend = (
        tmf.groupby(["period_date", "mineral"], as_index=False)
        .agg(value=("value", "sum"))
        .sort_values(["period_date", "mineral"])
    )

    pivot = trend.pivot(index="period_date", columns="mineral", values="value").sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for mineral in pivot.columns:
        ax.plot(pivot.index, pivot[mineral], marker="o", linewidth=1.8, label=str(mineral))

    ax.set_title("Official monthly production by mineral (tm.f)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Production")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def save_plot_gap_by_mineral(agg_mineral: pd.DataFrame, output_path: Path) -> bool:
    if agg_mineral.empty:
        return False

    data = agg_mineral.copy()
    data["abs_pct_gap"] = data["pct_gap_vs_total"].abs()

    top = (
        data.groupby("mineral", as_index=False)
        .agg(avg_abs_pct_gap=("abs_pct_gap", "mean"))
        .sort_values("avg_abs_pct_gap", ascending=False)
        .head(12)
    )

    if top.empty:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(top["mineral"].astype(str), top["avg_abs_pct_gap"], color="#2E86AB")
    ax.set_title("Average absolute gap vs official total by mineral")
    ax.set_xlabel("Mineral")
    ax.set_ylabel("Average absolute pct gap")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def save_plot_top_departamentos_tmf(detail: pd.DataFrame, output_path: Path) -> bool:
    tmf = detail[detail["unidad"] == "tm.f"].copy()
    if tmf.empty:
        return False

    ranking = (
        tmf.groupby("departamento", as_index=False)
        .agg(total_value=("value", "sum"))
        .sort_values("total_value", ascending=False)
        .head(10)
    )

    if ranking.empty:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(ranking["departamento"].astype(str), ranking["total_value"], color="#F18F01")
    ax.set_title("Top 10 departamentos by cumulative production (tm.f)")
    ax.set_xlabel("Departamento")
    ax.set_ylabel("Cumulative production")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def build_findings_markdown(findings: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Fase 6 - EDA Findings",
        "",
        "## Coverage",
        f"- Period range: {findings['period_range']}",
        f"- Number of periods: {findings['n_periods']}",
        f"- Rows analyzed (tidy): {findings['rows_tidy']}",
        "",
        "## Main signals",
        f"- Highest cumulative official mineral-unit: {findings['top_mineral_unit']['label']} ({findings['top_mineral_unit']['value']:.6f})",
        f"- Departamento with highest cumulative detail volume (tm.f): {findings['top_departamento_tmf']['label']} ({findings['top_departamento_tmf']['value']:.6f})",
        f"- Mineral with highest average absolute gap: {findings['top_gap_mineral']['label']} ({findings['top_gap_mineral']['value']:.6f}%)",
        "",
        "## Data quality indicators",
        f"- Gap quality flags: ok={findings['gap_flags']['ok']}, warning={findings['gap_flags']['warning']}, critical={findings['gap_flags']['critical']}",
        f"- IQR outliers in detail series (mineral, unidad): {findings['n_outliers_detail_iqr']}",
        "",
        "## Notes",
        "- Units are intentionally kept separate (grs.f, kg.f, tm.f).",
        "- Aggregations are read from Phase 4 outputs and not recomputed against raw files.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eda(
    tidy_path: Path,
    agg_mineral_path: Path,
    agg_dept_path: Path,
    report_path: Path,
    findings_path: Path,
    figures_dir: Path,
    log_path: Path,
    gap_ok_threshold: float,
    gap_warning_threshold: float,
) -> None:
    logger = build_logger(log_path)
    logger.info("Starting Phase 6 EDA")

    if gap_warning_threshold <= gap_ok_threshold:
        raise ValueError("gap_warning_threshold must be greater than gap_ok_threshold")

    tidy = pd.read_csv(tidy_path)
    agg_mineral = pd.read_csv(agg_mineral_path)
    agg_dept = pd.read_csv(agg_dept_path)

    tidy["period_date"] = pd.to_datetime(tidy["period_date"], errors="coerce")
    agg_mineral["period_date"] = pd.to_datetime(agg_mineral["period_date"], errors="coerce")
    agg_dept["period_date"] = pd.to_datetime(agg_dept["period_date"], errors="coerce")
    tidy["is_total_departamento"] = parse_bool_series(tidy["is_total_departamento"])

    official = tidy[tidy["is_total_departamento"]].copy()
    detail = tidy[~tidy["is_total_departamento"]].copy()

    agg_mineral["abs_pct_gap"] = agg_mineral["pct_gap_vs_total"].abs()
    agg_mineral["gap_quality_flag"] = agg_mineral["abs_pct_gap"].apply(
        lambda x: classify_gap(x, gap_ok_threshold, gap_warning_threshold)
    )

    period_min = tidy["period_date"].min()
    period_max = tidy["period_date"].max()
    n_periods = int(tidy["period_yyyymm"].nunique())

    top_mineral_unit_row = (
        official.groupby(["mineral", "unidad"], as_index=False)
        .agg(total_value=("value", "sum"))
        .sort_values("total_value", ascending=False)
        .head(1)
    )
    if top_mineral_unit_row.empty:
        top_mineral_unit = {"label": "n/a", "value": 0.0}
    else:
        mineral = str(top_mineral_unit_row.iloc[0]["mineral"])
        unidad = str(top_mineral_unit_row.iloc[0]["unidad"])
        value = float(top_mineral_unit_row.iloc[0]["total_value"])
        top_mineral_unit = {"label": f"{mineral} ({unidad})", "value": value}

    top_departamento_tmf_row = (
        detail[detail["unidad"] == "tm.f"]
        .groupby("departamento", as_index=False)
        .agg(total_value=("value", "sum"))
        .sort_values("total_value", ascending=False)
        .head(1)
    )
    if top_departamento_tmf_row.empty:
        top_departamento_tmf = {"label": "n/a", "value": 0.0}
    else:
        top_departamento_tmf = {
            "label": str(top_departamento_tmf_row.iloc[0]["departamento"]),
            "value": float(top_departamento_tmf_row.iloc[0]["total_value"]),
        }

    top_gap_mineral_row = (
        agg_mineral.groupby("mineral", as_index=False)
        .agg(avg_abs_pct_gap=("abs_pct_gap", "mean"))
        .sort_values("avg_abs_pct_gap", ascending=False)
        .head(1)
    )
    if top_gap_mineral_row.empty:
        top_gap_mineral = {"label": "n/a", "value": 0.0}
    else:
        top_gap_mineral = {
            "label": str(top_gap_mineral_row.iloc[0]["mineral"]),
            "value": float(top_gap_mineral_row.iloc[0]["avg_abs_pct_gap"]),
        }

    gap_counts = agg_mineral["gap_quality_flag"].value_counts(dropna=False).to_dict()
    gap_flags = {
        "ok": int(gap_counts.get("ok", 0)),
        "warning": int(gap_counts.get("warning", 0)),
        "critical": int(gap_counts.get("critical", 0)),
    }

    n_outliers_detail_iqr = compute_iqr_outlier_count(
        df=detail,
        group_cols=["mineral", "unidad"],
        value_col="value",
    )

    figures: dict[str, str] = {}
    trend_path = figures_dir / "phase6_trend_official_tmf.png"
    gap_path = figures_dir / "phase6_gap_avg_by_mineral.png"
    dept_path = figures_dir / "phase6_top_departamentos_tmf.png"

    if save_plot_trend_official_tmf(official, trend_path):
        figures["trend_official_tmf"] = str(trend_path)
    if save_plot_gap_by_mineral(agg_mineral, gap_path):
        figures["gap_avg_by_mineral"] = str(gap_path)
    if save_plot_top_departamentos_tmf(detail, dept_path):
        figures["top_departamentos_tmf"] = str(dept_path)

    findings_payload = {
        "period_range": f"{period_min.date() if pd.notna(period_min) else 'n/a'} -> {period_max.date() if pd.notna(period_max) else 'n/a'}",
        "n_periods": n_periods,
        "rows_tidy": int(len(tidy)),
        "top_mineral_unit": top_mineral_unit,
        "top_departamento_tmf": top_departamento_tmf,
        "top_gap_mineral": top_gap_mineral,
        "gap_flags": gap_flags,
        "n_outliers_detail_iqr": n_outliers_detail_iqr,
    }

    build_findings_markdown(findings_payload, findings_path)

    report = {
        "phase": "Fase 6 - EDA",
        "inputs": {
            "tidy_path": str(tidy_path),
            "agg_mineral_path": str(agg_mineral_path),
            "agg_dept_path": str(agg_dept_path),
        },
        "rows": {
            "tidy": int(len(tidy)),
            "official": int(len(official)),
            "detail": int(len(detail)),
            "agg_mineral": int(len(agg_mineral)),
            "agg_dept": int(len(agg_dept)),
        },
        "coverage": {
            "period_min": period_min.strftime("%Y-%m-%d") if pd.notna(period_min) else None,
            "period_max": period_max.strftime("%Y-%m-%d") if pd.notna(period_max) else None,
            "n_periods": n_periods,
            "n_minerales": int(tidy["mineral"].nunique()),
            "n_departamentos": int(detail["departamento"].nunique()),
            "units": sorted(tidy["unidad"].dropna().astype(str).unique().tolist()),
        },
        "quality": {
            "gap_thresholds_pct": {
                "ok_max_abs": gap_ok_threshold,
                "warning_max_abs": gap_warning_threshold,
            },
            "gap_flags": gap_flags,
            "n_outliers_detail_iqr": n_outliers_detail_iqr,
        },
        "signals": {
            "top_mineral_unit": top_mineral_unit,
            "top_departamento_tmf": top_departamento_tmf,
            "top_gap_mineral": top_gap_mineral,
        },
        "artifacts": {
            "findings_markdown": str(findings_path),
            "figures": figures,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved EDA report: %s", report_path)
    logger.info("Saved findings markdown: %s", findings_path)
    logger.info("Saved %s figure(s) in: %s", len(figures), figures_dir)
    logger.info("Phase 6 EDA finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 6 EDA pipeline.")
    parser.add_argument(
        "--tidy-path",
        type=Path,
        default=Path("data/processed/mineria_mensual_tidy.csv"),
        help="Path to tidy dataset from Phase 4.",
    )
    parser.add_argument(
        "--agg-mineral-path",
        type=Path,
        default=Path("data/processed/mineria_agg_mineral_period.csv"),
        help="Path to mineral aggregate dataset from Phase 4.",
    )
    parser.add_argument(
        "--agg-dept-path",
        type=Path,
        default=Path("data/processed/mineria_agg_departamento_period.csv"),
        help="Path to department aggregate dataset from Phase 4.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase6_eda_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--findings-path",
        type=Path,
        default=Path("reports/phase6_eda_findings.md"),
        help="Output markdown findings path.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures/phase6"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase6_eda.log"),
        help="Output log path.",
    )
    parser.add_argument(
        "--gap-ok-threshold",
        type=float,
        default=float(os.getenv("GAP_OK_THRESHOLD_PCT", "1")),
        help="Absolute pct gap threshold for flag=ok.",
    )
    parser.add_argument(
        "--gap-warning-threshold",
        type=float,
        default=float(os.getenv("GAP_WARNING_THRESHOLD_PCT", "5")),
        help="Absolute pct gap threshold for flag=warning.",
    )
    return parser


def run_from_cli() -> int:
    args = build_parser().parse_args()
    run_eda(
        tidy_path=args.tidy_path,
        agg_mineral_path=args.agg_mineral_path,
        agg_dept_path=args.agg_dept_path,
        report_path=args.report_path,
        findings_path=args.findings_path,
        figures_dir=args.figures_dir,
        log_path=args.log_path,
        gap_ok_threshold=args.gap_ok_threshold,
        gap_warning_threshold=args.gap_warning_threshold,
    )
    return 0
