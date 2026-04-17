from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def configure_report_style() -> None:
    """Set a consistent, presentation-ready style for all figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "figure.dpi": 130,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.2,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _prepare_time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=0)


def _finalize_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_historical_vs_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots()

    ax.plot(
        historical_df["date"],
        historical_df["actual"],
        linestyle="-",
        label="Historico",
    )
    ax.plot(
        forecast_df["date"],
        forecast_df["predicted"],
        linestyle="--",
        marker="o",
        markersize=3,
        label="Prediccion",
    )

    if not forecast_df.empty:
        forecast_start = pd.to_datetime(forecast_df["date"]).min()
        ax.axvline(forecast_start, linestyle=":", linewidth=1.3, label="Inicio forecast")

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    _prepare_time_axis(ax)
    _finalize_figure(fig, output_path)


def plot_residuals(
    evaluation_df: pd.DataFrame,
    output_path: Path,
    title: str,
    y_label: str = "Error (real - predicho)",
) -> None:
    eval_df = evaluation_df.copy()
    eval_df["residual"] = eval_df["actual"] - eval_df["predicted"]

    fig, ax = plt.subplots()
    ax.plot(
        eval_df["date"],
        eval_df["residual"],
        linestyle="-",
        marker="o",
        markersize=4,
        label="Residual",
    )
    ax.axhline(0.0, linestyle=":", linewidth=1.2, label="Error cero")

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    _prepare_time_axis(ax)
    _finalize_figure(fig, output_path)


def plot_trend(
    evaluation_df: pd.DataFrame,
    output_path: Path,
    title: str,
    window: int = 3,
) -> None:
    eval_df = evaluation_df.copy().sort_values("date").reset_index(drop=True)
    eval_df["actual_trend"] = eval_df["actual"].rolling(window=window, min_periods=1).mean()
    eval_df["predicted_trend"] = eval_df["predicted"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots()
    ax.plot(eval_df["date"], eval_df["actual_trend"], linestyle="-", label=f"Real suavizado ({window}m)")
    ax.plot(
        eval_df["date"],
        eval_df["predicted_trend"],
        linestyle="--",
        marker="o",
        markersize=3,
        label=f"Predicho suavizado ({window}m)",
    )

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Produccion suavizada")
    ax.legend(loc="best")
    _prepare_time_axis(ax)
    _finalize_figure(fig, output_path)


def plot_category_lines(
    category_df: pd.DataFrame,
    categories: Iterable[str],
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots()
    for category in categories:
        subset = category_df[category_df["category"] == category].sort_values("date")
        ax.plot(subset["date"], subset["value"], linestyle="-", marker="o", markersize=2.5, label=str(category))

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left", ncols=2)
    _prepare_time_axis(ax)
    _finalize_figure(fig, output_path)
