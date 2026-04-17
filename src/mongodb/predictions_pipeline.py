from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, UpdateOne
from pymongo.collection import Collection

from src.mongodb.pipeline import build_mongo_client, ensure_collection, execute_bulk_upserts


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase9_mongodb_predictions")
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


def _to_optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _to_optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def load_operational_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "level",
        "series_id",
        "mineral",
        "departamento",
        "unidad",
        "selected_model",
        "last_observed_period",
        "forecast_period_date",
        "forecast_period_yyyymm",
        "forecast_value",
    }
    if not required.issubset(set(df.columns)):
        missing = sorted(list(required.difference(set(df.columns))))
        raise ValueError(f"Missing required columns in operational forecast file: {missing}")

    output = df.copy()
    output["last_observed_period"] = pd.to_datetime(output["last_observed_period"], errors="coerce")
    output["forecast_period_date"] = pd.to_datetime(output["forecast_period_date"], errors="coerce")
    output["forecast_value"] = pd.to_numeric(output["forecast_value"], errors="coerce")
    output = output.dropna(subset=["selected_model", "level", "series_id", "forecast_period_date"]).copy()
    output = output.sort_values(["selected_model", "level", "series_id", "forecast_period_date"]).reset_index(drop=True)
    return output


def _parse_selection_metadata(operational_report_path: Path, logger: logging.Logger) -> dict[str, Any]:
    if not operational_report_path.exists():
        logger.warning("Operational report file not found, source metadata will be partial: %s", operational_report_path)
        return {}

    try:
        with operational_report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except Exception as error:
        logger.warning("Could not parse operational report file %s: %s", operational_report_path, error)
        return {}

    selection_rule = report.get("selection_rule", {}) if isinstance(report, dict) else {}
    output: dict[str, Any] = {
        "selection_mode": selection_rule.get("mode"),
        "dynamic_tolerance_pct": selection_rule.get("dynamic_tolerance_pct"),
    }
    return {key: value for key, value in output.items() if value is not None}


def predictions_validator_schema() -> dict[str, Any]:
    return {
        "bsonType": "object",
        "required": [
            "model",
            "level",
            "series_id",
            "forecast_window_start",
            "forecast_window_end",
            "forecast_window_months",
            "forecasts",
            "schema_version",
        ],
        "properties": {
            "model": {"bsonType": "string"},
            "level": {"bsonType": "string"},
            "series_id": {"bsonType": "string"},
            "forecast_window_start": {"bsonType": "date"},
            "forecast_window_end": {"bsonType": "date"},
            "forecast_window_months": {"bsonType": ["int", "long"]},
            "forecasts": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "required": ["forecast_period_date", "forecast_period_yyyymm", "forecast_value"],
                    "properties": {
                        "forecast_period_date": {"bsonType": "date"},
                        "forecast_period_yyyymm": {"bsonType": "string"},
                        "forecast_value": {"bsonType": ["double", "int", "long", "decimal"]},
                    },
                },
            },
            "schema_version": {"bsonType": "string"},
            "source": {"bsonType": ["object", "null"]},
        },
    }


def ensure_prediction_indexes(collection: Collection, logger: logging.Logger) -> None:
    collection.create_index(
        [
            ("model", ASCENDING),
            ("level", ASCENDING),
            ("series_id", ASCENDING),
            ("forecast_window_start", ASCENDING),
            ("forecast_window_end", ASCENDING),
        ],
        unique=True,
        name="uq_model_level_series_window",
    )
    collection.create_index(
        [("model", ASCENDING), ("forecast_window_start", ASCENDING), ("forecast_window_end", ASCENDING)],
        name="idx_model_window",
    )
    collection.create_index(
        [
            ("model", ASCENDING),
            ("mineral", ASCENDING),
            ("departamento", ASCENDING),
            ("unidad", ASCENDING),
        ],
        name="idx_model_mineral_depto_unidad",
    )
    collection.create_index(
        [("level", ASCENDING), ("series_id", ASCENDING)],
        name="idx_level_series",
    )
    collection.create_index(
        [("model", ASCENDING), ("forecasts.forecast_period_date", ASCENDING)],
        name="idx_model_forecast_period",
    )
    logger.info("Indexes ensured for predictions collection")


def _is_flat(values: list[float], tolerance: float = 1e-12) -> bool:
    if not values:
        return False
    return math.isclose(min(values), max(values), rel_tol=0.0, abs_tol=tolerance)


def build_prediction_operations(
    operational_df: pd.DataFrame,
    now_utc: datetime,
    source_metadata: dict[str, Any],
) -> tuple[list[UpdateOne], dict[str, Any]]:
    operations: list[UpdateOne] = []
    docs_by_model: dict[str, int] = {}
    flat_docs_by_model: dict[str, int] = {}

    for (model, level, series_id), group in operational_df.groupby(["selected_model", "level", "series_id"], sort=False):
        series_group = group.sort_values("forecast_period_date").reset_index(drop=True)

        forecast_points: list[dict[str, Any]] = []
        numeric_values: list[float] = []

        for row in series_group.itertuples(index=False):
            period_date = pd.to_datetime(getattr(row, "forecast_period_date"), errors="coerce")
            forecast_value = _to_optional_float(getattr(row, "forecast_value", None))
            period_yyyymm = _to_optional_str(getattr(row, "forecast_period_yyyymm", None))

            if pd.isna(period_date) or forecast_value is None or period_yyyymm is None:
                continue

            numeric_values.append(float(forecast_value))
            forecast_points.append(
                {
                    "forecast_period_date": period_date.to_pydatetime(),
                    "forecast_period_yyyymm": period_yyyymm,
                    "forecast_value": float(forecast_value),
                }
            )

        if not forecast_points:
            continue

        window_start = forecast_points[0]["forecast_period_date"]
        window_end = forecast_points[-1]["forecast_period_date"]
        first_row = series_group.iloc[0]
        flat_forecast = _is_flat(numeric_values)

        document = {
            "prediction_id": f"{model}|{level}|{series_id}|{window_start:%Y-%m}|{window_end:%Y-%m}",
            "model": str(model),
            "level": str(level),
            "series_id": str(series_id),
            "mineral": _to_optional_str(first_row.get("mineral")),
            "departamento": _to_optional_str(first_row.get("departamento")),
            "unidad": _to_optional_str(first_row.get("unidad")),
            "last_observed_period": pd.to_datetime(first_row.get("last_observed_period"), errors="coerce").to_pydatetime()
            if not pd.isna(pd.to_datetime(first_row.get("last_observed_period"), errors="coerce"))
            else None,
            "forecast_window_start": window_start,
            "forecast_window_end": window_end,
            "forecast_window_months": len(forecast_points),
            "forecasts": forecast_points,
            "forecast_stats": {
                "n_points": len(forecast_points),
                "min_value": float(min(numeric_values)),
                "max_value": float(max(numeric_values)),
                "mean_value": float(sum(numeric_values) / len(numeric_values)),
                "is_flat_forecast": bool(flat_forecast),
            },
            "source": {
                "phase": "fase8_operational",
                **source_metadata,
            },
            "schema_version": "v1",
            "updated_at": now_utc,
        }

        filter_spec = {
            "model": document["model"],
            "level": document["level"],
            "series_id": document["series_id"],
            "forecast_window_start": document["forecast_window_start"],
            "forecast_window_end": document["forecast_window_end"],
        }
        update_spec = {
            "$set": document,
            "$setOnInsert": {
                "ingested_at": now_utc,
            },
        }

        operations.append(UpdateOne(filter_spec, update_spec, upsert=True))
        docs_by_model[document["model"]] = docs_by_model.get(document["model"], 0) + 1
        if flat_forecast:
            flat_docs_by_model[document["model"]] = flat_docs_by_model.get(document["model"], 0) + 1

    summary = {
        "documents_by_model": docs_by_model,
        "flat_documents_by_model": flat_docs_by_model,
        "total_documents": int(sum(docs_by_model.values())),
    }
    return operations, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 9 MongoDB predictions load pipeline.")
    parser.add_argument(
        "--operational-long-path",
        type=Path,
        default=Path("data/processed/mineria_phase8_operational_forecast_2026.csv"),
        help="Path to long-format operational forecast generated by phase 8.",
    )
    parser.add_argument(
        "--operational-report-path",
        type=Path,
        default=Path("reports/phase8_operational_report.json"),
        help="Path to phase 8 operational report for metadata traceability.",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=os.getenv("MONGODB_ATLAS_URI"),
        help="MongoDB Atlas URI. Defaults to MONGODB_ATLAS_URI env.",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default=os.getenv("MONGODB_DB_NAME", "mineria_peru"),
        help="MongoDB database name.",
    )
    parser.add_argument(
        "--predictions-collection",
        type=str,
        default=os.getenv("MONGODB_COLLECTION_PREDICTIONS", "predicciones"),
        help="Predictions collection name.",
    )
    parser.add_argument(
        "--skip-validators",
        action="store_true",
        help="Skip collection validator creation/update.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip index creation/update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate operations without writing to MongoDB.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase9_mongodb_predictions_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase9_mongodb_predictions.log"),
        help="Log output path.",
    )
    return parser


def run_from_cli() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    logger = build_logger(args.log_path)
    logger.info("Starting Phase 9 MongoDB predictions load")

    operational_df = load_operational_forecasts(args.operational_long_path)
    source_metadata = _parse_selection_metadata(args.operational_report_path, logger)

    now_utc = datetime.now(timezone.utc)
    operations, operation_summary = build_prediction_operations(
        operational_df=operational_df,
        now_utc=now_utc,
        source_metadata=source_metadata,
    )

    load_counters = {
        "predictions": {
            "n_operations": len(operations),
            "matched_count": 0,
            "modified_count": 0,
            "upserted_count": 0,
        }
    }

    if args.dry_run:
        logger.info("Dry-run mode enabled: MongoDB writes skipped")
    else:
        if not args.mongo_uri:
            raise ValueError("Mongo URI is required. Set MONGODB_ATLAS_URI or pass --mongo-uri.")

        client = build_mongo_client(args.mongo_uri, logger)
        db = client[args.db_name]

        if not args.skip_validators:
            ensure_collection(db, args.predictions_collection, predictions_validator_schema(), logger)

        predictions_collection = db[args.predictions_collection]

        if not args.skip_indexes:
            ensure_prediction_indexes(predictions_collection, logger)

        load_counters["predictions"] = execute_bulk_upserts(predictions_collection, operations)
        logger.info("Prediction upserts: %s", load_counters["predictions"])
        client.close()

    report = {
        "phase": "Fase 9 - MongoDB predictions load",
        "dry_run": bool(args.dry_run),
        "mongodb": {
            "database": args.db_name,
            "predictions_collection": args.predictions_collection,
        },
        "inputs": {
            "operational_long_path": str(args.operational_long_path),
            "operational_report_path": str(args.operational_report_path),
            "rows_operational_input": int(len(operational_df)),
        },
        "documents": operation_summary,
        "operations": load_counters,
        "generated_at_utc": now_utc.isoformat(),
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved load report: %s", args.report_path)
    logger.info("Phase 9 MongoDB predictions load finished")
    return 0
