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
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase5_mongodb")
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


def _parse_optional_int_env(env_name: str, logger: logging.Logger) -> int | None:
    raw_value = os.getenv(env_name)
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%s", env_name, raw_value)
        return None
    if value <= 0:
        logger.warning("Ignoring non-positive env %s=%s", env_name, raw_value)
        return None
    return value


def build_mongo_client(uri: str, logger: logging.Logger) -> MongoClient:
    # No arbitrary pool tuning is applied by default. Optional timeouts can be set via env.
    client_kwargs: dict[str, Any] = {"appname": "mineria-phase5-loader"}

    timeout_mapping = {
        "MONGODB_CONNECT_TIMEOUT_MS": "connectTimeoutMS",
        "MONGODB_SOCKET_TIMEOUT_MS": "socketTimeoutMS",
        "MONGODB_SERVER_SELECTION_TIMEOUT_MS": "serverSelectionTimeoutMS",
        "MONGODB_WAIT_QUEUE_TIMEOUT_MS": "waitQueueTimeoutMS",
    }

    for env_name, pymongo_name in timeout_mapping.items():
        parsed_value = _parse_optional_int_env(env_name, logger)
        if parsed_value is not None:
            client_kwargs[pymongo_name] = parsed_value

    client = MongoClient(uri, **client_kwargs)
    client.admin.command("ping")
    return client


def _to_optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _to_optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _to_optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def classify_gap_quality(abs_pct_gap: float | None, ok_max_abs: float, warning_max_abs: float) -> str:
    if abs_pct_gap is None or math.isnan(abs_pct_gap):
        return "critical"
    if abs_pct_gap <= ok_max_abs:
        return "ok"
    if abs_pct_gap <= warning_max_abs:
        return "warning"
    return "critical"


def parse_period_column(df: pd.DataFrame, column_name: str = "period_date") -> pd.DataFrame:
    output = df.copy()
    output[column_name] = pd.to_datetime(output[column_name], errors="coerce")
    return output


def select_rows_by_strategy(
    tidy: pd.DataFrame,
    agg_mineral: pd.DataFrame,
    agg_dept: pd.DataFrame,
    strategy: str,
    period_yyyymm: str | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    if strategy not in {"incremental", "full_refresh"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    selected_period = period_yyyymm

    if strategy == "incremental":
        if selected_period is None:
            periods = tidy["period_yyyymm"].dropna().astype(str)
            if periods.empty:
                raise ValueError("No periods found in tidy dataset for incremental loading")
            selected_period = periods.max()
        logger.info("Incremental strategy selected for period: %s", selected_period)

        tidy = tidy[tidy["period_yyyymm"].astype(str) == selected_period].copy()
        agg_mineral = agg_mineral[agg_mineral["period_yyyymm"].astype(str) == selected_period].copy()
        agg_dept = agg_dept[agg_dept["period_yyyymm"].astype(str) == selected_period].copy()
    else:
        if selected_period is not None:
            logger.info("Full refresh constrained to period: %s", selected_period)
            tidy = tidy[tidy["period_yyyymm"].astype(str) == selected_period].copy()
            agg_mineral = agg_mineral[agg_mineral["period_yyyymm"].astype(str) == selected_period].copy()
            agg_dept = agg_dept[agg_dept["period_yyyymm"].astype(str) == selected_period].copy()
        else:
            logger.info("Full refresh strategy selected for all available periods")

    return tidy, agg_mineral, agg_dept, selected_period


def build_fact_operations(tidy: pd.DataFrame, now_utc: datetime) -> list[UpdateOne]:
    operations: list[UpdateOne] = []

    for row in tidy.itertuples(index=False):
        period_date = pd.to_datetime(getattr(row, "period_date"), errors="coerce")
        if pd.isna(period_date):
            continue

        document = {
            "record_id": _to_optional_str(getattr(row, "record_id", None)),
            "period_date": period_date.to_pydatetime(),
            "period_yyyymm": _to_optional_str(getattr(row, "period_yyyymm", None)),
            "year": _to_optional_int(getattr(row, "year", None)),
            "month": _to_optional_int(getattr(row, "month", None)),
            "quarter": _to_optional_int(getattr(row, "quarter", None)),
            "semester": _to_optional_int(getattr(row, "semester", None)),
            "series_code": _to_optional_str(getattr(row, "series_code", None)),
            "series_description": _to_optional_str(getattr(row, "series_description", None)),
            "mineral": _to_optional_str(getattr(row, "mineral", None)),
            "departamento": _to_optional_str(getattr(row, "departamento", None)),
            "unidad": _to_optional_str(getattr(row, "unidad", None)),
            "value": _to_optional_float(getattr(row, "value", None)),
            "source_dataset": _to_optional_str(getattr(row, "source_dataset", None)),
            "metric_level": _to_optional_str(getattr(row, "metric_level", None)),
            "area_geografica": _to_optional_str(getattr(row, "area_geografica", None)),
            "is_total_departamento": bool(getattr(row, "is_total_departamento", False)),
            "is_missing": bool(getattr(row, "is_missing", False)),
            "is_invalid_numeric": bool(getattr(row, "is_invalid_numeric", False)),
            "schema_version": "v1",
            "updated_at": now_utc,
        }

        period_key = document["period_date"]
        series_key = document["series_code"]
        if period_key is None or series_key is None:
            continue

        update_spec = {
            "$set": document,
            "$setOnInsert": {
                "ingested_at": now_utc,
            },
        }

        operations.append(
            UpdateOne(
                {
                    "period_date": period_key,
                    "series_code": series_key,
                },
                update_spec,
                upsert=True,
            )
        )

    return operations


def build_aggregate_operations(
    agg_mineral: pd.DataFrame,
    agg_dept: pd.DataFrame,
    now_utc: datetime,
    gap_ok_max_abs: float,
    gap_warning_max_abs: float,
) -> list[UpdateOne]:
    operations: list[UpdateOne] = []

    for row in agg_mineral.itertuples(index=False):
        period_date = pd.to_datetime(getattr(row, "period_date"), errors="coerce")
        if pd.isna(period_date):
            continue

        pct_gap = _to_optional_float(getattr(row, "pct_gap_vs_total", None))
        abs_pct_gap = abs(pct_gap) if pct_gap is not None else None
        gap_flag = classify_gap_quality(abs_pct_gap, gap_ok_max_abs, gap_warning_max_abs)

        mineral = _to_optional_str(getattr(row, "mineral", None))
        unidad = _to_optional_str(getattr(row, "unidad", None))
        period_yyyymm = _to_optional_str(getattr(row, "period_yyyymm", None))

        document = {
            "aggregate_id": f"mineral_period_unit|{period_yyyymm}|{unidad}|{mineral}",
            "aggregation_level": "mineral_period_unit",
            "period_date": period_date.to_pydatetime(),
            "period_yyyymm": period_yyyymm,
            "year": _to_optional_int(getattr(row, "year", None)),
            "month": _to_optional_int(getattr(row, "month", None)),
            "quarter": _to_optional_int(getattr(row, "quarter", None)),
            "semester": _to_optional_int(getattr(row, "semester", None)),
            "unidad": unidad,
            "mineral": mineral,
            "departamento": None,
            "dept_sum_value": _to_optional_float(getattr(row, "dept_sum_value", None)),
            "n_departamentos": _to_optional_int(getattr(row, "n_departamentos", None)),
            "official_total_value": _to_optional_float(getattr(row, "official_total_value", None)),
            "abs_gap_vs_total": _to_optional_float(getattr(row, "abs_gap_vs_total", None)),
            "pct_gap_vs_total": pct_gap,
            "gap_quality_flag": gap_flag,
            "schema_version": "v1",
            "computed_at": now_utc,
            "updated_at": now_utc,
        }

        update_spec = {
            "$set": document,
            "$setOnInsert": {
                "ingested_at": now_utc,
            },
        }

        operations.append(
            UpdateOne(
                {
                    "aggregation_level": document["aggregation_level"],
                    "period_date": document["period_date"],
                    "unidad": document["unidad"],
                    "mineral": document["mineral"],
                    "departamento": document["departamento"],
                },
                update_spec,
                upsert=True,
            )
        )

    for row in agg_dept.itertuples(index=False):
        period_date = pd.to_datetime(getattr(row, "period_date"), errors="coerce")
        if pd.isna(period_date):
            continue

        departamento = _to_optional_str(getattr(row, "departamento", None))
        unidad = _to_optional_str(getattr(row, "unidad", None))
        period_yyyymm = _to_optional_str(getattr(row, "period_yyyymm", None))

        document = {
            "aggregate_id": f"departamento_period_unit|{period_yyyymm}|{unidad}|{departamento}",
            "aggregation_level": "departamento_period_unit",
            "period_date": period_date.to_pydatetime(),
            "period_yyyymm": period_yyyymm,
            "year": _to_optional_int(getattr(row, "year", None)),
            "month": _to_optional_int(getattr(row, "month", None)),
            "quarter": _to_optional_int(getattr(row, "quarter", None)),
            "semester": _to_optional_int(getattr(row, "semester", None)),
            "unidad": unidad,
            "mineral": None,
            "departamento": departamento,
            "dept_unit_sum_value": _to_optional_float(getattr(row, "dept_unit_sum_value", None)),
            "n_minerales": _to_optional_int(getattr(row, "n_minerales", None)),
            "gap_quality_flag": None,
            "schema_version": "v1",
            "computed_at": now_utc,
            "updated_at": now_utc,
        }

        update_spec = {
            "$set": document,
            "$setOnInsert": {
                "ingested_at": now_utc,
            },
        }

        operations.append(
            UpdateOne(
                {
                    "aggregation_level": document["aggregation_level"],
                    "period_date": document["period_date"],
                    "unidad": document["unidad"],
                    "mineral": document["mineral"],
                    "departamento": document["departamento"],
                },
                update_spec,
                upsert=True,
            )
        )

    return operations


def fact_validator_schema() -> dict[str, Any]:
    return {
        "bsonType": "object",
        "required": [
            "period_date",
            "period_yyyymm",
            "year",
            "month",
            "series_code",
            "mineral",
            "departamento",
            "unidad",
            "value",
            "schema_version",
        ],
        "properties": {
            "period_date": {"bsonType": "date"},
            "period_yyyymm": {"bsonType": "string"},
            "year": {"bsonType": ["int", "long"]},
            "month": {"bsonType": ["int", "long"]},
            "series_code": {"bsonType": "string"},
            "mineral": {"bsonType": "string"},
            "departamento": {"bsonType": "string"},
            "unidad": {"bsonType": "string"},
            "value": {"bsonType": ["double", "int", "long", "decimal"]},
            "schema_version": {"bsonType": "string"},
        },
    }


def aggregate_validator_schema() -> dict[str, Any]:
    return {
        "bsonType": "object",
        "required": [
            "aggregation_level",
            "period_date",
            "period_yyyymm",
            "year",
            "month",
            "unidad",
            "schema_version",
        ],
        "properties": {
            "aggregation_level": {
                "enum": [
                    "mineral_period_unit",
                    "departamento_period_unit",
                ]
            },
            "period_date": {"bsonType": "date"},
            "period_yyyymm": {"bsonType": "string"},
            "year": {"bsonType": ["int", "long"]},
            "month": {"bsonType": ["int", "long"]},
            "unidad": {"bsonType": "string"},
            "schema_version": {"bsonType": "string"},
            "gap_quality_flag": {
                "bsonType": ["string", "null"],
                "enum": ["ok", "warning", "critical", None],
            },
        },
    }


def ensure_collection(
    db: Database,
    collection_name: str,
    validator_schema: dict[str, Any],
    logger: logging.Logger,
) -> None:
    existing = set(db.list_collection_names())
    validator = {"$jsonSchema": validator_schema}

    if collection_name not in existing:
        db.create_collection(
            collection_name,
            validator=validator,
            validationLevel="moderate",
            validationAction="warn",
        )
        logger.info("Created collection with validator: %s", collection_name)
        return

    try:
        db.command(
            {
                "collMod": collection_name,
                "validator": validator,
                "validationLevel": "moderate",
                "validationAction": "warn",
            }
        )
        logger.info("Updated collection validator: %s", collection_name)
    except PyMongoError as error:
        logger.warning("Could not update validator for %s: %s", collection_name, error)


def ensure_indexes(fact_collection: Collection, agg_collection: Collection, logger: logging.Logger) -> None:
    fact_collection.create_index(
        [("period_date", ASCENDING), ("series_code", ASCENDING)],
        unique=True,
        name="uq_period_series",
    )
    fact_collection.create_index(
        [("series_code", ASCENDING), ("period_date", ASCENDING)],
        name="idx_series_period",
    )
    fact_collection.create_index(
        [("mineral", ASCENDING), ("departamento", ASCENDING), ("period_date", ASCENDING)],
        name="idx_mineral_depto_period",
    )
    fact_collection.create_index(
        [("period_date", ASCENDING)],
        name="idx_period",
    )
    fact_collection.create_index(
        [("unidad", ASCENDING), ("mineral", ASCENDING), ("period_date", ASCENDING)],
        name="idx_unidad_mineral_period",
    )

    agg_collection.create_index(
        [
            ("aggregation_level", ASCENDING),
            ("period_date", ASCENDING),
            ("unidad", ASCENDING),
            ("mineral", ASCENDING),
            ("departamento", ASCENDING),
        ],
        unique=True,
        name="uq_agg_period_unit_mineral_depto",
    )
    agg_collection.create_index(
        [
            ("aggregation_level", ASCENDING),
            ("mineral", ASCENDING),
            ("unidad", ASCENDING),
            ("period_date", ASCENDING),
        ],
        name="idx_agg_mineral_period",
    )
    agg_collection.create_index(
        [
            ("aggregation_level", ASCENDING),
            ("departamento", ASCENDING),
            ("unidad", ASCENDING),
            ("period_date", ASCENDING),
        ],
        name="idx_agg_depto_period",
    )
    agg_collection.create_index(
        [("gap_quality_flag", ASCENDING), ("period_date", ASCENDING)],
        name="idx_gap_flag_period",
    )

    logger.info("Indexes ensured for both collections")


def execute_bulk_upserts(collection: Collection, operations: list[UpdateOne], batch_size: int = 1000) -> dict[str, int]:
    counters = {
        "n_operations": len(operations),
        "matched_count": 0,
        "modified_count": 0,
        "upserted_count": 0,
    }

    if not operations:
        return counters

    for start in range(0, len(operations), batch_size):
        chunk = operations[start : start + batch_size]
        result = collection.bulk_write(chunk, ordered=False)
        counters["matched_count"] += int(result.matched_count)
        counters["modified_count"] += int(result.modified_count)
        counters["upserted_count"] += int(result.upserted_count)

    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 5 MongoDB load pipeline.")

    parser.add_argument(
        "--tidy-path",
        type=Path,
        default=Path("data/processed/mineria_mensual_tidy.csv"),
        help="Path to tidy monthly dataset.",
    )
    parser.add_argument(
        "--agg-mineral-path",
        type=Path,
        default=Path("data/processed/mineria_agg_mineral_period.csv"),
        help="Path to mineral-period aggregate dataset.",
    )
    parser.add_argument(
        "--agg-dept-path",
        type=Path,
        default=Path("data/processed/mineria_agg_departamento_period.csv"),
        help="Path to department-period aggregate dataset.",
    )
    parser.add_argument(
        "--strategy",
        choices=["incremental", "full_refresh"],
        default=os.getenv("PHASE5_LOAD_STRATEGY", "incremental"),
        help="Load strategy: incremental monthly (default) or full_refresh.",
    )
    parser.add_argument(
        "--period-yyyymm",
        type=str,
        default=None,
        help="Optional period filter in YYYY-MM. In incremental, defaults to max period found.",
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
        "--fact-collection",
        type=str,
        default=os.getenv("MONGODB_COLLECTION_FACT", "mineria_observaciones_mensuales"),
        help="Fact/granular collection name.",
    )
    parser.add_argument(
        "--agg-collection",
        type=str,
        default=os.getenv("MONGODB_COLLECTION_AGG", "mineria_aggregates_mensuales"),
        help="Aggregates collection name.",
    )
    parser.add_argument(
        "--gap-ok-threshold",
        type=float,
        default=float(os.getenv("GAP_OK_THRESHOLD_PCT", "1")),
        help="Absolute pct_gap_vs_total max value for 'ok'.",
    )
    parser.add_argument(
        "--gap-warning-threshold",
        type=float,
        default=float(os.getenv("GAP_WARNING_THRESHOLD_PCT", "5")),
        help="Absolute pct_gap_vs_total max value for 'warning'.",
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
        default=Path("reports/phase5_mongodb_load_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/phase5_mongodb_load.log"),
        help="Log output path.",
    )

    return parser


def run_from_cli() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    logger = build_logger(args.log_path)
    logger.info("Starting Phase 5 MongoDB load")

    if args.gap_warning_threshold <= args.gap_ok_threshold:
        raise ValueError("gap_warning_threshold must be greater than gap_ok_threshold")

    tidy = parse_period_column(pd.read_csv(args.tidy_path))
    agg_mineral = parse_period_column(pd.read_csv(args.agg_mineral_path))
    agg_dept = parse_period_column(pd.read_csv(args.agg_dept_path))

    selected_tidy, selected_agg_mineral, selected_agg_dept, selected_period = select_rows_by_strategy(
        tidy=tidy,
        agg_mineral=agg_mineral,
        agg_dept=agg_dept,
        strategy=args.strategy,
        period_yyyymm=args.period_yyyymm,
        logger=logger,
    )

    now_utc = datetime.now(timezone.utc)
    fact_operations = build_fact_operations(selected_tidy, now_utc)
    agg_operations = build_aggregate_operations(
        agg_mineral=selected_agg_mineral,
        agg_dept=selected_agg_dept,
        now_utc=now_utc,
        gap_ok_max_abs=args.gap_ok_threshold,
        gap_warning_max_abs=args.gap_warning_threshold,
    )

    load_counters = {
        "fact": {
            "n_operations": len(fact_operations),
            "matched_count": 0,
            "modified_count": 0,
            "upserted_count": 0,
        },
        "aggregates": {
            "n_operations": len(agg_operations),
            "matched_count": 0,
            "modified_count": 0,
            "upserted_count": 0,
        },
    }

    if args.dry_run:
        logger.info("Dry-run mode enabled: MongoDB writes skipped")
    else:
        if not args.mongo_uri:
            raise ValueError("Mongo URI is required. Set MONGODB_ATLAS_URI or pass --mongo-uri.")

        client = build_mongo_client(args.mongo_uri, logger)
        db = client[args.db_name]

        if not args.skip_validators:
            ensure_collection(db, args.fact_collection, fact_validator_schema(), logger)
            ensure_collection(db, args.agg_collection, aggregate_validator_schema(), logger)

        fact_collection = db[args.fact_collection]
        agg_collection = db[args.agg_collection]

        if not args.skip_indexes:
            ensure_indexes(fact_collection, agg_collection, logger)

        load_counters["fact"] = execute_bulk_upserts(fact_collection, fact_operations)
        load_counters["aggregates"] = execute_bulk_upserts(agg_collection, agg_operations)

        logger.info("Fact upserts: %s", load_counters["fact"])
        logger.info("Aggregate upserts: %s", load_counters["aggregates"])

        client.close()

    report = {
        "phase": "Fase 5 - MongoDB Load",
        "strategy": args.strategy,
        "selected_period_yyyymm": selected_period,
        "dry_run": bool(args.dry_run),
        "mongodb": {
            "database": args.db_name,
            "fact_collection": args.fact_collection,
            "agg_collection": args.agg_collection,
        },
        "gap_thresholds_pct": {
            "ok_max_abs": args.gap_ok_threshold,
            "warning_max_abs": args.gap_warning_threshold,
        },
        "inputs": {
            "tidy_path": str(args.tidy_path),
            "agg_mineral_path": str(args.agg_mineral_path),
            "agg_dept_path": str(args.agg_dept_path),
            "rows_tidy_input": int(len(tidy)),
            "rows_agg_mineral_input": int(len(agg_mineral)),
            "rows_agg_dept_input": int(len(agg_dept)),
            "rows_tidy_selected": int(len(selected_tidy)),
            "rows_agg_mineral_selected": int(len(selected_agg_mineral)),
            "rows_agg_dept_selected": int(len(selected_agg_dept)),
        },
        "operations": load_counters,
        "generated_at_utc": now_utc.isoformat(),
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info("Saved load report: %s", args.report_path)
    logger.info("Phase 5 MongoDB load finished")
    return 0
