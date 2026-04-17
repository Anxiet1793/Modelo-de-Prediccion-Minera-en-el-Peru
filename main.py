from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_FILENAME = "DatosProduccionMinera.xlsx"


@dataclass
class PipelineStep:
    name: str
    script_path: Path
    args: list[str]


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline_main")
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


def resolve_raw_file(raw_dir: Path, raw_file_name: str) -> Path:
    raw_path = raw_dir / raw_file_name
    if raw_path.exists() and raw_path.is_file():
        return raw_path

    available_files = sorted([p.name for p in raw_dir.glob("*") if p.is_file()])
    raise FileNotFoundError(
        "No se encontro el archivo esperado en data/raw. "
        f"Esperado: {raw_file_name}. Disponibles: {available_files}"
    )


def build_steps(args: argparse.Namespace, raw_file_path: Path, run_mongo_steps: bool) -> list[PipelineStep]:
    steps: list[PipelineStep] = []

    if not args.skip_phase0_check:
        steps.append(
            PipelineStep(
                name="Fase 0 - Setup check",
                script_path=ROOT / "scripts" / "check_phase0_setup.py",
                args=[],
            )
        )

    steps.extend(
        [
            PipelineStep(
                name="Fase 2 - Cleaning",
                script_path=ROOT / "scripts" / "run_phase2_cleaning.py",
                args=["--excel-path", str(raw_file_path)],
            ),
            PipelineStep(
                name="Fase 3 - Validation",
                script_path=ROOT / "scripts" / "run_phase3_validation.py",
                args=[],
            ),
            PipelineStep(
                name="Fase 4 - Transforms",
                script_path=ROOT / "scripts" / "run_phase4_transforms.py",
                args=[],
            ),
        ]
    )

    if run_mongo_steps:
        steps.append(
            PipelineStep(
                name="Fase 5 - MongoDB load",
                script_path=ROOT / "scripts" / "run_phase5_mongodb_load.py",
                args=["--strategy", args.mongo_strategy],
            )
        )

    steps.extend(
        [
            PipelineStep(
                name="Fase 6 - EDA",
                script_path=ROOT / "scripts" / "run_phase6_eda.py",
                args=[],
            ),
            PipelineStep(
                name="Fase 7 - Features",
                script_path=ROOT / "scripts" / "run_phase7_features.py",
                args=[],
            ),
            PipelineStep(
                name="Fase 8 - Forecast baselines",
                script_path=ROOT / "scripts" / "run_phase8_forecasting.py",
                args=[],
            ),
            PipelineStep(
                name="Fase 8 - Operational forecast",
                script_path=ROOT / "scripts" / "run_phase8_operational_forecast.py",
                args=[
                    "--selection-mode",
                    args.selection_mode,
                    "--dynamic-tolerance-pct",
                    str(args.dynamic_tolerance_pct),
                ],
            ),
        ]
    )

    viz_args: list[str] = [
        "--level",
        args.viz_level,
        "--top-n-categories",
        str(args.viz_top_n_categories),
        "--trend-window",
        str(args.viz_trend_window),
    ]
    if args.viz_series_id:
        viz_args.extend(["--series-id", args.viz_series_id])

    steps.append(
        PipelineStep(
            name="Fase 8 - Visualizations",
            script_path=ROOT / "scripts" / "run_phase8_visualizations.py",
            args=viz_args,
        )
    )

    if run_mongo_steps:
        steps.append(
            PipelineStep(
                name="Fase 9 - MongoDB predictions",
                script_path=ROOT / "scripts" / "run_phase9_mongodb_predictions_load.py",
                args=[],
            )
        )

    return steps


def execute_step(step: PipelineStep, env: dict[str, str], logger: logging.Logger, dry_run: bool) -> dict[str, object]:
    command = [sys.executable, str(step.script_path), *step.args]
    command_text = " ".join(command)

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()

    logger.info("Running step: %s", step.name)
    logger.info("Command: %s", command_text)

    if dry_run:
        finished_at = datetime.now(timezone.utc)
        return {
            "name": step.name,
            "command": command_text,
            "status": "DRY_RUN",
            "return_code": 0,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": round(perf_counter() - started_perf, 3),
        }

    process = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    finished_at = datetime.now(timezone.utc)

    status = "OK" if process.returncode == 0 else "FAILED"
    result = {
        "name": step.name,
        "command": command_text,
        "status": status,
        "return_code": int(process.returncode),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": round(perf_counter() - started_perf, 3),
    }

    if process.returncode == 0:
        logger.info("Step finished successfully: %s", step.name)
    else:
        logger.error("Step failed: %s (return_code=%s)", step.name, process.returncode)

    return result


def write_run_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Iniciador principal del pipeline. "
            "Solo necesitas colocar el archivo esperado en data/raw y ejecutar este main."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Carpeta donde se coloca el archivo fuente canonico.",
    )
    parser.add_argument(
        "--raw-file-name",
        type=str,
        default=DEFAULT_RAW_FILENAME,
        help="Nombre esperado del archivo fuente dentro de data/raw.",
    )
    parser.add_argument(
        "--mongo-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto: ejecuta fases Mongo si hay URI; always: obliga Mongo; never: omite Mongo.",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=None,
        help="URI de MongoDB Atlas. Si no se pasa, se usa MONGODB_ATLAS_URI del entorno/.env.",
    )
    parser.add_argument(
        "--mongo-strategy",
        choices=["incremental", "full_refresh"],
        default="full_refresh",
        help="Estrategia para fase 5 cuando Mongo esta habilitado.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["level_mae", "series_mae", "series_mae_dynamic"],
        default="series_mae_dynamic",
        help="Modo de seleccion de modelo para forecast operativo.",
    )
    parser.add_argument(
        "--dynamic-tolerance-pct",
        type=float,
        default=10.0,
        help="Tolerancia de MAE para permitir modelos dinamicos en mode=series_mae_dynamic.",
    )
    parser.add_argument(
        "--viz-level",
        choices=["nacional_mineral", "departamento_mineral"],
        default="departamento_mineral",
        help="Nivel para graficos de fase 8.",
    )
    parser.add_argument(
        "--viz-series-id",
        type=str,
        default="Cobre|Ancash|tm.f",
        help="Serie principal para graficos. Si se deja vacio, se toma la de mayor forecast.",
    )
    parser.add_argument(
        "--viz-top-n-categories",
        type=int,
        default=6,
        help="Numero de categorias en grafico comparativo.",
    )
    parser.add_argument(
        "--viz-trend-window",
        type=int,
        default=3,
        help="Ventana rolling para grafico de tendencia.",
    )
    parser.add_argument(
        "--skip-phase0-check",
        action="store_true",
        help="Omitir check de estructura de fase 0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar y registrar comandos sin ejecutar scripts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/main_pipeline_run_report.json"),
        help="Ruta del reporte consolidado de ejecucion.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/main_pipeline.log"),
        help="Ruta del log principal.",
    )
    return parser


def run_main() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    logger = build_logger(args.log_path)
    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()

    raw_dir = ROOT / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = resolve_raw_file(raw_dir, args.raw_file_name)

    mongo_uri = args.mongo_uri or os.getenv("MONGODB_ATLAS_URI")
    if args.mongo_mode == "always" and not mongo_uri:
        raise ValueError("mongo-mode=always requiere --mongo-uri o MONGODB_ATLAS_URI")

    run_mongo_steps = False
    if args.mongo_mode == "always":
        run_mongo_steps = True
    elif args.mongo_mode == "auto":
        run_mongo_steps = bool(mongo_uri)

    env = os.environ.copy()
    if mongo_uri:
        env["MONGODB_ATLAS_URI"] = mongo_uri

    logger.info("Starting main pipeline")
    logger.info("Expected raw file: %s", raw_file_path)
    logger.info("Mongo steps enabled: %s", run_mongo_steps)

    steps = build_steps(args=args, raw_file_path=raw_file_path, run_mongo_steps=run_mongo_steps)
    step_results: list[dict[str, object]] = []

    status = "OK"
    failure_message: str | None = None

    try:
        for step in steps:
            result = execute_step(step=step, env=env, logger=logger, dry_run=args.dry_run)
            step_results.append(result)
            if result["status"] == "FAILED":
                status = "FAILED"
                failure_message = f"Step failed: {step.name}"
                break
    except Exception as error:
        status = "FAILED"
        failure_message = str(error)
        logger.exception("Main pipeline failed with exception")

    finished_at = datetime.now(timezone.utc)
    report_payload = {
        "pipeline": "Main orchestrator",
        "status": status,
        "dry_run": bool(args.dry_run),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": round(perf_counter() - started_perf, 3),
        "raw_input": {
            "raw_dir": str(raw_dir),
            "expected_file_name": args.raw_file_name,
            "resolved_file_path": str(raw_file_path),
        },
        "mongo": {
            "mode": args.mongo_mode,
            "enabled": run_mongo_steps,
            "uri_present": bool(mongo_uri),
            "strategy_phase5": args.mongo_strategy,
        },
        "operational_selection": {
            "selection_mode": args.selection_mode,
            "dynamic_tolerance_pct": args.dynamic_tolerance_pct,
        },
        "visualizations": {
            "level": args.viz_level,
            "series_id": args.viz_series_id,
            "top_n_categories": args.viz_top_n_categories,
            "trend_window": args.viz_trend_window,
        },
        "steps": step_results,
        "failure_message": failure_message,
    }

    write_run_report(report_path=ROOT / args.report_path, payload=report_payload)
    logger.info("Saved main pipeline report: %s", ROOT / args.report_path)

    if status == "OK":
        logger.info("Main pipeline finished successfully")
        return 0

    logger.error("Main pipeline finished with errors")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_main())
