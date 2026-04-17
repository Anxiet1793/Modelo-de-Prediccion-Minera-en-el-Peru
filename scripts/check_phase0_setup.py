from __future__ import annotations

import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "interim",
    ROOT / "data" / "processed",
    ROOT / "notebooks",
    ROOT / "src",
    ROOT / "src" / "ingestion",
    ROOT / "src" / "profiling",
    ROOT / "src" / "cleaning",
    ROOT / "src" / "validation",
    ROOT / "src" / "transforms",
    ROOT / "src" / "mongodb",
    ROOT / "src" / "eda",
    ROOT / "src" / "features",
    ROOT / "src" / "models",
    ROOT / "src" / "utils",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "logs",
    ROOT / "reports",
]

REQUIRED_FILES = [
    ROOT / "requirements.txt",
    ROOT / ".gitignore",
    ROOT / ".env.example",
    ROOT / "README.md",
]


def check_python() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major == 3 and minor >= 11)
    detail = f"Python detected: {platform.python_version()} (required: >=3.11)"
    return ok, detail


def check_paths() -> tuple[bool, list[str]]:
    errors: list[str] = []

    for directory in REQUIRED_DIRS:
        if not directory.exists() or not directory.is_dir():
            errors.append(f"Missing directory: {directory}")

    for file_path in REQUIRED_FILES:
        if not file_path.exists() or not file_path.is_file():
            errors.append(f"Missing file: {file_path}")

    return len(errors) == 0, errors


def main() -> int:
    print("=== PHASE 0 SETUP CHECK ===")

    py_ok, py_msg = check_python()
    print(py_msg)

    paths_ok, path_errors = check_paths()

    if paths_ok:
        print("Directory and file structure: OK")
    else:
        print("Directory and file structure: FAILED")
        for error in path_errors:
            print(f" - {error}")

    overall_ok = py_ok and paths_ok
    print(f"Overall status: {'OK' if overall_ok else 'FAILED'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
