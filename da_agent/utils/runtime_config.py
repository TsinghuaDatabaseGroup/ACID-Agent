"""Runtime configuration helpers for ACID-Agent.

This module centralizes filesystem paths and runtime defaults that may vary
across machines. Values can be overridden through environment variables so the
codebase does not depend on user-specific absolute paths.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _env_path(name: str, default: Path | str) -> str:
    return str(Path(os.getenv(name, str(default))).expanduser())

KRAMABENCH_DATA_BASE = _env_path(
    "KRAMABENCH_DATA_BASE",
    PROJECT_ROOT / "data" / "data-agent" / "kramabench",
)
KRAMABENCH_DATASET_DIR = _env_path(
    "KRAMABENCH_DATASET_DIR",
    Path(KRAMABENCH_DATA_BASE) / "datasets",
)
KRAMABENCH_OUTPUT_DIR = _env_path(
    "KRAMABENCH_OUTPUT_DIR",
    Path(KRAMABENCH_DATA_BASE) / "output",
)
KRAMABENCH_LOG_DIR = _env_path(
    "KRAMABENCH_LOG_DIR",
    Path(KRAMABENCH_DATA_BASE) / "logs",
)

ACID_RUN_STORAGE_BASE = _env_path(
    "ACID_RUN_STORAGE_BASE",
    PROJECT_ROOT / "data" / "data-agent" / "runs",
)

QWEN_SCORER_MODEL_PATH = _env_path(
    "ACID_LOCAL_SCORER_MODEL",
    PROJECT_ROOT / "models" / "Qwen3-0.6B-Base",
)
PROB_CONTRAST_MODEL_PATH = _env_path(
    "ACID_PROB_CONTRAST_MODEL",
    QWEN_SCORER_MODEL_PATH,
)
EVIDENCE_SURPRISE_MODEL_PATH = _env_path(
    "ACID_EVIDENCE_SURPRISE_MODEL",
    QWEN_SCORER_MODEL_PATH,
)

