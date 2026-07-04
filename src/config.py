"""Project paths — extended with phase-specific settings as modules are added."""

import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_source"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
