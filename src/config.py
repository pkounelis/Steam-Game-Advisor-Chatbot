"""Project configuration: paths and hyperparameters."""

import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_source"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

STEAM_GAMES_CSV_PATH = DATA_DIR / "steam_games.csv"

STEAM_GAMES_DATASET_URL = (
    "https://huggingface.co/datasets/FronkonGames/steam-games-dataset/"
    "resolve/main/games.csv"
)

EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "steam_games"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# RTX 3050 Ti 4GB: MiniLM is small; fp16 + moderate batches stay within VRAM
EMBEDDING_BATCH_SIZE = 128
CHROMA_ADD_BATCH_SIZE = 256

# Minimum reviews when filtering by rating_pct
DEFAULT_MIN_REVIEW_COUNT = 50

# Retrieval
NAIVE_N_RESULTS = 4
RETRIEVAL_OVERFETCH_MULTIPLIER = 5

# Generation / query parsing
LLM_MODEL_ID = "google/gemma-2-2b-it"
GENERATION_MAX_NEW_TOKENS = 200
PARSE_MAX_NEW_TOKENS = 120


def get_device() -> str:
    """Return cuda, mps, or cpu depending on hardware availability."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
