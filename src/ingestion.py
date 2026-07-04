"""Load Steam games CSV, chunk text, embed, and persist to ChromaDB."""

from __future__ import annotations

import argparse
import urllib.request
from io import StringIO
from pathlib import Path

import chromadb
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    CHROMA_ADD_BATCH_SIZE,
    CHROMA_COLLECTION_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_ID,
    STEAM_GAMES_CSV_PATH,
    STEAM_GAMES_DATASET_URL,
    VECTOR_DB_DIR,
    get_device,
)

RENAME_COLUMNS = {
    "AppID": "app_id",
    "Name": "name",
    "Price": "price",
    "Genres": "genres",
    "Tags": "tags",
    "About the game": "about_the_game",
    "Positive": "positive_ratings",
    "Negative": "negative_ratings",
}

REQUIRED_COLUMNS = ("app_id", "name")


def download_games_csv(
    dest: Path = STEAM_GAMES_CSV_PATH,
    url: str = STEAM_GAMES_DATASET_URL,
) -> Path:
    """Download the dataset with pandas and save locally."""
    print(f"Downloading dataset from {url}")

    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")

    lines = text.splitlines()
    if lines and "DiscountDLC count" in lines[0]:
        lines[0] = lines[0].replace("DiscountDLC count", "Discount,DLC count")

    df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    print(f"Download complete: {dest} ({len(df):,} rows)")
    return dest


def ensure_games_csv(csv_path: Path = STEAM_GAMES_CSV_PATH) -> Path:
    """Return path to the games CSV, downloading it if missing."""
    if csv_path.exists():
        return csv_path
    return download_games_csv(csv_path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map FronkonGames dataset columns to a canonical schema."""
    normalized = df.rename(columns=RENAME_COLUMNS)
    missing = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns {missing}. Found: {list(df.columns)}"
        )
    return normalized


def _safe_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_games_csv(csv_path: Path, *, limit: int | None = None) -> pd.DataFrame:
    """Load and normalize the saved Steam games CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalize_columns(df)
    df["app_id"] = df["app_id"].astype(str)
    df = df[df["app_id"].str.match(r"^\d+$", na=False)]
    df = df.drop_duplicates(subset=["app_id"], keep="first")
    df = df[df["name"].map(_safe_str).astype(bool)]
    df = df.reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df


def build_game_text(row: pd.Series) -> str:
    """Combine game fields into a single searchable document."""
    parts = [
        f"Game: {_safe_str(row.get('name'))}",
        f"Genres: {_safe_str(row.get('genres'))}",
        f"Tags: {_safe_str(row.get('tags'))}",
        f"Description: {_safe_str(row.get('about_the_game'))}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


def _parse_csv_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _review_metrics(row: pd.Series) -> tuple[float, int]:
    """Return (rating_pct, review_count) from positive/negative rating columns."""
    pos = int(row["positive_ratings"]) if pd.notna(row.get("positive_ratings")) else 0
    neg = int(row["negative_ratings"]) if pd.notna(row.get("negative_ratings")) else 0
    total = pos + neg
    rating_pct = round(100.0 * pos / total, 1) if total > 0 else 0.0
    return rating_pct, total


def _chunk_records_for_game(
    row: pd.Series, splitter: RecursiveCharacterTextSplitter
) -> list[dict]:
    """Split one game into chunk records with metadata."""
    text = build_game_text(row)
    if not text:
        return []

    app_id = _safe_str(row["app_id"])
    genres = _parse_csv_list(row.get("genres"))
    tags = _parse_csv_list(row.get("tags"))
    rating_pct, review_count = _review_metrics(row)
    base_metadata = {
        "app_id": app_id,
        "name": _safe_str(row.get("name")),
        "price": float(row["price"]) if pd.notna(row.get("price")) else 0.0,
        "rating_pct": rating_pct,
        "review_count": review_count,
        "positive_ratings": int(row["positive_ratings"])
        if pd.notna(row.get("positive_ratings"))
        else 0,
        "negative_ratings": int(row["negative_ratings"])
        if pd.notna(row.get("negative_ratings"))
        else 0,
    }
    if genres:
        base_metadata["genres"] = genres
    if tags:
        base_metadata["tags"] = tags

    chunks = splitter.split_text(text)
    return [
        {
            "id": f"{app_id}_{idx}",
            "text": chunk,
            "metadata": {**base_metadata, "chunk_index": idx},
        }
        for idx, chunk in enumerate(chunks)
    ]


def build_chunk_records(df: pd.DataFrame) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    records: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking games"):
        records.extend(_chunk_records_for_game(row, splitter))
    return records


def get_chroma_client() -> chromadb.PersistentClient:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_DB_DIR.resolve()))


def get_collection(
    client: chromadb.PersistentClient | None = None,
    *,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    client = client or get_chroma_client()
    if force_rebuild:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _add_batches(
    collection: chromadb.Collection,
    records: list[dict],
    embed_model: SentenceTransformer,
) -> int:
    total = 0
    batch_starts = range(0, len(records), CHROMA_ADD_BATCH_SIZE)
    for start in tqdm(batch_starts, desc="Embedding & indexing"):
        batch = records[start : start + CHROMA_ADD_BATCH_SIZE]
        texts = [item["text"] for item in batch]
        embeddings = embed_model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=[item["id"] for item in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in batch],
        )
        total += len(batch)
    return total


def load_embedding_model() -> SentenceTransformer:
    device = get_device()
    print(f"Loading embedding model on {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_ID, device=device)
    if device == "cuda":
        import torch

        model = model.half()
        torch.cuda.empty_cache()
    return model


def build_index(
    csv_path: Path | None = None,
    *,
    force_rebuild: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Build or load the Chroma index from the Steam games CSV."""
    csv_path = ensure_games_csv(csv_path or STEAM_GAMES_CSV_PATH)

    if limit is not None:
        print(f"Indexing sample of {limit:,} games (omit --limit for full dataset)")

    df = load_games_csv(csv_path, limit=limit)

    client = get_chroma_client()
    collection = get_collection(client, force_rebuild=force_rebuild)

    if collection.count() > 0 and not force_rebuild:
        return {
            "games": len(df),
            "chunks": collection.count(),
            "rebuilt": False,
        }

    records = build_chunk_records(df)
    if not records:
        raise ValueError("No text chunks were produced from the CSV.")

    print(f"Built {len(records):,} chunks from {len(df):,} games")

    embed_model = load_embedding_model()
    chunk_count = _add_batches(collection, records, embed_model)

    return {
        "games": len(df),
        "chunks": chunk_count,
        "rebuilt": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Steam games vector index.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"Path to games CSV (default: {STEAM_GAMES_CSV_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N games (default: full dataset)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the Chroma collection even if it already exists",
    )
    args = parser.parse_args()

    csv_path = args.csv or STEAM_GAMES_CSV_PATH

    stats = build_index(csv_path, force_rebuild=args.force, limit=args.limit)
    action = "Rebuilt" if stats["rebuilt"] else "Loaded existing"
    print(
        f"{action} index: {stats['games']} games, {stats['chunks']} chunks "
        f"in {VECTOR_DB_DIR}"
    )


if __name__ == "__main__":
    main()
