"""LLM parse → Chroma where → naive vector search."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    DEFAULT_MIN_REVIEW_COUNT,
    NAIVE_N_RESULTS,
    RETRIEVAL_OVERFETCH_MULTIPLIER,
)
from src.generation import GameQueryParams, load_generator, parse_query_with_llm
from src.ingestion import get_collection, load_embedding_model

FilterMode = Literal["llm", "none"]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    params: GameQueryParams
    chroma_where: dict | None = None


def build_chroma_where(params: GameQueryParams) -> dict | None:
    """Map parsed LLM params to a Chroma metadata filter."""
    clauses: list[dict] = []

    if params.max_price is not None:
        clauses.append({"price": {"$lte": params.max_price}})

    if params.min_rating_pct is not None:
        clauses.append({"rating_pct": {"$gte": params.min_rating_pct}})
        review_floor = params.min_review_count or DEFAULT_MIN_REVIEW_COUNT
        clauses.append({"review_count": {"$gte": review_floor}})
    elif params.min_review_count is not None:
        clauses.append({"review_count": {"$gte": params.min_review_count}})

    if params.genres:
        genre_clauses = [{"genres": {"$contains": genre}} for genre in params.genres]
        clauses.append(
            genre_clauses[0] if len(genre_clauses) == 1 else {"$or": genre_clauses}
        )

    if params.tags:
        tag_clauses = [{"tags": {"$contains": tag}} for tag in params.tags]
        clauses.append(
            tag_clauses[0] if len(tag_clauses) == 1 else {"$or": tag_clauses}
        )

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _chunks_from_query_results(results: dict) -> list[RetrievedChunk]:
    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return [
        RetrievedChunk(
            chunk_id=doc_id,
            text=document,
            metadata=metadata,
            score=1.0 / (1.0 + distance),
        )
        for doc_id, distance, document, metadata in zip(
            ids, distances, documents, metadatas
        )
    ]


def dedup_by_app_id(chunks: list[RetrievedChunk], n_games: int) -> list[RetrievedChunk]:
    """Keep the highest-scoring chunk per game (results are already ranked)."""
    unique: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        app_id = str(chunk.metadata.get("app_id", ""))
        if not app_id or app_id in seen:
            continue
        seen.add(app_id)
        unique.append(chunk)
        if len(unique) >= n_games:
            break
    return unique


def retrieve_naive(
    semantic_query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    *,
    top_k: int = NAIVE_N_RESULTS,
    where: dict | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and run cosine similarity search, optionally filtered."""
    overfetch = top_k * RETRIEVAL_OVERFETCH_MULTIPLIER
    n_results = min(overfetch, collection.count()) if not where else overfetch
    if n_results == 0:
        return []

    query_embedding = embed_model.encode(
        [semantic_query], show_progress_bar=False
    ).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    if not results["ids"][0]:
        return []
    return dedup_by_app_id(_chunks_from_query_results(results), top_k)


def retrieve(
    query: str,
    *,
    filter_mode: FilterMode = "llm",
    collection: chromadb.Collection | None = None,
    embed_model: SentenceTransformer | None = None,
    generator=None,
    top_k: int = NAIVE_N_RESULTS,
) -> RetrievalResult:
    """
    Pipeline:
    1. Parse query with LLM (or skip for baseline)
    2. Build Chroma where from structured params
    3. Naive vector search on filtered chunks
    """
    collection = collection or get_collection()
    if collection.count() == 0:
        raise ValueError(
            "Chroma collection is empty. Run: python -m src.ingestion --force"
        )

    if filter_mode == "none":
        params = GameQueryParams(semantic_query=query.strip())
        chroma_where = None
    else:
        if generator is None:
            generator = load_generator()
        params = parse_query_with_llm(generator, query)
        chroma_where = build_chroma_where(params) if params.has_filters() else None

    embed_model = embed_model or load_embedding_model()
    chunks = retrieve_naive(
        params.semantic_query or query,
        collection,
        embed_model,
        top_k=top_k,
        where=chroma_where,
    )
    return RetrievalResult(
        chunks=chunks,
        params=params,
        chroma_where=chroma_where,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test filtered naive RAG retrieval.")
    parser.add_argument("query", help="User query")
    parser.add_argument(
        "--mode",
        choices=["llm", "none"],
        default="llm",
        help="llm: parse + Chroma filter; none: baseline naive RAG",
    )
    parser.add_argument("--top-k", type=int, default=NAIVE_N_RESULTS)
    args = parser.parse_args()

    generator = load_generator() if args.mode == "llm" else None
    result = retrieve(
        args.query,
        filter_mode=args.mode,
        generator=generator,
        top_k=args.top_k,
    )

    print("Parsed:", json.dumps(result.params.to_dict(), indent=2))
    print("Chroma where:", result.chroma_where)
    print(f"Chunks ({len(result.chunks)}):")
    for chunk in result.chunks:
        name = chunk.metadata.get("name", "?")
        price = chunk.metadata.get("price", "?")
        rating = chunk.metadata.get("rating_pct", "?")
        print(f"  [{chunk.score:.3f}] {name} (${price}, {rating}% pos)")
        print(f"    {chunk.text}")


if __name__ == "__main__":
    main()
