"""Local Gemma generation and LLM query parsing."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import torch
from dotenv import load_dotenv
from transformers import pipeline

from src.config import (
    GENERATION_MAX_NEW_TOKENS,
    LLM_MODEL_ID,
    PARSE_MAX_NEW_TOKENS,
    PROJECT_ROOT,
    get_device,
)

load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN", "")


@dataclass
class GameQueryParams:
    max_price: float | None = None
    min_rating_pct: float | None = None
    min_review_count: int | None = None
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    semantic_query: str = ""

    def has_filters(self) -> bool:
        return (
            self.max_price is not None
            or self.min_rating_pct is not None
            or self.min_review_count is not None
            or bool(self.genres)
            or bool(self.tags)
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object from raw LLM output."""
    text = text.strip()
    text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _as_str_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_params(data: dict, fallback_query: str) -> GameQueryParams:
    max_price = data.get("max_price")
    if max_price is not None:
        max_price = float(max_price)

    min_rating_pct = data.get("min_rating_pct")
    if min_rating_pct is not None:
        min_rating_pct = float(min_rating_pct)

    min_review_count = data.get("min_review_count")
    if min_review_count is not None:
        min_review_count = int(min_review_count)

    semantic_query = (data.get("semantic_query") or fallback_query).strip()
    return GameQueryParams(
        max_price=max_price,
        min_rating_pct=min_rating_pct,
        min_review_count=min_review_count,
        genres=_as_str_list(data.get("genres")),
        tags=_as_str_list(data.get("tags")),
        semantic_query=semantic_query or fallback_query,
    )


@lru_cache(maxsize=1)
def load_generator():
    if not HF_TOKEN:
        raise ValueError(
            "Missing HF_TOKEN in .env — copy .env.example and add your token."
        )

    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=HF_TOKEN,
    )


def generate_text(
    generator,
    prompt: str,
    *,
    max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
    do_sample: bool = False,
) -> str:
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        do_sample=do_sample,
    )
    return result[0]["generated_text"].strip()


def parse_query_with_llm(generator, query: str) -> GameQueryParams:
    """Use Gemma to extract structured Steam search parameters from natural language."""
    prompt = f"""Extract Steam game search parameters from the user message.
Return JSON only with this schema:
{{
  "max_price": number or null,
  "min_rating_pct": number or null,
  "min_review_count": number or null,
  "genres": ["Genre", ...] or [],
  "tags": ["tag", ...] or [],
  "semantic_query": "short phrase for similarity search"
}}

Rules:
- max_price: cheap/free/under $X requests; 0 for free
- min_rating_pct: highly rated / X%+ positive requests
- min_review_count: set when user wants popular/well-reviewed games; else null
- genres: Steam genres (Action, RPG, Indie, ...)
- tags: gameplay tags (Roguelike, Open World, ...)
- semantic_query: rewrite the recommendation intent for vector search

User message:
{query}

JSON:"""

    raw = generate_text(generator, prompt, max_new_tokens=PARSE_MAX_NEW_TOKENS)
    data = _extract_json(raw)
    if not data:
        raise ValueError(f"LLM returned invalid JSON: {raw!r}")
    return _normalize_params(data, query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM query parsing.")
    parser.add_argument("query", help="Natural language game recommendation request")
    args = parser.parse_args()

    generator = load_generator()
    params = parse_query_with_llm(generator, args.query)
    print(json.dumps(params.to_dict(), indent=2))


if __name__ == "__main__":
    main()
