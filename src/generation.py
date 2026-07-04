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
    CONVERSATION_MEMORY_TURNS,
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
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    return [str(item).strip().lower() for item in value if str(item).strip()]


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
  "genres": ["genre", ...] or [],
  "tags": ["tag", ...] or [],
  "semantic_query": "short phrase for similarity search"
}}

Rules:
- Use null for any field NOT explicitly mentioned in the user message
- Do NOT guess or infer filters the user did not ask for
- max_price: set only when price is mentioned
  - "free" or "free to play" → 0
  - "under $15" / "below $20" → that number
  - "cheap" alone → null (not 0)
- min_rating_pct: set only when user mentions ratings, % positive, or "highly rated"
  - Do NOT set min_rating_pct for price-only or genre-only requests
- min_review_count: set only when user mentions popularity or review volume
- genres: Steam store genres only — action, adventure, casual, indie, rpg, simulation, strategy, sports, racing
- tags: gameplay/style tags — roguelike, story rich, open world, cozy, survival, metroidvania, souls-like
  - roguelike, story rich, cozy, etc. go in tags, NEVER in genres
- semantic_query: rewrite the recommendation intent for vector search (drop price/rating constraints)
- All genre and tag strings must be lowercase

Examples:
User: free roguelike
{{"max_price": 0, "min_rating_pct": null, "min_review_count": null, "genres": [], "tags": ["roguelike"], "semantic_query": "roguelike games"}}

User: cheap indie roguelike under $15
{{"max_price": 15, "min_rating_pct": null, "min_review_count": null, "genres": ["indie"], "tags": ["roguelike"], "semantic_query": "indie roguelike games"}}

User: story rich RPG with 90% positive reviews
{{"max_price": null, "min_rating_pct": 90, "min_review_count": null, "genres": ["rpg"], "tags": ["story rich"], "semantic_query": "story rich RPG"}}

User message:
{query}

JSON:"""

    raw = generate_text(generator, prompt, max_new_tokens=PARSE_MAX_NEW_TOKENS)
    data = _extract_json(raw)
    if not data:
        raise ValueError(f"LLM returned invalid JSON: {raw!r}")
    return _normalize_params(data, query)


@dataclass
class ChatTurn:
    role: str
    content: str


def trim_history(
    history: list[ChatTurn],
    max_turns: int = CONVERSATION_MEMORY_TURNS,
) -> list[ChatTurn]:
    """Keep the last N user/assistant turn pairs."""
    return history[-(max_turns * 2) :]


def format_history(history: list[ChatTurn]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)


def _format_list_field(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def build_context_from_chunks(chunks, games_df=None) -> str:
    """Format retrieved games into a context block for the RAG prompt."""
    from src.ingestion import get_description_for_app_id

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        app_id = str(meta.get("app_id", ""))
        name = meta.get("name", "?")
        price = meta.get("price", "?")
        rating = meta.get("rating_pct", "?")
        description = chunk.text
        if games_df is not None and app_id:
            full_description = get_description_for_app_id(games_df, app_id)
            if full_description:
                description = full_description

        blocks.append(
            f"Game {index}: {name}\n"
            f"Price: ${price} | Rating: {rating}% positive\n"
            f"Genres: {_format_list_field(meta.get('genres'))}\n"
            f"Tags: {_format_list_field(meta.get('tags'))}\n"
            f"Description: {description}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(
    query: str,
    chunks,
    history: list[ChatTurn] | None = None,
    *,
    games_df=None,
) -> str:
    """Build a Steam advisor prompt with retrieved context and chat history."""
    context = build_context_from_chunks(chunks, games_df)
    history = trim_history(history or [])
    history_block = format_history(history)

    parts = [
        "You are a helpful Steam game advisor. Recommend games from the context only.",
        "If the context does not contain suitable games, say so briefly.",
        "",
    ]
    if history_block:
        parts.extend(["Previous conversation:", history_block, ""])
    parts.extend(
        [
            "Context:",
            context or "No matching games were found.",
            "",
            f"User: {query}",
            "Assistant:",
        ]
    )
    return "\n".join(parts)


def answer_with_rag(
    generator,
    query: str,
    chunks,
    history: list[ChatTurn] | None = None,
    *,
    games_df=None,
) -> str:
    prompt = build_rag_prompt(query, chunks, history, games_df=games_df)
    return generate_text(generator, prompt)


def main() -> None:
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] not in ("parse", "chat"):
        sys.argv.insert(1, "parse")

    parser = argparse.ArgumentParser(description="Gemma query parsing and RAG chat.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Test LLM query parsing")
    parse_parser.add_argument(
        "query", help="Natural language game recommendation request"
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Retrieve games and answer with RAG"
    )
    chat_parser.add_argument("query", help="User message")
    chat_parser.add_argument(
        "--mode",
        choices=["llm", "none"],
        default="llm",
        help="Retrieval mode: llm (filtered) or none (baseline)",
    )

    args = parser.parse_args()

    if args.command == "parse":
        generator = load_generator()
        params = parse_query_with_llm(generator, args.query)
        print(json.dumps(params.to_dict(), indent=2))
        return

    from src.ingestion import ensure_games_csv, load_games_csv
    from src.retrieval import retrieve

    generator = load_generator()
    games_df = load_games_csv(ensure_games_csv())
    result = retrieve(args.query, filter_mode=args.mode, generator=generator)
    answer = answer_with_rag(
        generator,
        args.query,
        result.chunks,
        games_df=games_df,
    )
    print("Answer:\n", answer)


if __name__ == "__main__":
    main()
