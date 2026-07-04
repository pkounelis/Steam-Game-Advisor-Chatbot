# Steam Game Advisor Chatbot

**Author:** Panagiotis Kounelis

A local RAG chatbot that recommends Steam games from natural-language requests. The pipeline parses user intent with Gemma, filters candidates in ChromaDB, retrieves relevant game descriptions, and generates grounded answers.

## Features

- **LLM query parsing** — Gemma extracts price, rating, genre, and tag filters plus a semantic search phrase
- **Metadata-filtered retrieval** — Chroma `where` clauses narrow results before vector search
- **Naive RAG baseline** — compare `filter_mode=llm` vs `filter_mode=none` for evaluation
- **Conversation memory** — last 5 user/assistant turns passed into the RAG prompt
- **Streamlit UI** — chat interface with retrieval debug panel

## Architecture

Two pipelines: build the index offline, then answer queries online.

**Index (run once)**

```
FronkonGames CSV  →  chunk descriptions  →  MiniLM embed  →  ChromaDB
```

**Query (per message)**

```
User query
    │
    ├─ llm mode ──► Gemma parse ──► Chroma where ──┐
    │                                               │
    └─ none mode ───────────────────────────────────┤
                                                    ▼
                                         vector search (MiniLM)
                                                    │
                                              dedup by app_id
                                                    │
                                    Gemma RAG (+ 5-turn history)
                                                    │
                                                 answer
```

- **`llm`** — Gemma extracts price, rating, genre, and tag filters into a Chroma `where` clause, then searches on `semantic_query`.
- **`none`** — skips parsing and filters; pure semantic search baseline.

Chunk metadata: `app_id`, `name`, `price`, `rating_pct`, `review_count`, `genres`, `tags`. RAG context uses full descriptions from the CSV when available.

## Tech stack

| Layer | Choice |
|-------|--------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB ≥ 1.5 |
| LLM | `google/gemma-2-2b-it` (Hugging Face) |
| UI | Streamlit |
| Data | FronkonGames Steam games CSV |

## Prerequisites

- Python 3.11+
- CUDA GPU recommended (Gemma 2B fits on ~4 GB VRAM with `device_map="auto"`)
- [Hugging Face account](https://huggingface.co/) and access to Gemma (accept the model license)
- HF token with read access

## Setup

```bash
git clone <repo-url>
cd Steam-Game-Advisor-Chatbot

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
```

### Build the vector index

First run downloads the dataset (~126k games) and builds the Chroma index. Use `--limit` for a quick smoke test.

```bash
# Full index (slow; ~30+ min depending on hardware)
python -m src.ingestion --force

# Quick dev index
python -m src.ingestion --force --limit 1000
```

Artifacts are written to `data_source/steam_games.csv` and `vector_db/` (both gitignored).

## Usage

### Streamlit chat (recommended)

`streamlit run` adds `src/` to `sys.path`, not the project root — set `PYTHONPATH`:

```bash
PYTHONPATH=. streamlit run src/app.py
```

On Windows (PowerShell):

```powershell
$env:PYTHONPATH="."; streamlit run src/app.py
```

Use the sidebar to switch between **LLM + Chroma filters** and **Naive RAG (baseline)**. Expand **Retrieval details** to inspect parsed params and retrieved games.

### CLI — query parsing

```bash
python -m src.generation parse 'cheap indie roguelike under $15'
```

Use single quotes in bash so `$15` is not expanded by the shell.

### CLI — retrieval

```bash
python -m src.retrieval 'story rich RPG' --mode llm
python -m src.retrieval 'story rich RPG' --mode none
```

### CLI — full RAG chat

```bash
python -m src.generation chat 'cozy farming sim' --mode llm
```

## Project structure

```
Steam-Game-Advisor-Chatbot/
├── src/
│   ├── config.py       # paths, model IDs, hyperparameters
│   ├── ingestion.py    # CSV download, chunking, Chroma indexing
│   ├── generation.py   # Gemma parse + RAG generation
│   ├── retrieval.py    # Chroma where + vector search
│   └── app.py          # Streamlit UI
├── data_source/        # steam_games.csv (gitignored)
├── vector_db/          # Chroma persistence (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Key settings in `src/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `NAIVE_N_RESULTS` | 4 | Games returned after dedup |
| `RETRIEVAL_OVERFETCH_MULTIPLIER` | 5 | Over-fetch chunks before dedup |
| `DEFAULT_MIN_REVIEW_COUNT` | 50 | Review floor when filtering by rating |
| `CONVERSATION_MEMORY_TURNS` | 5 | Chat history window |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 100 | Description chunking |

## Dataset

Games and metadata come from the [FronkonGames Steam games dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) on Hugging Face. Descriptions are embedded; structured fields are stored as Chroma metadata for filtering.

## Future work

- **Live Steam data** — optional refresh of price, discount, and review stats via the Steam Web API instead of static CSV snapshots.
- **Retrieval evaluation** — `eval_queries.jsonl` with ground-truth filters and an `evaluate_retrieval.py` script to compare `filter_mode=llm` vs `filter_mode=none` (precision, recall, MRR).
- **Guardrails** — refuse off-topic requests, detect empty retrieval, and constrain answers to retrieved context only.
