"""Streamlit chat UI for the Steam Game Advisor."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.generation import (
    ChatTurn,
    answer_with_rag,
    load_generator,
    trim_history,
)
from src.ingestion import ensure_games_csv, load_embedding_model, load_games_csv
from src.retrieval import retrieve

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="Steam Game Advisor", layout="wide")


@st.cache_resource(show_spinner="Loading Gemma (first run may take a few minutes)...")
def get_generator():
    return load_generator()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    return load_embedding_model()


@st.cache_data(show_spinner="Loading games catalog...")
def get_games_df():
    return load_games_csv(ensure_games_csv())


def _history_from_session() -> list[ChatTurn]:
    return [
        ChatTurn(role=message["role"], content=message["content"])
        for message in st.session_state.messages
    ]


def main() -> None:
    st.title("Steam Game Advisor")
    st.caption(
        "Ask for game recommendations. Retrieval can use LLM filters or naive RAG."
    )

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        st.error("Missing HF_TOKEN in .env — copy .env.example and add your token.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = None

    filter_mode = st.sidebar.selectbox(
        "Retrieval mode",
        options=["llm", "none"],
        format_func=lambda value: (
            "LLM + Chroma filters" if value == "llm" else "Naive RAG (baseline)"
        ),
    )

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_retrieval = None
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What kind of game are you looking for?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                generator = get_generator()
                games_df = get_games_df()
                embed_model = get_embed_model()
                history = trim_history(_history_from_session()[:-1])

                result = retrieve(
                    prompt,
                    filter_mode=filter_mode,
                    generator=generator if filter_mode == "llm" else None,
                    embed_model=embed_model,
                )
                st.session_state.last_retrieval = {
                    "params": result.params.to_dict(),
                    "chroma_where": result.chroma_where,
                    "chunks": [
                        {
                            "name": chunk.metadata.get("name"),
                            "app_id": chunk.metadata.get("app_id"),
                            "score": chunk.score,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        }
                        for chunk in result.chunks
                    ],
                }

                if not result.chunks:
                    answer = (
                        "I couldn't find any games matching your request. "
                        "Try broadening your filters or rephrasing."
                    )
                else:
                    answer = answer_with_rag(
                        generator,
                        prompt,
                        result.chunks,
                        history=history,
                        games_df=games_df,
                    )

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.last_retrieval:
        with st.sidebar.expander("Retrieval details", expanded=False):
            debug = st.session_state.last_retrieval
            st.markdown("**Parsed params**")
            st.json(debug["params"])
            st.markdown("**Chroma where**")
            st.json(debug["chroma_where"])
            st.markdown("**Retrieved games**")
            for chunk in debug["chunks"]:
                name = chunk.get("name", "?")
                score = chunk.get("score", 0)
                st.markdown(f"- **{name}** (score {score:.3f})")


if __name__ == "__main__":
    main()
