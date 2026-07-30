from __future__ import annotations

import os

import chromadb
import streamlit as st
import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_system.config import (
    LLM_BASE_URL,
    CACHE_DIR,
    CUSTOM_CSS,
    EMBEDDING_MODEL_NAME,
    MEDICAL_COLLECTION_NAME,
    PAGE_LAYOUT,
    PAGE_TITLE,
    PERSIST_DIR,
    RERANK_MODEL_NAME,
    VECTOR_COLLECTION_NAME,
)
from rag_system.core.state import AppResources


def configure_page() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("knowledge_base", None)
    st.session_state.setdefault("kb_mode", None)
    st.session_state.setdefault("chunks", [])


@st.cache_resource
def load_resources() -> AppResources:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未在环境变量中找到 DASHSCOPE_API_KEY")

    client_ai = OpenAI(
        api_key=api_key,
        base_url=LLM_BASE_URL,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    rerank_model = CrossEncoder(RERANK_MODEL_NAME, device=device)

    db_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = db_client.get_or_create_collection(name=VECTOR_COLLECTION_NAME)

    return AppResources(
        client_ai=client_ai,
        embed_model=embed_model,
        rerank_model=rerank_model,
        collection=collection,
    )


def get_medical_collection():
    db_client = chromadb.PersistentClient(path=PERSIST_DIR)
    return db_client.get_or_create_collection(name=MEDICAL_COLLECTION_NAME)
