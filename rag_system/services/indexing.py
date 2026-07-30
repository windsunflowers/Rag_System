from __future__ import annotations

import os
import pickle
from typing import List

import jieba
from rank_bm25 import BM25Okapi

from rag_system.core.state import KnowledgeBase


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clear_index(collection, batch_size: int = 1000) -> None:
    existing_ids = collection.get().get("ids", [])
    for i in range(0, len(existing_ids), batch_size):
        collection.delete(ids=existing_ids[i : i + batch_size])


def _get_cache_paths(collection_name: str, cache_dir: str) -> tuple:
    _ensure_dir(cache_dir)
    return (
        os.path.join(cache_dir, f"{collection_name}_chunks_data.pkl"),
        os.path.join(cache_dir, f"{collection_name}_child_texts.pkl"),
        os.path.join(cache_dir, f"{collection_name}_bm25_tokens.pkl"),
    )


def save_knowledge_base(knowledge_base: KnowledgeBase, collection_name: str, cache_dir: str) -> None:
    chunks_path, texts_path, bm25_path = _get_cache_paths(collection_name, cache_dir)
    with open(chunks_path, "wb") as f:
        pickle.dump(knowledge_base.chunks_data, f)
    with open(texts_path, "wb") as f:
        pickle.dump(knowledge_base.child_texts, f)
    with open(bm25_path, "wb") as f:
        pickle.dump([list(jieba.cut(text)) for text in knowledge_base.child_texts], f)


def load_knowledge_base(collection_name: str, cache_dir: str) -> KnowledgeBase | None:
    chunks_path, texts_path, bm25_path = _get_cache_paths(collection_name, cache_dir)
    if not all(os.path.exists(p) for p in (chunks_path, texts_path, bm25_path)):
        return None

    with open(chunks_path, "rb") as f:
        chunks_data = pickle.load(f)
    with open(texts_path, "rb") as f:
        child_texts = pickle.load(f)
    with open(bm25_path, "rb") as f:
        tokenized_corpus = pickle.load(f)

    bm25_engine = BM25Okapi(tokenized_corpus)
    return KnowledgeBase(
        chunks_data=chunks_data,
        child_texts=child_texts,
        bm25_engine=bm25_engine,
    )


def build_index(
    chunks_data,
    embed_model,
    collection,
    collection_name: str,
    cache_dir: str,
    batch_size: int = 1000,
) -> KnowledgeBase:
    clear_index(collection)

    child_texts = [item["child"] for item in chunks_data]

    # Encode and add to Chroma in a single loop to keep memory low.
    encode_batch_size = 32
    total = len(chunks_data)
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_texts = child_texts[i:batch_end]
        batch_metadatas = [{"parent": item["parent"]} for item in chunks_data[i:batch_end]]
        batch_ids = [f"id_{index}" for index in range(i, batch_end)]
        batch_embeddings = embed_model.encode(
            batch_texts,
            batch_size=encode_batch_size,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
        if (i // batch_size + 1) % 10 == 0 or batch_end == total:
            print(f"  indexed {batch_end}/{total} chunks", flush=True)

    tokenized_corpus = [list(jieba.cut(text)) for text in child_texts]
    bm25_engine = BM25Okapi(tokenized_corpus)
    knowledge_base = KnowledgeBase(
        chunks_data=chunks_data,
        child_texts=child_texts,
        bm25_engine=bm25_engine,
    )
    save_knowledge_base(knowledge_base, collection_name, cache_dir)
    return knowledge_base


def get_top_bm25_indices(knowledge_base: KnowledgeBase, query: str, top_k: int = 10) -> List[int]:
    tokenized_query = list(jieba.cut(query))
    scores = knowledge_base.bm25_engine.get_scores(tokenized_query)
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return [index for index in ranked[:top_k] if scores[index] > 0]
