from __future__ import annotations

from typing import List

import jieba
from rank_bm25 import BM25Okapi

from rag_system.core.state import KnowledgeBase


def clear_index(collection) -> None:
    existing_ids = collection.get().get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)


def build_index(chunks_data, embed_model, collection) -> KnowledgeBase:
    clear_index(collection)

    child_texts = [item["child"] for item in chunks_data]
    metadatas = [{"parent": item["parent"]} for item in chunks_data]
    ids = [f"id_{index}" for index in range(len(chunks_data))]
    embeddings = embed_model.encode(child_texts).tolist()

    collection.add(
        documents=child_texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    tokenized_corpus = [list(jieba.cut(text)) for text in child_texts]
    bm25_engine = BM25Okapi(tokenized_corpus)
    return KnowledgeBase(
        chunks_data=chunks_data,
        child_texts=child_texts,
        bm25_engine=bm25_engine,
    )


def get_top_bm25_indices(knowledge_base: KnowledgeBase, query: str, top_k: int = 10) -> List[int]:
    tokenized_query = list(jieba.cut(query))
    scores = knowledge_base.bm25_engine.get_scores(tokenized_query)
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return [index for index in ranked[:top_k] if scores[index] > 0]
