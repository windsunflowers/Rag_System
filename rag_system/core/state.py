from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


@dataclass
class AppResources:
    client_ai: Any
    embed_model: Any
    rerank_model: Any
    collection: Any


@dataclass
class KnowledgeBase:
    chunks_data: List[Dict[str, str]]
    child_texts: List[str]
    bm25_engine: BM25Okapi
