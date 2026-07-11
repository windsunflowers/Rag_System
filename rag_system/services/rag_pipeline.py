from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from rag_system.config import QUERY_REWRITE_MODEL_NAME
from rag_system.core.state import AppResources, KnowledgeBase
from rag_system.services.indexing import get_top_bm25_indices


SYSTEM_PROMPT = """你是一个智能文档助手。请遵循以下规则回答：
1. 【核心规则】：如果用户的问题是关于文档内容的，请务必严格且仅根据提供的【参考内容】进行回答。
2. 【历史对话】：如果用户询问的是你们之间的对话历史（例如“我上一个问题是什么”、“根据上文…”），请直接基于对话历史回答，此时无需受参考内容的限制。
3. 如果既无法在参考内容中找到答案，也无法在历史记录中找到答案，请诚实告知。"""


def rewrite_query(query: str, history: Optional[List[Dict]], resources: AppResources) -> str:
    if not history:
        return query

    recent_history = "\n".join(
        f"{message['role']}: {message['content']}" for message in history[-4:]
    )
    rewrite_prompt = f"""
    你是一个意图识别专家。请结合以下历史对话，将用户的最新简短问题重写为一个完整、独立的问题。
    如果用户的原问题已经很完整，或者与历史对话无关，请直接输出原问题。不要输出任何额外解释。
    【历史对话】:\n{recent_history}\n【最新用户问题】: {query}\n重写后的独立问题：
    """

    try:
        response = resources.client_ai.chat.completions.create(
            model=QUERY_REWRITE_MODEL_NAME,
            messages=[{"role": "user", "content": rewrite_prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query


def retrieve_parent_contexts(
    search_query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    top_k: int = 10,
) -> List[str]:
    query_emb = resources.embed_model.encode([search_query]).tolist()
    results = resources.collection.query(query_embeddings=query_emb, n_results=top_k)
    vector_metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    bm25_metadatas = [
        {"parent": knowledge_base.chunks_data[index]["parent"]}
        for index in get_top_bm25_indices(knowledge_base, search_query, top_k)
    ]

    unique_parents: List[str] = []
    seen = set()
    for metadata in (vector_metadatas or []) + bm25_metadatas:
        parent_text = metadata.get("parent")
        if parent_text and parent_text not in seen:
            seen.add(parent_text)
            unique_parents.append(parent_text)
    return unique_parents


def rerank_context(search_query: str, unique_parents: List[str], resources: AppResources) -> str:
    if not unique_parents:
        return ""

    pairs = [[search_query, parent] for parent in unique_parents]
    scores = resources.rerank_model.predict(pairs)
    ranked = sorted(zip(unique_parents, scores), key=lambda item: item[1], reverse=True)

    final_context = [parent for parent, score in ranked[:2] if score > -5.0]
    if not final_context and ranked:
        final_context = [ranked[0][0]]
    return "\n".join(final_context)


def build_messages(query: str, context: str, history: Optional[List[Dict]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": f"参考内容：\n{context}\n\n当前问题：{query}"})
    return messages


def prepare_context(
    query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    history: Optional[List[Dict]] = None,
) -> Tuple[str, str]:
    search_query = rewrite_query(query, history, resources)
    unique_parents = retrieve_parent_contexts(search_query, resources, knowledge_base)
    final_context_text = rerank_context(search_query, unique_parents, resources)
    return search_query, final_context_text


def rag_pipeline(
    query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    history: Optional[List[Dict]] = None,
    answer_model: str = "qwen-turbo",
) -> Tuple[str, str]:
    _, final_context_text = prepare_context(query, resources, knowledge_base, history)
    response = resources.client_ai.chat.completions.create(
        model=answer_model,
        messages=build_messages(query, final_context_text, history),
    )
    return response.choices[0].message.content, final_context_text


def rag_pipeline_stream(
    query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    history: Optional[List[Dict]] = None,
    answer_model: str = "qwen-turbo",
) -> Tuple[Iterable[str], str]:
    _, final_context_text = prepare_context(query, resources, knowledge_base, history)
    response = resources.client_ai.chat.completions.create(
        model=answer_model,
        messages=build_messages(query, final_context_text, history),
        stream=True,
    )

    def stream_generator():
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return stream_generator(), final_context_text
