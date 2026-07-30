from __future__ import annotations

import json
import re

from typing import Dict, Iterable, List, Optional, Tuple

from rag_system.config import (
    DEFAULT_ANSWER_MODEL_NAME,
    QUERY_REWRITE_MODEL_NAME,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
)
from rag_system.core.state import AppResources, KnowledgeBase
from rag_system.services.indexing import get_top_bm25_indices


MEDICAL_SYSTEM_PROMPT = """你是一位严谨的医学知识库助手。请严格遵循以下规则：

1. 【核心规则】必须仅基于用户提供的【参考内容】进行回答，不得引入参考内容之外的知识或猜测。
2. 【回答风格】使用标准、规范的医学行业术语；首次出现较生僻术语时可附简短释义。
3. 【排版要求】
   - 回答开头不要写任何大标题，严禁使用 # 或 ##，小节标题统一使用 `### `（### 后必须跟一个空格）。
   - 每个要点必须单独成行，使用 `- ` 或 `1. ` 列出；禁止把多个症状、检查项或治疗方式写成连续长句。
   - 不要在段落外用 `**` 包裹整段内容，仅对关键医学术语加粗。
4. 【回答结构】请按以下顺序分小节输出，每节内容用分点列出：
   ### 相关疾病概述
   - 疾病定义与范畴
   - 主要病因与发病机制要点

   ### 典型症状与鉴别要点
   - 症状1（附简要说明）
   - 症状2（附简要说明）

   ### 推荐检查项目
   - 检查项1
   - 检查项2

   ### 治疗方式、周期及预后
   - 治疗方式1
   - 周期与预后说明

   ### 预防措施
   - 预防措施1

   ### 引用说明
   - 信息来源疾病名称
5. 【上下文理解】若用户当前问题是对前文的承接（如“吃什么呢”“怎么治”“它有什么症状”“该病的检查”或带“它/这个/该病”等省略/指代），请结合历史对话明确其指代的疾病或症状，并基于该疾病/症状的参考内容回答，不要泛化到其他无关疾病。
6. 【资料缺失】若参考内容无法回答某一部分，请明确说明“根据现有知识库未收录该信息”。
7. 【免责声明】每次回答末尾必须固定输出：
   "以上内容基于医学知识库整理，仅用于医学知识学习与参考，不构成诊断、治疗建议或处方依据。如有身体不适，请咨询执业医师。"
"""


# 医疗问答意图维度（与 medical_loader 子块维度标签、父块字段名对齐）
MEDICAL_INTENT_DIMENSIONS = [
    "疾病描述", "典型症状", "病因", "推荐检查", "治疗", "就诊科室", "其他",
]


def rewrite_query(query: str, history: Optional[List[Dict]], resources: AppResources) -> Tuple[str, str]:
    """医疗场景 Query 改写：口语→标准术语、多轮补全，并识别意图维度。

    返回 (rewritten_query, intent_dimension)。
    intent_dimension ∈ MEDICAL_INTENT_DIMENSIONS；无法判定时回退 "其他"。
    """
    history_text = ""
    if history:
        # 改写只取最近一轮对话，避免过长历史干扰指代消解
        history_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in history[-2:]
        )

    dim_list = "、".join(MEDICAL_INTENT_DIMENSIONS)
    rewrite_prompt = f"""你是一位医疗检索改写专家。请根据历史对话和当前问题，生成一个自包含的标准医学术语检索式，并判断用户问题聚焦的医学维度。

改写要求：
- 若当前问题是独立问句（含具体症状、疾病名、身体部位等），直接改写为标准医学检索式。
- 若当前问题是承接上文的省略/指代/短句（如“吃什么呢”“怎么治”“什么原因”“它有什么症状”“该病的检查”“饮食注意事项”等），必须结合历史对话中的核心疾病/症状，将问题补全为完整检索式。
  示例：
  - 上文讨论“急性中耳炎”，当前问“吃什么呢” → 改写为“急性中耳炎 饮食建议 宜吃 忌吃 食物”。
  - 上文讨论“偏头痛”，当前问“做什么检查” → 改写为“偏头痛 推荐检查 影像学检查”。
- 将口语症状转为标准医学术语（例如“头疼”改为“头痛”，“想吐”改为“恶心呕吐”，“浑身没劲”改为“乏力”）。
- 保留症状的持续时间和严重程度关键词（如“三天”、“剧烈”）。
- 在改写结果末尾补充2-3个可能相关的疾病名称作为扩展召回词（用空格分隔）。
- 判断意图维度：用户核心想了解的是以下哪一个维度（只能选一个）：{dim_list}。
  维度判断示例：“推荐进行哪些检查” → 推荐检查；“怎么治疗” → 治疗；“有什么症状” → 典型症状；“什么原因引起” → 病因；“挂什么科” → 就诊科室；“这个病是什么” → 疾病描述；无法明确归入上述任一维度 → 其他。

【历史对话】：
{history_text if history_text else "无"}

【用户原问题】：{query}

只输出 JSON，不要解释，格式：
{{"rewritten_query": "改写后的医学检索式", "intent_dimension": "维度"}}"""

    try:
        response = resources.client_ai.chat.completions.create(
            model=QUERY_REWRITE_MODEL_NAME,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        return _parse_rewrite_json(content)
    except Exception:
        return query, "其他"


def _parse_rewrite_json(content: str) -> Tuple[str, str]:
    """从改写模型输出解析 (rewritten_query, intent_dimension)，失败时回退。"""
    try:
        text = content
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        parsed = json.loads(text)
        rewritten = str(parsed.get("rewritten_query", "")).strip()
        intent = str(parsed.get("intent_dimension", "其他")).strip()
        if intent not in MEDICAL_INTENT_DIMENSIONS:
            intent = "其他"
        return (rewritten or "", intent)
    except Exception:
        # 退化：模型直接输出纯文本检索式，意图归其他
        return (content.strip(), "其他")


def _parse_dimension(child: str) -> Optional[str]:
    """从子块文本 `[疾病·维度] 内容` 解析维度标签。"""
    if not child:
        return None
    m = re.match(r"^\[(.+?)·(.+?)\]\s*(.*)", child, flags=re.DOTALL)
    return m.group(2).strip() if m else None


def _retrieve_hits(
    search_query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    collection,
    top_k: int,
) -> List[Tuple[str, Optional[str]]]:
    """单路混合检索，返回 (parent, dimension) 命中列表（未跨路去重）。"""
    hits: List[Tuple[str, Optional[str]]] = []

    # 向量召回：从 Chroma id（id_<index>）反查 chunks_data 取子块维度
    query_emb = resources.embed_model.encode([search_query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    vector_ids = results.get("ids", [[]])[0] if results.get("ids") else []
    for cid in vector_ids:
        try:
            idx = int(str(cid).split("_")[1])
            item = knowledge_base.chunks_data[idx]
        except (ValueError, IndexError, KeyError, TypeError):
            continue
        hits.append((item.get("parent"), _parse_dimension(item.get("child", ""))))

    # BM25 召回
    for idx in get_top_bm25_indices(knowledge_base, search_query, top_k):
        try:
            item = knowledge_base.chunks_data[idx]
        except (IndexError, KeyError):
            continue
        hits.append((item.get("parent"), _parse_dimension(item.get("child", ""))))

    return hits


def retrieve_chunk_records(
    query: str,
    rewritten_query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    collection,
    top_k: int = RETRIEVAL_TOP_K,
) -> Dict[str, set]:
    """双链路召回并按父块聚合维度命中：{parent_text: {维度集合}}。"""
    hits = _retrieve_hits(query, resources, knowledge_base, collection, top_k)
    hits += _retrieve_hits(rewritten_query, resources, knowledge_base, collection, top_k)
    parent_dims: Dict[str, set] = {}
    for parent, dim in hits:
        if not parent:
            continue
        parent_dims.setdefault(parent, set())
        if dim:
            parent_dims[parent].add(dim)
    return parent_dims


# 意图维度 → 父块中需保留的字段（与 _build_parent_text 字段名对齐）
DIMENSION_PARENT_FIELDS: Dict[str, List[str]] = {
    "疾病描述": ["疾病描述"],
    "典型症状": ["典型症状"],
    "病因": ["病因"],
    "推荐检查": ["推荐检查"],
    "治疗": ["治疗方式", "治疗周期", "治愈概率", "参考费用"],
    "就诊科室": ["就诊科室"],
}


def rerank_parents(
    search_query: str,
    parents: List[str],
    resources: AppResources,
    top_n: int = RERANK_TOP_N,
) -> List[str]:
    """CrossEncoder 重排，返回打分最高的 top_n 个父块。"""
    if not parents:
        return []
    pairs = [[search_query, parent] for parent in parents]
    scores = resources.rerank_model.predict(pairs)
    ranked = sorted(zip(parents, scores), key=lambda item: item[1], reverse=True)
    return [parent for parent, _ in ranked[:top_n]]


def build_messages(query: str, context: str, history: Optional[List[Dict]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": f"参考内容：\n{context}\n\n当前问题：{query}"})
    return messages


def prepare_context(
    query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    collection,
    history: Optional[List[Dict]] = None,
) -> Tuple[str, str]:
    search_query, intent = rewrite_query(query, history, resources)
    parent_dims = retrieve_chunk_records(query, search_query, resources, knowledge_base, collection)
    all_parents = list(parent_dims.keys())

    if intent in DIMENSION_PARENT_FIELDS and all_parents:
        matched = [p for p in all_parents if intent in parent_dims[p]]
        other = [p for p in all_parents if intent not in parent_dims[p]]
        if matched:
            # 维度命中优先：在命中集合内重排取满 top_n，不足再从其他补充
            final_parents = rerank_parents(search_query, matched, resources, RERANK_TOP_N)
            need = max(0, RERANK_TOP_N - len(final_parents))
            if need:
                final_parents += rerank_parents(search_query, other, resources, need)
        else:
            final_parents = rerank_parents(search_query, all_parents, resources, RERANK_TOP_N)
    else:
        final_parents = rerank_parents(search_query, all_parents, resources, RERANK_TOP_N)

    # 维度感知排序：命中维度的父块优先，但返回完整父块，避免 ground_truth 跨维度事实被裁剪丢失
    final_context_text = "\n\n---\n\n".join(final_parents)
    return search_query, final_context_text


def rag_pipeline(
    query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    collection,
    history: Optional[List[Dict]] = None,
    answer_model: str = DEFAULT_ANSWER_MODEL_NAME,
) -> Tuple[str, str]:
    _, final_context_text = prepare_context(query, resources, knowledge_base, collection, history)
    response = resources.client_ai.chat.completions.create(
        model=answer_model,
        messages=build_messages(query, final_context_text, history),
        temperature=0.3,
    )
    return response.choices[0].message.content, final_context_text


def rag_pipeline_stream(
    query: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    collection,
    history: Optional[List[Dict]] = None,
    answer_model: str = DEFAULT_ANSWER_MODEL_NAME,
) -> Tuple[Iterable[str], str]:
    _, final_context_text = prepare_context(query, resources, knowledge_base, collection, history)
    response = resources.client_ai.chat.completions.create(
        model=answer_model,
        messages=build_messages(query, final_context_text, history),
        temperature=0.3,
        stream=True,
    )

    def stream_generator():
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return stream_generator(), final_context_text
