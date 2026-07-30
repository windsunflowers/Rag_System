from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Optional

from rag_system.config import DEFAULT_GENERATOR_MODEL_NAME, DEFAULT_JUDGE_MODEL_NAME
from rag_system.core.state import AppResources, KnowledgeBase
from rag_system.services.rag_pipeline import rag_pipeline


def _parse_child_tag(child: str):
    """从子块文本 `[疾病·维度] 内容` 解析出 (疾病, 维度, 内容)。"""
    m = re.match(r"^\[(.+?)·(.+?)\]\s*(.*)", child, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3)
    return None


def generate_evaluation_dataset(
    chunks_data: List[Dict],
    resources: AppResources,
    num_cases: int = 5,
    gen_model: str = DEFAULT_GENERATOR_MODEL_NAME,
    avoid: Optional[List[str]] = None,
) -> List[Dict]:
    """生成评测数据集。

    关键改进：
    1. 以「子块」为采样单元，按 [疾病·维度] 标签解析，保证每题来自不同疾病、
       且维度尽量不同，彻底解决题目扎堆在同一疾病/同一维度的问题。
    2. 标准答案要求完整覆盖关键支撑要点（含必要医学细节与数值），不再过度精简。
    3. 支持 avoid 参数传入历史已生成问题，实现跨批次去重。
    """
    # 1) 解析所有子块的疾病/维度
    entries: List[tuple] = []
    for item in chunks_data:
        parsed = _parse_child_tag(item.get("child", ""))
        if parsed:
            entries.append(parsed)
    if not entries:
        return []

    # 2) 按疾病分组，挑选 num_cases 个不同疾病，每个疾病优先选未用过的维度
    by_disease: Dict[str, List[tuple]] = {}
    for disease, field, text in entries:
        by_disease.setdefault(disease, []).append((field, text))

    diseases = list(by_disease.keys())
    random.shuffle(diseases)
    if num_cases > len(diseases):
        num_cases = len(diseases)

    used_dims: set = set()
    selected: List[tuple] = []
    for d in diseases[:num_cases]:
        choices = by_disease[d]
        # 优先选尚未出现过的维度，使维度尽量分散
        choices_sorted = sorted(choices, key=lambda ft: 0 if ft[0] not in used_dims else 1)
        field, text = choices_sorted[0]
        used_dims.add(field)
        selected.append((d, field, text))

    # 3) 拼装素材，每段标注疾病与维度
    sample_parts = [
        f"【素材：{disease} - {field}】\n{text}" for (disease, field, text) in selected
    ]
    sample_text = "\n\n".join(sample_parts)

    avoid_text = "\n".join(f"- {q}" for q in (avoid or [])) if avoid else "无"

    prompt = f"""
你是一个严谨的医疗考试出题专家。请基于下面提供的若干段素材，生成 {num_cases} 个事实性问答对。

【硬性要求】
1. 一一对应：第 i 段素材【素材：X - Y】用于生成第 i 个问题，问题须聚焦该素材标注的「维度」（如标注「典型症状」就问症状，「推荐检查」就问检查项目）。
2. 答案不可过度精简：标准答案须完整覆盖该问题的关键支撑要点，包含必要的医学细节、数值范围与限定条件，避免一句话带过；但必须严格基于素材原文，不得编造或引入素材之外的信息。
3. 疾病与维度不重复：{num_cases} 个问题须分别来自不同疾病，并尽量覆盖不同维度（如症状、检查、治疗、预防、病因、并发症、科室、饮食等）。
4. 避免与以下已出问题雷同（疾病、维度、句式都不要重复）：
{avoid_text}
5. 只输出 JSON 数组，不要任何解释文字。格式：
[
  {{"query": "问题", "ground_truth": "基于素材的完整标准答案"}}
]

【素材】
{sample_text}
"""
    try:
        response = resources.client_ai.chat.completions.create(
            model=gen_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        parsed = json.loads(content)
        if isinstance(parsed, dict) and len(parsed.keys()) == 1:
            parsed = list(parsed.values())[0]
        if not isinstance(parsed, list):
            return []
        # 用真实选中的疾病/维度回填，保证去重与展示准确
        for case, (d, f, _) in zip(parsed, selected):
            case["disease"] = d
            case["dimension"] = f
        return parsed
    except Exception:
        return []


# 与 rag_pipeline.MEDICAL_SYSTEM_PROMPT 第 7 条保持一致：固定免责声明起止句。
# 评测时从送评的 answer 中剥离该段，避免每轮必带的合规套话污染
# faithfulness / answer_relevancy 两项"只看 Answer"的指标。
_DISCLAIMER_START = "以上内容基于医学知识库整理"
_DISCLAIMER_END = "请咨询执业医师。"


def strip_disclaimer(text: str) -> str:
    """从医疗回答中剥离固定免责声明（用于评测），UI 展示不调用本函数。"""
    if not text:
        return text
    # 主匹配：从免责声明起始句到结束句（含末尾句号），跨换行。
    cleaned = re.sub(
        r"\s*" + re.escape(_DISCLAIMER_START) + r"[\s\S]*?" + re.escape(_DISCLAIMER_END) + r"\s*",
        "\n",
        text,
    )
    # 兜底：极端情况下结束标点缺失，退化为删除到文本末尾。
    if _DISCLAIMER_START in cleaned and _DISCLAIMER_END not in cleaned:
        cleaned = re.sub(r"\s*" + re.escape(_DISCLAIMER_START) + r"[\s\S]*$", "", cleaned)
    cleaned = cleaned.strip()
    # 保护：若剥离后几乎为空（异常情形），回退使用原始回答，避免评测失真。
    return cleaned if len(cleaned) >= 5 else text


def evaluate_answer(
    query: str,
    generated_answer: str,
    ground_truth: str,
    context: str,
    resources: AppResources,
    judge_model: str = DEFAULT_JUDGE_MODEL_NAME,
) -> Dict:
    prompt = f"""
你是专业的 RAG 系统评测专家。请依据 RAGAS 风格，对 Answer 进行四维打分。

【硬性要求】
1. 只输出 JSON，不要输出任何额外文字。
2. 所有 score 必须是 0-100 的整数。
3. reason 必须简短，控制在 1 句话。
4. 不要把不同维度混在一起评分。

【输入】
- Query: {query}
- Context: {context}
- Ground Truth: {ground_truth}
- Answer: {generated_answer}

【评分步骤】
第一步：先将 Ground Truth 拆成若干关键事实点。
第二步：判断 Context 覆盖了多少事实点，用于 context_recall。
第三步：判断 Answer 是否被 Context 支持，用于 faithfulness。
第四步：判断 Answer 是否直接回应 Query，用于 answer_relevancy。
第五步：判断 Context 是否以有用信息为主，用于 context_precision。

【四个维度定义】
1. faithfulness：Answer 是否严格基于 Context。
2. answer_relevancy：Answer 是否直接、明确、完整回应 Query。
3. context_recall：Context 是否覆盖 Ground Truth 的关键事实点。
4. context_precision：Context 中有用信息是否占主导。

【输出格式】
{{
  "faithfulness": {{"score": 85, "reason": "简短理由"}},
  "answer_relevancy": {{"score": 92, "reason": "简短理由"}},
  "context_recall": {{"score": 60, "reason": "简短理由"}},
  "context_precision": {{"score": 35, "reason": "简短理由"}}
}}
"""

    try:
        response = resources.client_ai.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        return json.loads(content)
    except Exception as exc:
        error_payload = {"score": 0, "reason": f"模型未按规范输出或解析异常: {exc}"}
        return {
            "faithfulness": error_payload,
            "answer_relevancy": error_payload,
            "context_recall": error_payload,
            "context_precision": error_payload,
        }


def process_single_evaluation(
    case: Dict,
    rag_model: str,
    judge_model: str,
    resources: AppResources,
    knowledge_base: KnowledgeBase,
    collection,
) -> Dict:
    query = case.get("query")
    ground_truth = case.get("ground_truth")

    rag_answer, retrieved_context = rag_pipeline(
        query,
        resources=resources,
        knowledge_base=knowledge_base,
        collection=collection,
        answer_model=rag_model,
    )
    # 评测用剥离免责声明的干净回答，避免固定套话污染忠实度/相关性；
    # UI 展示仍返回原始 rag_answer（含免责声明）。
    eval_answer = strip_disclaimer(rag_answer)
    evaluation_result = evaluate_answer(
        query,
        eval_answer,
        ground_truth,
        retrieved_context,
        resources=resources,
        judge_model=judge_model,
    )

    return {
        "query": query,
        "ground_truth": ground_truth,
        "rag_ans": rag_answer,
        "eval_result": evaluation_result,
    }
