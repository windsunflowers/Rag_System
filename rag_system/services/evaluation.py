from __future__ import annotations

import json
import random
import re
from typing import Dict, List

from rag_system.core.state import AppResources, KnowledgeBase
from rag_system.services.rag_pipeline import rag_pipeline


def generate_evaluation_dataset(
    chunks_data: List[Dict],
    resources: AppResources,
    num_cases: int = 5,
    gen_model: str = "qwen-plus",
) -> List[Dict]:
    unique_parents = list({item["parent"] for item in chunks_data})
    if not unique_parents:
        return []

    sample_text = "\n".join(random.sample(unique_parents, min(10, len(unique_parents))))
    prompt = f"""
    你是一个严谨的考试出题专家。请基于以下提供的文本，生成 {num_cases} 个具体的事实性问答对。
    要求：
    1. 问题必须能从文本中找到明确答案。
    2. 标准答案必须准确、精炼。
    3. 只输出 JSON 数组格式，不要包含其他任何解释文字。格式如下：
    [
      {{"query": "问题1", "ground_truth": "标准答案1"}}
    ]
    文本内容：\n{sample_text}
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
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def evaluate_answer(
    query: str,
    generated_answer: str,
    ground_truth: str,
    context: str,
    resources: AppResources,
    judge_model: str = "qwen-max",
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
) -> Dict:
    query = case.get("query")
    ground_truth = case.get("ground_truth")

    rag_answer, retrieved_context = rag_pipeline(
        query,
        resources=resources,
        knowledge_base=knowledge_base,
        answer_model=rag_model,
    )
    evaluation_result = evaluate_answer(
        query,
        rag_answer,
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
