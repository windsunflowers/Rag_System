from __future__ import annotations

import html
import json
import os
import re
from typing import Dict, List

from rag_system.config import CHILD_SLIDE_WINDOW, MAX_PARENT_TEXT_LENGTH, MEDICAL_JSON_PATH


def _clean_text(text: str) -> str:
    if not text:
        return ""
    if isinstance(text, list):
        text = "、".join(str(item) for item in text if item)
    text = str(text).replace("\\n", "\n").strip()
    # Normalize various HTML line breaks to actual newlines
    text = re.sub(r"<br\s*/?>\s*", "\n", text, flags=re.IGNORECASE)
    # Remove other HTML tags (e.g. <b>, <p>, <li>) while preserving inner text
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities such as &nbsp; &lt; &gt; &amp;
    text = html.unescape(text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[内容过长，已按安全上限截断]"


def _slide_window(text: str, window: int = CHILD_SLIDE_WINDOW) -> List[str]:
    """长文本按字数滑窗切分，保持子块检索粒度。使用 20% 重叠以减少子块总量。"""
    if len(text) <= window:
        return [text] if text else []
    chunks: List[str] = []
    stride = int(window * 0.8)
    start = 0
    while start < len(text):
        end = start + window
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks


def _build_parent_text(record: Dict, name: str) -> str:
    sections = [
        ("疾病名称", name),
        ("所属分类", _clean_text(record.get("category", ""))),
        ("就诊科室", _clean_text(record.get("cure_department", ""))),
        ("疾病描述", _clean_text(record.get("desc", ""))),
        ("典型症状", _clean_text(record.get("symptom", ""))),
        ("病因", _clean_text(record.get("cause", ""))),
        ("预防措施", _clean_text(record.get("prevent", ""))),
        ("推荐检查", _clean_text(record.get("check", ""))),
        ("治疗方式", _clean_text(record.get("cure_way", ""))),
        ("治疗周期", _clean_text(record.get("cure_lasttime", ""))),
        ("治愈概率", _clean_text(record.get("cured_prob", ""))),
        ("参考费用", _clean_text(record.get("cost_money", ""))),
        ("医保状态", _clean_text(record.get("yibao_status", ""))),
        ("患病概率", _clean_text(record.get("get_prob", ""))),
        ("传染性", _clean_text(record.get("get_way", ""))),
        ("并发症", _clean_text(record.get("acompany", ""))),
        ("推荐药物", _clean_text(record.get("recommand_drug", ""))),
        ("药物明细", _clean_text(record.get("drug_detail", ""))),
    ]
    lines = []
    for label, value in sections:
        if value:
            lines.append(f"【{label}】{value}")
    full_text = "\n".join(lines)
    return _truncate(full_text, MAX_PARENT_TEXT_LENGTH)


def _build_child_chunks(record: Dict, name: str) -> List[Dict[str, str]]:
    """把疾病记录拆成多个字段级子块，长字段做滑窗。"""
    parent_text = _build_parent_text(record, name)
    if not parent_text:
        return []

    # 子块仅保留高频检索字段；完整信息保留在父块中
    fields = {
        "疾病描述": _clean_text(record.get("desc", "")),
        "典型症状": _clean_text(record.get("symptom", "")),
        "病因": _clean_text(record.get("cause", "")),
        "推荐检查": _clean_text(record.get("check", "")),
        "治疗": "、".join(
            filter(
                None,
                [
                    _clean_text(record.get("cure_way", "")),
                    _clean_text(record.get("cure_lasttime", "")),
                    _clean_text(record.get("cured_prob", "")),
                    _clean_text(record.get("cost_money", "")),
                ],
            )
        ),
        "就诊科室": _clean_text(record.get("cure_department", "")),
    }

    child_chunks: List[Dict[str, str]] = []
    for field_name, field_value in fields.items():
        if not field_value:
            continue
        tag = f"[{name}·{field_name}]"
        # 长字段仅保留一个检索子块（按安全长度截断），父块仍保留完整内容
        field_text = field_value if len(field_value) <= CHILD_SLIDE_WINDOW else field_value[:CHILD_SLIDE_WINDOW]
        child_chunks.append({"child": f"{tag} {field_text}", "parent": parent_text})

    return child_chunks


def load_medical_chunks(json_path: str = MEDICAL_JSON_PATH) -> List[Dict[str, str]]:
    """加载 medical.json（JSON Lines 格式），返回父子分块列表。"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到医疗数据文件：{json_path}")

    all_chunks: List[Dict[str, str]] = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            name = _clean_text(record.get("name", ""))
            if not name:
                continue

            child_chunks = _build_child_chunks(record, name)
            all_chunks.extend(child_chunks)

    return all_chunks
