from __future__ import annotations

import base64
import io
import os
import re
import tempfile
from typing import Dict, List

import docx
import pdfplumber
from openai import OpenAI
from PIL import Image

from rag_system.config import VISION_MODEL_NAME


def image_to_base64(pil_img: Image.Image, quality: int = 85) -> str:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_pdf_text(file_path: str, client_ai: OpenAI) -> str:
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"

            for table in page.extract_tables():
                if not table:
                    continue
                md_table = "\n[文档结构化表格数据]\n"
                for i, row in enumerate(table):
                    clean_row = [
                        str(cell).replace("\n", " ").strip() if cell else ""
                        for cell in row
                    ]
                    md_table += "| " + " | ".join(clean_row) + " |\n"
                    if i == 0:
                        md_table += "|" + "|".join(["---"] * len(clean_row)) + "|\n"
                full_text += md_table + "\n\n"

            for image in page.images:
                x0, y0, x1, y1 = image["x0"], image["top"], image["x1"], image["bottom"]
                if (x1 - x0) < 100 or (y1 - y0) < 100:
                    continue

                try:
                    cropped_page = page.crop((x0, y0, x1, y1))
                    pil_img = cropped_page.to_image(resolution=150).original
                    base64_image = image_to_base64(pil_img)
                    prompt = """
                    你是一个专业的企业文档结构化解析引擎。这是一张从PDF中提取的插图。
                    请详细、客观地描述该图片的核心内容，以便用于后续的文字检索系统。
                    【规则】：
                    1. 如果是“流程图/架构图”：请用文本逻辑描述出先后步骤或层级关系。
                    2. 如果是“数据图表(柱状/折线/饼图)”：请提取出核心的数值、趋势或结论。
                    3. 如果是“产品/设备示意图”：请说明设备的名称及图中标注的各部件信息。
                    4. 忽略毫无意义的背景装饰。
                    """
                    response = client_ai.chat.completions.create(
                        model=VISION_MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                    )
                    description = response.choices[0].message.content.strip()
                    full_text += f"\n[第{page_num + 1}页-重要插图解析]\n{description}\n\n"
                except Exception as exc:
                    print(f"提取第 {page_num + 1} 页图片失败，跳过: {exc}")

    return full_text


def extract_docx_text(file_path: str) -> str:
    document = docx.Document(file_path)
    return "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )


def extract_image_text(file_path: str, client_ai: OpenAI) -> str:
    image = Image.open(file_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    max_size = 2048
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    prompt = """
    你是一个企业级的文档排版与内容还原专家。请精准提取这张图片中的所有文字内容。
    【严格要求】：
    1. 必须100%忠实于原图，不要遗漏、不要总结、不要瞎编（产生幻觉）。
    2. 【重中之重】：精准识别文章的层级结构。遇到各级标题（如“第一章”、“第一节”、“1.”、“1.1”、“步骤一”等），必须【独立成行】。
    3. 遇到正文的自然段落，请使用换行符分隔，不要把不同段落的内容揉成一段。
    """
    base64_image = image_to_base64(image)
    response = client_ai.chat.completions.create(
        model=VISION_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()


def extract_text_from_file(file_path: str, file_extension: str, client_ai: OpenAI) -> str:
    if file_extension == "pdf":
        return extract_pdf_text(file_path, client_ai)
    if file_extension == "docx":
        return extract_docx_text(file_path)
    if file_extension in ["jpg", "jpeg", "png"]:
        return extract_image_text(file_path, client_ai)
    raise ValueError("不支持的文件格式")


def split_into_hierarchical_chunks(full_text: str) -> List[Dict[str, str]]:
    normalized_text = re.sub(r" +", " ", full_text)
    raw_lines = [line.strip() for line in normalized_text.split("\n") if line.strip()]

    h1_pattern = re.compile(r"^(第[一二三四五六七八九十百]+[章节部分篇]|[\d]{1,2}、|[A-Z]\.)")
    h2_pattern = re.compile(r"^(\d{1,2}\.\d{1,2}|[\(（][一二三四五六七八九十][\)）])")

    current_h1 = "通用规范"
    current_h2 = ""
    parents: List[Dict[str, str]] = []
    current_parent_lines: List[str] = []

    for line in raw_lines:
        if h1_pattern.match(line):
            if current_parent_lines:
                context_tag = f"{current_h1}{' > ' + current_h2 if current_h2 else ''}"
                parents.append({"context": context_tag, "text": "\n".join(current_parent_lines)})
            current_h1 = line
            current_h2 = ""
            current_parent_lines = [line]
        elif h2_pattern.match(line):
            if current_parent_lines:
                context_tag = f"{current_h1}{' > ' + current_h2 if current_h2 else ''}"
                parents.append({"context": context_tag, "text": "\n".join(current_parent_lines)})
            current_h2 = line
            current_parent_lines = [line]
        else:
            current_parent_lines.append(line)

    if current_parent_lines:
        context_tag = f"{current_h1}{' > ' + current_h2 if current_h2 else ''}"
        parents.append({"context": context_tag, "text": "\n".join(current_parent_lines)})

    hierarchical_chunks: List[Dict[str, str]] = []
    window_size = 2

    for parent in parents:
        context_path = parent["context"]
        parent_text = parent["text"]

        lines = parent_text.split("\n")
        if len(lines) > 1 and lines[0].replace(" ", "") in context_path.replace(" ", ""):
            body_text = "\n".join(lines[1:])
        else:
            body_text = parent_text

        clean_body = body_text.replace("\n", "")
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？；])", clean_body)
            if sentence.strip()
        ]
        if not sentences:
            continue

        if len(sentences) < window_size:
            hierarchical_chunks.append(
                {"child": f"[{context_path}] {sentences[0]}", "parent": parent_text}
            )
        else:
            for index in range(len(sentences) - window_size + 1):
                child_text = " ".join(sentences[index : index + window_size])
                if len(child_text) > 10:
                    hierarchical_chunks.append(
                        {"child": f"[{context_path}] {child_text}", "parent": parent_text}
                    )

    return hierarchical_chunks


def process_uploaded_file(uploaded_file, client_ai: OpenAI) -> List[Dict[str, str]]:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        full_text = extract_text_from_file(temp_path, file_extension, client_ai)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return split_into_hierarchical_chunks(full_text)
