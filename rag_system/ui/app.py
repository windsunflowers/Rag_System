from __future__ import annotations

import concurrent.futures
import html
import re
import sys
import time
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from rag_system.config import (
    AVAILABLE_MODELS,
    CACHE_DIR,
    CUSTOM_CSS,
    DEFAULT_ANSWER_MODEL_INDEX,
    DEFAULT_GENERATOR_MODEL_INDEX,
    DEFAULT_JUDGE_MODEL_INDEX,
    MEDICAL_COLLECTION_NAME,
    MEDICAL_JSON_PATH,
    MODEL_INFO,
)
from rag_system.core.resources import configure_page, get_medical_collection, initialize_session_state, load_resources
from rag_system.core.state import AppResources
from rag_system.services.evaluation import generate_evaluation_dataset, process_single_evaluation
from rag_system.services.indexing import clear_index, load_knowledge_base, save_knowledge_base
from rag_system.services.medical_loader import load_medical_chunks
from rag_system.services.rag_pipeline import rag_pipeline_stream
from rag_system.core.state import KnowledgeBase
import jieba
from rank_bm25 import BM25Okapi


def clean_display_text(text: str) -> str:
    """把 HTML 标签清洗为普通换行，并规范化 Markdown 显示格式。"""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>\s*", "\n", str(text), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    # 1) 确保 # 标题标记后有空格，否则 Streamlit 不会识别为标题而挤成一段
    text = re.sub(r"(^|\n)(#{1,6})([^ #])", r"\1\2 \3", text, flags=re.MULTILINE)

    # 2) 修复压缩编号列表：中文或标点后的 "1. " 前补换行
    text = re.sub(r"([\u4e00-\u9fa5；;。])\s*(\d+\.\s+)", r"\1\n\2", text)

    # 3) 把开头可能出现的 #/## 大标题降级为 ###，避免首段字过大
    text = re.sub(r"^(#{1,2})\s+", "### ", text)

    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _active_collection():
    return get_medical_collection()


def _build_bm25_cache(chunks):
    child_texts = [item["child"] for item in chunks]
    tokenized_corpus = [list(jieba.cut(text)) for text in child_texts]
    bm25_engine = BM25Okapi(tokenized_corpus)
    return KnowledgeBase(chunks_data=chunks, child_texts=child_texts, bm25_engine=bm25_engine)


def _set_medical_kb_state(kb: KnowledgeBase, chunks: list) -> None:
    """统一设置医疗知识库会话状态并触发页面刷新。"""
    st.session_state["knowledge_base"] = kb
    st.session_state["chunks"] = chunks
    st.session_state["kb_mode"] = "medical"
    st.session_state.setdefault("messages", [])
    st.session_state.pop("test_cases", None)
    st.rerun()


def load_medical_knowledge_base(resources: AppResources, *, auto: bool = False) -> bool:
    """
    加载医疗知识库到会话状态。
    若索引已完整构建则直接读取缓存；否则执行一次增量构建批次。
    返回 True 表示已成功加载并可进入问答界面。
    """
    medical_collection = get_medical_collection()

    try:
        chunks = load_medical_chunks(MEDICAL_JSON_PATH)
    except Exception as exc:
        st.error(f"医疗数据加载失败：{exc}")
        return False

    if not chunks:
        st.warning("未能从医疗数据中提取到有效分块。")
        return False

    current_count = medical_collection.count()

    # Fully built: load from cache/collection directly
    if current_count == len(chunks):
        cached_kb = load_knowledge_base(MEDICAL_COLLECTION_NAME, CACHE_DIR)
        if cached_kb:
            if not auto:
                st.success(f"医疗知识库已就绪（共 {len(chunks)} 子块）")
            _set_medical_kb_state(cached_kb, cached_kb.chunks_data)
            return True

        kb = _build_bm25_cache(chunks)
        save_knowledge_base(kb, MEDICAL_COLLECTION_NAME, CACHE_DIR)
        if not auto:
            st.success(f"医疗知识库已就绪（共 {len(chunks)} 子块）")
        _set_medical_kb_state(kb, chunks)
        return True

    # Incremental build: add one batch to avoid process kills in constrained envs
    batch_size = 3000
    next_index = current_count
    end_index = min(next_index + batch_size, len(chunks))

    with st.spinner(f"正在增量构建索引 {current_count} → {end_index} / {len(chunks)} ..."):
        batch = chunks[next_index:end_index]
        child_texts = [item["child"] for item in batch]
        metadatas = [{"parent": item["parent"]} for item in batch]
        ids = [f"id_{index}" for index in range(next_index, end_index)]
        embeddings = resources.embed_model.encode(
            child_texts, batch_size=32, show_progress_bar=False
        ).tolist()
        medical_collection.add(
            documents=child_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    if end_index == len(chunks):
        kb = _build_bm25_cache(chunks)
        save_knowledge_base(kb, MEDICAL_COLLECTION_NAME, CACHE_DIR)
        st.success(f"医疗知识库构建完成（共 {len(chunks)} 子块）")
        _set_medical_kb_state(kb, chunks)
        return True

    # Still building
    st.info(f"已构建 {end_index}/{len(chunks)} 子块，继续自动构建中...")
    partial_chunks = chunks[:end_index]
    kb = _build_bm25_cache(partial_chunks)
    _set_medical_kb_state(kb, partial_chunks)
    return True


def auto_load_medical_knowledge_base(resources: AppResources) -> None:
    """页面启动时自动尝试加载医疗知识库；若已加载则跳过。"""
    if st.session_state.get("kb_mode") == "medical" and st.session_state.get("knowledge_base"):
        return

    with st.spinner("正在自动加载医疗知识库，请稍候..."):
        load_medical_knowledge_base(resources, auto=True)


def _model_icon_html(info: dict) -> str:
    """渲染灰色感叹号 + 悬停气泡（简介与输入输出价格）。"""
    return (
        '<span class="model-tip">ⓘ'
        '<span class="model-tip-box">'
        f'<b>{info["id"]}</b>（{info.get("name", "")}）· {info.get("modality", "")}<br>'
        f'{info.get("desc", "")}<br>'
        f'上下文：{info.get("context", "—")}<br>'
        f'输入：{info.get("input_price", "—")} 元 / 百万 tokens<br>'
        f'输出：{info.get("output_price", "—")} 元 / 百万 tokens'
        "</span></span>"
    )


def _render_model_select(label: str, default_index: int, key: str) -> str:
    """带右侧灰色感叹号（悬停弹简介/价格）的模型下拉。"""
    st.markdown(f"**{label}**")
    c1, c2 = st.columns([6, 1])
    with c1:
        sel = st.selectbox("", AVAILABLE_MODELS, index=default_index,
                           label_visibility="collapsed", key=key)
    with c2:
        st.markdown(_model_icon_html(MODEL_INFO[sel]), unsafe_allow_html=True)
    return sel


def render_sidebar(resources: AppResources):
    with st.sidebar:
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.header("系统模型配置")
        st.markdown("请选择相应模型；每个下拉**右侧灰色 ⓘ** 悬停可查看简介与价格。")

        selected_rag_model = _render_model_select("答题模型 (被测对象)", DEFAULT_ANSWER_MODEL_INDEX, "rag_model")
        selected_gen_model = _render_model_select("出题模型 (标准生成)", DEFAULT_GENERATOR_MODEL_INDEX, "gen_model")
        selected_judge_model = _render_model_select("裁判模型 (智能评分)", DEFAULT_JUDGE_MODEL_INDEX, "judge_model")

        st.markdown("---")
        st.header("医疗知识库")

        kb_mode = st.session_state.get("kb_mode")
        if kb_mode == "medical":
            chunks = st.session_state.get("chunks", [])
            st.success(
                f"当前模式：医疗知识库\n"
                f"疾病子块总数：{len(chunks)}"
            )
            if st.button("重新构建医疗知识库", use_container_width=True):
                clear_index(get_medical_collection())
                st.session_state["knowledge_base"] = None
                st.session_state["kb_mode"] = None
                st.session_state["chunks"] = []
                st.rerun()
        else:
            st.info("医疗知识库尚未加载")
            if st.button("一键加载 medical.json", use_container_width=True):
                load_medical_knowledge_base(resources)

    return selected_rag_model, selected_gen_model, selected_judge_model


def render_chat_tab(resources: AppResources, selected_rag_model: str) -> None:
    st.subheader("医疗知识问答")
    st.caption(f"当前使用的生成模型：{selected_rag_model}")

    messages = st.session_state.setdefault("messages", [])

    # 1) 先渲染全部历史消息
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(clean_display_text(message["content"]))

    # 2) 若最后一条是用户提问，则生成助手回复（生成完毕后会 rerun，使输入框回到最下方）
    if messages and messages[-1]["role"] == "user":
        _generate_assistant_response(resources, selected_rag_model, messages)
        return

    # 3) 输入框始终放在所有消息之后，即页面底部
    prompt = st.chat_input("请输入您关于文档的问题")
    if prompt:
        messages.append({"role": "user", "content": clean_display_text(prompt)})
        st.rerun()


def _generate_assistant_response(
    resources: AppResources, selected_rag_model: str, messages: list
) -> None:
    """为最后一条用户消息生成流式回复，并追加到会话历史中。"""
    with st.chat_message("assistant"):
        history_for_rag = messages[:-1]
        with st.spinner("正在检索知识库并思考..."):
            stream_generator, _ = rag_pipeline_stream(
                messages[-1]["content"],
                resources=resources,
                knowledge_base=st.session_state["knowledge_base"],
                collection=_active_collection(),
                history=history_for_rag,
                answer_model=selected_rag_model,
            )

            def cleaned_stream():
                for chunk in stream_generator:
                    yield clean_display_text(chunk)

            full_response = st.write_stream(cleaned_stream())

    messages.append({"role": "assistant", "content": clean_display_text(full_response)})
    st.rerun()


def render_evaluation_tab(
    resources: AppResources,
    selected_rag_model: str,
    selected_gen_model: str,
    selected_judge_model: str,
) -> None:
    st.subheader("RAG 系统多维度自动化测评 (RAGAs 标准)")

    # 评测集与去重记忆在会话内累积，便于多次生成凑足 100 条且不重复
    st.session_state.setdefault("test_cases", [])
    st.session_state.setdefault("generated_queries", [])

    col_input, col_action = st.columns([1, 1])
    with col_input:
        num_cases = st.number_input("设定评测样本数量", min_value=1, max_value=50, value=5)
    with col_action:
        st.write("")
        st.write("")
        generate_btn = st.button("生成评测数据集", use_container_width=True)

    if generate_btn:
        with st.spinner(f"正在调用 {selected_gen_model} 构建标准参考题库..."):
            new_cases = generate_evaluation_dataset(
                st.session_state["chunks"],
                resources=resources,
                num_cases=num_cases,
                gen_model=selected_gen_model,
                avoid=st.session_state.get("generated_queries", []),
            )
            if new_cases:
                st.session_state["test_cases"].extend(new_cases)
                st.session_state["generated_queries"].extend(c["query"] for c in new_cases)
                st.success(
                    f"新增 {len(new_cases)} 条，当前共 {len(st.session_state['test_cases'])} 条。"
                )
            else:
                st.warning("本次未生成有效题目，请重试。")

    test_cases = st.session_state.get("test_cases", [])
    if not test_cases:
        st.info("尚未生成测试集，请在上方设定数量后点击「生成评测数据集」。")
        return

    st.markdown(f"#### 当前评测数据集（共 {len(test_cases)} 条）")
    _, col_clear = st.columns([3, 1])
    with col_clear:
        if st.button("清空测试集", use_container_width=True):
            st.session_state["test_cases"] = []
            st.session_state["generated_queries"] = []
            st.rerun()
    st.dataframe(test_cases, use_container_width=True)

    if not st.button("执行 RAGAS 标准评测", type="primary", use_container_width=True):
        return

    st.markdown("---")
    scores_total = {"fa": 0, "ar": 0, "cr": 0, "cp": 0}
    total_cases = len(test_cases)

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"正在启动多线程并发测评 (共 {total_cases} 题)...")

    results_list = [None] * total_cases
    completed_count = 0
    knowledge_base = st.session_state["knowledge_base"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(
                process_single_evaluation,
                case,
                selected_rag_model,
                selected_judge_model,
                resources,
                knowledge_base,
                _active_collection(),
            ): index
            for index, case in enumerate(test_cases)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results_list[index] = future.result()
            except Exception as exc:
                st.error(f"测试集 {index + 1} 测评发生异常: {exc}")
                results_list[index] = {
                    "query": test_cases[index].get("query"),
                    "ground_truth": test_cases[index].get("ground_truth"),
                    "rag_ans": "请求失败或超时",
                    "eval_result": {},
                }

            completed_count += 1
            progress_bar.progress(completed_count / total_cases)
            status_text.text(f"测评进度: {completed_count} / {total_cases} 完成")

    status_text.text("测评数据计算完毕，正在生成报告...")
    time.sleep(0.5)
    status_text.empty()

    for index, result in enumerate(results_list):
        if not result:
            continue

        eval_result = result["eval_result"]
        default_eval = {"score": 0, "reason": "解析失败或未返回"}
        faithfulness = eval_result.get("faithfulness", default_eval)
        answer_relevancy = eval_result.get("answer_relevancy", default_eval)
        context_recall = eval_result.get("context_recall", default_eval)
        context_precision = eval_result.get("context_precision", default_eval)

        scores_total["fa"] += faithfulness.get("score", 0)
        scores_total["ar"] += answer_relevancy.get("score", 0)
        scores_total["cr"] += context_recall.get("score", 0)
        scores_total["cp"] += context_precision.get("score", 0)

        with st.expander(f"测试集 {index + 1}: {clean_display_text(result['query'])}", expanded=False):
            st.markdown(f"**【基准答案】** {clean_display_text(result['ground_truth'])}")
            st.markdown(f"**【模型输出】** ({selected_rag_model}) {clean_display_text(result['rag_ans'])}")
            st.markdown("---")

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("忠实性", f"{faithfulness.get('score', 0)}%")
            sc2.metric("回答相关性", f"{answer_relevancy.get('score', 0)}%")
            sc3.metric("上下文召回率", f"{context_recall.get('score', 0)}%")
            sc4.metric("上下文精确率", f"{context_precision.get('score', 0)}%")

            st.markdown(
                f"> 裁判评语 ({selected_judge_model}): \n"
                f"> - 忠实性: {clean_display_text(faithfulness.get('reason'))}\n"
                f"> - 回答相关性: {clean_display_text(answer_relevancy.get('reason'))}\n"
                f"> - 上下文召回率: {clean_display_text(context_recall.get('reason'))}\n"
                f"> - 上下文精确率: {clean_display_text(context_precision.get('reason'))}"
            )

    st.markdown("### RAGAS 核心指标大盘")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("忠实性\n(Faithfulness)", f"{scores_total['fa'] / total_cases:.1f}%")
    col_m2.metric("回答相关性\n(Answer Relevancy)", f"{scores_total['ar'] / total_cases:.1f}%")
    col_m3.metric("上下文召回率\n(Context Recall)", f"{scores_total['cr'] / total_cases:.1f}%")
    col_m4.metric("上下文精确率\n(Context Precision)", f"{scores_total['cp'] / total_cases:.1f}%")


def main() -> None:
    configure_page()

    try:
        resources = load_resources()
    except RuntimeError as exc:
        st.error(f"系统配置错误：{exc}")
        st.stop()

    initialize_session_state()

    # 页面启动即自动加载医疗知识库（若已构建则秒级完成）
    auto_load_medical_knowledge_base(resources)

    st.title("基于通义千问的医疗 RAG 检索增强生成系统")
    st.markdown("---")

    selected_rag_model, selected_gen_model, selected_judge_model = render_sidebar(resources)

    kb = st.session_state.get("knowledge_base")
    kb_mode = st.session_state.get("kb_mode")
    if kb and kb_mode:
        tab1, tab2 = st.tabs(["交互式文档问答", "系统多维准确度测评"])
        with tab1:
            render_chat_tab(resources, selected_rag_model)
        with tab2:
            render_evaluation_tab(
                resources,
                selected_rag_model,
                selected_gen_model,
                selected_judge_model,
            )
    else:
        st.info("医疗知识库正在初始化，请稍候；若长时间无响应，请检查左侧“一键加载 medical.json”按钮。")


if __name__ == "__main__":
    main()
