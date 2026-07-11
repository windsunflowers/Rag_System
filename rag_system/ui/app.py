from __future__ import annotations

import concurrent.futures
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
    DEFAULT_ANSWER_MODEL_INDEX,
    DEFAULT_GENERATOR_MODEL_INDEX,
    DEFAULT_JUDGE_MODEL_INDEX,
    SUPPORTED_FILE_TYPES,
)
from rag_system.core.resources import configure_page, initialize_session_state, load_resources
from rag_system.core.state import AppResources
from rag_system.services.document_processing import process_uploaded_file
from rag_system.services.evaluation import generate_evaluation_dataset, process_single_evaluation
from rag_system.services.indexing import build_index, clear_index
from rag_system.services.rag_pipeline import rag_pipeline_stream


def reset_loaded_documents(resources: AppResources) -> None:
    for key in ["current_file_hash", "chunks", "test_cases", "messages", "knowledge_base"]:
        st.session_state.pop(key, None)
    clear_index(resources.collection)


def handle_uploaded_files(uploaded_files, resources: AppResources) -> None:
    if not uploaded_files:
        reset_loaded_documents(resources)
        return

    file_hash = "|".join(f"{file.name}_{file.size}" for file in uploaded_files)
    if st.session_state.get("current_file_hash") == file_hash:
        return

    with st.spinner("系统正在读取并合并向量化文档..."):
        all_chunks = []
        for uploaded_file in uploaded_files:
            try:
                chunks = process_uploaded_file(uploaded_file, resources.client_ai)
            except Exception as exc:
                st.error(f"文件 {uploaded_file.name} 解析失败：{exc}")
                continue
            if chunks:
                all_chunks.extend(chunks)

        if not all_chunks:
            st.warning("未能在文档中提取到有效文本。")
            return

        knowledge_base = build_index(all_chunks, resources.embed_model, resources.collection)
        st.session_state["chunks"] = all_chunks
        st.session_state["knowledge_base"] = knowledge_base
        st.session_state["current_file_hash"] = file_hash
        st.session_state["messages"] = []
        st.session_state.pop("test_cases", None)


def render_sidebar(resources: AppResources):
    with st.sidebar:
        st.header("系统模型配置")
        st.markdown("请选择相应模型：")

        selected_rag_model = st.selectbox(
            "答题模型 (被测对象)",
            AVAILABLE_MODELS,
            index=DEFAULT_ANSWER_MODEL_INDEX,
        )
        selected_gen_model = st.selectbox(
            "出题模型 (标准生成)",
            AVAILABLE_MODELS,
            index=DEFAULT_GENERATOR_MODEL_INDEX,
        )
        selected_judge_model = st.selectbox(
            "裁判模型 (智能评分)",
            AVAILABLE_MODELS,
            index=DEFAULT_JUDGE_MODEL_INDEX,
        )

        st.markdown("---")
        st.header("文件解析")
        st.markdown("支持格式：`.pdf`, `.docx`, `.jpg`, `.png`")

        uploaded_files = st.file_uploader(
            "拖拽文件至此区域自动解析 (支持多文件)",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
        )

        handle_uploaded_files(uploaded_files, resources)

        if uploaded_files and "knowledge_base" in st.session_state:
            st.success("多文档知识库已就绪")
            st.info(
                f"当前已加载文档数: {len(uploaded_files)} 个\n"
                f"融合片段总数: {len(st.session_state['chunks'])}"
            )

    return selected_rag_model, selected_gen_model, selected_judge_model


def render_chat_tab(resources: AppResources, selected_rag_model: str) -> None:
    st.subheader("文档内容问答")
    st.caption(f"当前使用的生成模型：{selected_rag_model}")

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("请输入您关于文档的问题")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history_for_rag = st.session_state["messages"][:-1]
        with st.spinner("正在检索知识库并思考..."):
            stream_generator, _ = rag_pipeline_stream(
                prompt,
                resources=resources,
                knowledge_base=st.session_state["knowledge_base"],
                history=history_for_rag,
                answer_model=selected_rag_model,
            )
            full_response = st.write_stream(stream_generator)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})


def render_evaluation_tab(
    resources: AppResources,
    selected_rag_model: str,
    selected_gen_model: str,
    selected_judge_model: str,
) -> None:
    st.subheader("RAG 系统多维度自动化测评 (RAGAs 标准)")

    col_input, col_action = st.columns([1, 1])
    with col_input:
        num_cases = st.number_input("设定评测样本数量", min_value=1, max_value=50, value=5)
    with col_action:
        st.write("")
        st.write("")
        generate_btn = st.button("生成评测数据集", use_container_width=True)

    if generate_btn:
        with st.spinner(f"正在调用 {selected_gen_model} 构建标准参考题库..."):
            test_cases = generate_evaluation_dataset(
                st.session_state["chunks"],
                resources=resources,
                num_cases=num_cases,
                gen_model=selected_gen_model,
            )
            st.session_state["test_cases"] = test_cases
            st.success("测试用例构建完成。")

    test_cases = st.session_state.get("test_cases", [])
    if not test_cases:
        return

    st.markdown("#### 当前评测数据集")
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

        with st.expander(f"测试集 {index + 1}: {result['query']}", expanded=False):
            st.markdown(f"**【基准答案】** {result['ground_truth']}")
            st.markdown(f"**【模型输出】** ({selected_rag_model}) {result['rag_ans']}")
            st.markdown("---")

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("忠实性", f"{faithfulness.get('score', 0)}%")
            sc2.metric("回答相关性", f"{answer_relevancy.get('score', 0)}%")
            sc3.metric("上下文召回率", f"{context_recall.get('score', 0)}%")
            sc4.metric("上下文精确率", f"{context_precision.get('score', 0)}%")

            st.markdown(
                f"> 裁判评语 ({selected_judge_model}): \n"
                f"> - 忠实性: {faithfulness.get('reason')}\n"
                f"> - 回答相关性: {answer_relevancy.get('reason')}\n"
                f"> - 上下文召回率: {context_recall.get('reason')}\n"
                f"> - 上下文精确率: {context_precision.get('reason')}"
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

    st.title("基于通义千问的 RAG 检索增强生成系统")
    st.markdown("---")

    selected_rag_model, selected_gen_model, selected_judge_model = render_sidebar(resources)

    if st.session_state.get("knowledge_base") and resources.collection.count() > 0:
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
        st.info("请先在左侧区域上传并解析文档，系统将自动解锁问答与测评功能。")
