import os
import re
import json
import random
import tempfile
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import torch
import pdfplumber

# ==========================================
# 1. 页面与全局配置
# ==========================================
st.set_page_config(page_title="智能文档解析与评测系统", layout="wide")

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    st.error("系统配置错误：未在环境变量中找到 DASHSCOPE_API_KEY")
    st.stop()

client_ai = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ==========================================
# 2. 模型与知识库初始化
# ==========================================
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
    rerank_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=device)
    db_client = chromadb.EphemeralClient()
    collection = db_client.create_collection(name="rag_docs_standard")
    return embed_model, rerank_model, collection

embed_model, rerank_model, collection = load_models()

# ==========================================
# 3. 多格式文档处理模块
# ==========================================
def extract_text_from_file(file_path: str, file_extension: str) -> str:
    """根据文件类型提取文本"""
    full_text = ""
    
    if file_extension == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    
    elif file_extension in ['docx', 'doc']:
        import docx
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text:
                full_text += para.text + "\n"
                
    elif file_extension in ['jpg', 'jpeg', 'png']:
        from PIL import Image
        import pytesseract
        try:
            # 指定语言为简体中文和英文
            img = Image.open(file_path)
            full_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        except Exception as e:
            st.error("图片解析失败。请确保已安装 Tesseract-OCR 并配置了中文语言包。")
            raise e
            
    else:
        raise ValueError("不支持的文件格式")
        
    return full_text

def process_uploaded_file(uploaded_file):
    """处理上传文件并进行语义分块"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        full_text = extract_text_from_file(tmp_path, file_extension)
    except Exception as e:
        st.error(f"提取文本时发生错误：{str(e)}")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    full_text = re.sub(r'\s+', ' ', full_text).strip()
    sentences = re.split(r'(?<=[。！？；])', full_text)
    
    chunks = []
    current_sentences = []
    current_length = 0
    chunk_size = 500
    overlap_size = 100

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: 
            continue
        sentence_len = len(sentence)
        if current_length + sentence_len <= chunk_size:
            current_sentences.append(sentence)
            current_length += sentence_len
        else:
            if current_sentences:
                chunks.append("".join(current_sentences))
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_sentences):
                if overlap_length + len(s) > overlap_size: 
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)
            current_sentences = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_sentences)

    if current_sentences:
        chunks.append("".join(current_sentences))
    
    return [c for c in chunks if len(c) > 20]

def build_index(chunks: List[str]):
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        
    embeddings = embed_model.encode(chunks).tolist()
    ids = [f"id_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

# ==========================================
# 4. RAG 与 三模型调度机制
# ==========================================
def rag_pipeline(query: str, history: List[Dict] = None) -> str:
    """支持多轮对话上下文的 RAG 核心管道"""
    
    # 1. 独立问题重写 (Query Rewrite)
    search_query = query
    if history and len(history) > 0:
        # 提取最近的几轮对话作为重写参考
        recent_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
        rewrite_prompt = f"""
        你是一个意图识别专家。请结合以下历史对话，将用户的最新简短问题重写为一个完整、独立的问题。
        如果用户的原问题已经很完整，或者与历史对话无关，请直接输出原问题。
        注意：只需输出重写后的问题文本，不要输出任何额外解释。
        
        【历史对话】:
        {recent_history}
        
        【最新用户问题】: {query}
        
        重写后的独立问题：
        """
        try:
            rewrite_response = client_ai.chat.completions.create(
                model="qwen-turbo", # 使用极速模型进行重写
                messages=[{"role": "user", "content": rewrite_prompt}]
            )
            search_query = rewrite_response.choices[0].message.content.strip()
            print(f"原始问题: {query} -> 重写问题: {search_query}") # 在终端打印观察效果
        except Exception as e:
            print(f"重写失败，使用原问题: {e}")

    # 2. 向量检索 (使用重写后的完整问题)
    query_emb = embed_model.encode([search_query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=10)
    
    if not results["documents"] or not results["documents"][0]:
        return "知识库为空或未检索到相关内容。"
        
    retrieved_docs = results["documents"][0]
    
    # 3. 语义重排序
    pairs = [[search_query, doc] for doc in retrieved_docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    final_context = [item[0] for item in ranked[:3]]
    
    # 4. 最终生成答案 (同时喂给大模型参考片段和历史对话)
    system_prompt = "你是一个专业的文档助手。请仅根据提供的参考内容回答用户问题。如果参考内容不足以回答，请诚实告知。"
    context_text = chr(10).join(final_context)
    user_prompt = f"参考内容：\n{context_text}\n\n当前问题：{query}"
    
    # 构建包含历史记录的消息列表
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-4:]) # 保留最近的两轮对话记忆
    messages.append({"role": "user", "content": user_prompt})

    response = client_ai.chat.completions.create(
        model="qwen-plus", # 答题模型可以稍微聪明一点
        messages=messages,
    )
    return response.choices[0].message.content
def generate_evaluation_dataset(chunks: List[str], num_cases=5) -> List[Dict]:
    """模型二：出题模型 (qwen-plus)"""
    sample_text = "\n".join(random.sample(chunks, min(10, len(chunks))))
    prompt = f"""
    你是一个严谨的考试出题专家。请基于以下提供的文本，生成 {num_cases} 个具体的事实性问答对。
    要求：
    1. 问题必须能从文本中找到明确答案。
    2. 标准答案必须准确、精炼。
    3. 只输出 JSON 数组格式，不要包含其他任何解释文字。格式如下：
    [
      {{"query": "问题1", "ground_truth": "标准答案1"}},
      {{"query": "问题2", "ground_truth": "标准答案2"}}
    ]
    
    文本内容：
    {sample_text}
    """
    try:
        response = client_ai.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        parsed = json.loads(content)
        if isinstance(parsed, dict) and len(parsed.keys()) == 1:
            parsed = list(parsed.values())[0]
        return parsed
    except Exception as e:
        st.error(f"题库生成发生异常: {e}")
        return []

def evaluate_with_qwen_max(query: str, generated_answer: str, ground_truth: str) -> Dict:
    """模型三：裁判模型 (qwen-max)"""
    prompt = f"""
    你是一个严谨的阅卷专家。请根据【用户问题】和【标准答案】，对【模型生成的答案】进行评分(0-10分)。
    核心事实一致即可得高分。
    
    【用户问题】: {query}
    【标准答案】: {ground_truth}
    【模型答案】: {generated_answer}

    必须严格输出 JSON 格式，包含 "score" (整数) 和 "reason" (简短解释)。
    """
    try:
        response = client_ai.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": 0, "reason": f"评估发生异常: {str(e)}"}

# ==========================================
# 5. UI 界面层
# ==========================================
st.title("文档解析与智能评测系统")
st.markdown("---")

# 侧边栏：文件管理
with st.sidebar:
    st.header("文件解析中心")
    st.markdown("支持格式：`.pdf`, `.docx`, `.jpg`, `.jpeg`, `.png`")
    uploaded_file = st.file_uploader("请将文件拖拽至下方", type=["pdf", "docx", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if st.button("执行解析与向量化建库", use_container_width=True):
            with st.spinner("系统正在提取并处理文档内容..."):
                chunks = process_uploaded_file(uploaded_file)
                if chunks:
                    st.session_state['chunks'] = chunks
                    build_index(chunks)
                    st.success("知识库构建完毕，系统已就绪。")
                else:
                    st.warning("未能在文档中提取到有效文本。")

# 主界面：功能分签
if 'chunks' in st.session_state and collection.count() > 0:
    tab1, tab2 = st.tabs(["交互式文档问答", "系统准确度测评"])
    
    # ---------------- 第一部分：交互问答 ----------------
    with tab1:
        st.subheader("文档内容问答")
        
        # 初始化对话历史
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 展示历史对话
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 接收用户输入
        if prompt := st.chat_input("请输入您关于文档的问题"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("模型思考中..."):
                    history_for_rag = st.session_state.messages[:-1] 
                    response = rag_pipeline(prompt, history=history_for_rag)
                    st.markdown(response)

    # ---------------- 第二部分：准确度测评 ----------------
    with tab2:
        st.subheader("RAG 回答准确度自动化测评")
        
        col_input, col_action = st.columns([1, 1])
        with col_input:
            num_cases = st.number_input("设定评测样本数量", min_value=1, max_value=20, value=5)
        with col_action:
            st.write("")
            st.write("")
            generate_btn = st.button("生成评测数据集", use_container_width=True)
            
        if generate_btn:
            with st.spinner("正在调用 qwen-plus 构建标准参考题库..."):
                test_cases = generate_evaluation_dataset(st.session_state['chunks'], num_cases)
                st.session_state['test_cases'] = test_cases
                st.success("测试用例构建完成。")
                
        if 'test_cases' in st.session_state and st.session_state['test_cases']:
            st.markdown("#### 当前评测数据集")
            st.dataframe(st.session_state['test_cases'], use_container_width=True)
            
            if st.button("执行准确率自动化测试", type="primary", use_container_width=True):
                st.markdown("---")
                total_score = 0
                progress_bar = st.progress(0)
                
                for i, case in enumerate(st.session_state['test_cases']):
                    q = case.get('query')
                    gt = case.get('ground_truth')
                    
                    rag_ans = rag_pipeline(q)
                    eval_result = evaluate_with_qwen_max(q, rag_ans, gt)
                    
                    score = eval_result.get("score", 0)
                    reason = eval_result.get("reason", "未提供理由")
                    total_score += score
                    
                    # 动态展示每一题的评测明细
                    with st.expander(f"测试集 {i+1}: {q}", expanded=False):
                        st.markdown(f"**基准答案 (Ground Truth):** {gt}")
                        st.markdown(f"**模型输出 (Generated):** {rag_ans}")
                        st.markdown(f"**系统评分:** {score} / 10")
                        st.markdown(f"**裁决依据:** {reason}")
                    
                    progress_bar.progress((i + 1) / len(st.session_state['test_cases']))
                    
                # 计算并展示最终百分比
                avg_score = total_score / len(st.session_state['test_cases'])
                percentage = avg_score * 10
                
                st.markdown("### 评测结果概览")
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric(label="测试用例总数", value=f"{len(st.session_state['test_cases'])} 题")
                with col_metric2:
                    st.metric(label="系统综合准确率", value=f"{percentage:.1f}%")
else:
    st.info("请先在左侧区域上传并解析文档，系统将自动解锁问答与测评功能。")