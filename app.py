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
# 1. 页面与全局配置 (严格无 Emoji, 专业风格)
# ==========================================
st.set_page_config(page_title="智能文档解析与评测系统", layout="wide")

# 注入自定义 CSS 以缩减页面和侧边栏顶部空白
st.markdown("""
    <style>
        /* 1. 调整右侧主界面顶部空白 */
        .block-container {
            padding-top: 2rem; 
            padding-bottom: 2rem;
        }
        
        /* 2. 调整左侧边栏顶部空白 */
        [data-testid="stSidebarUserContent"] {
            padding-top: 2rem; 
        }
    </style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    st.error("系统配置错误：未在环境变量中找到 DASHSCOPE_API_KEY")
    st.stop()

client_ai = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 阿里云百炼常用模型清单
AVAILABLE_MODELS = [
    "qwen-turbo",             # 速度最快，成本最低，适合基础对话和意图识别
    "qwen-plus",              # 能力均衡，适合长文档处理和复杂任务
    "qwen-max",               # 智力天花板，适合作为严苛的裁判或解决复杂推理
    "qwen-long",              # 专门针对超长上下文优化的模型
    "qwen2.5-72b-instruct",   # 开源界顶级模型，参数量极大，性能逼近 max
    "qwen2.5-32b-instruct",   # 开源中坚力量，性能与速度的良好平衡
    "qwen2.5-14b-instruct",   # 小参数量开源模型，响应极快
]

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
        
        # 指定 Tesseract 的执行文件路径
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        try:
            img = Image.open(file_path)
            full_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        except Exception as e:
            st.error("图片解析失败。请确保已安装 Tesseract-OCR 并配置了中文语言包。")
            raise e
    else:
        raise ValueError("不支持的文件格式")
    return full_text

def process_uploaded_file(uploaded_file) -> List[str]:
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
    chunk_size = 250
    overlap_size = 50

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
# 4. RAG 与 动态模型调度机制
# ==========================================
def rag_pipeline(query: str, history: List[Dict] = None, answer_model: str = "qwen-turbo") -> str:
    """模块一：答题 (将被用户配置的模型驱动)"""
    search_query = query
    # 意图重写环节固定使用 turbo 以保障速度和节约成本
    if history and len(history) > 0:
        recent_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
        rewrite_prompt = f"""
        你是一个意图识别专家。请结合以下历史对话，将用户的最新简短问题重写为一个完整、独立的问题。
        如果用户的原问题已经很完整，或者与历史对话无关，请直接输出原问题。不要输出任何额外解释。
        【历史对话】:\n{recent_history}\n【最新用户问题】: {query}\n重写后的独立问题：
        """
        try:
            rewrite_response = client_ai.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": rewrite_prompt}]
            )
            search_query = rewrite_response.choices[0].message.content.strip()
        except Exception:
            pass

    query_emb = embed_model.encode([search_query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=10)
    
    if not results["documents"] or not results["documents"][0]:
        return "知识库为空或未检索到相关内容。"
        
    retrieved_docs = results["documents"][0]
    pairs = [[search_query, doc] for doc in retrieved_docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    final_context = [item[0] for item in ranked[:3]]
    
    system_prompt = "你是一个专业的文档助手。请仅根据提供的参考内容回答用户问题。如果参考内容不足以回答，请诚实告知。"
    context_text = chr(10).join(final_context)
    user_prompt = f"参考内容：\n{context_text}\n\n当前问题：{query}"
    
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-4:]) 
    messages.append({"role": "user", "content": user_prompt})

    # 使用用户指定的模型进行答题
    response = client_ai.chat.completions.create(
        model=answer_model, 
        messages=messages,
    )
    return response.choices[0].message.content

def generate_evaluation_dataset(chunks: List[str], num_cases: int = 5, gen_model: str = "qwen-plus") -> List[Dict]:
    """模块二：出题 (将被用户配置的模型驱动)"""
    sample_text = "\n".join(random.sample(chunks, min(10, len(chunks))))
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
        response = client_ai.chat.completions.create(
            model=gen_model,
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

def evaluate_answer(query: str, generated_answer: str, ground_truth: str, judge_model: str = "qwen-max") -> Dict:
    """模块三：裁判 (将被用户配置的模型驱动)"""
    prompt = f"""
    你是一个客观且具备专业知识的阅卷专家。请根据【用户问题】和【标准答案】，对【模型生成的答案】进行评分(0-10分)。

    评分核心原则（请严格遵守）：
    1. 【核心覆盖即满分】：只要模型答案完整包含了标准答案的核心事实，就应当给予 10 分。语言表述无需完全一致。
    2. 【鼓励合理补充】：如果模型答案在覆盖标准答案的基础上，提供了额外的、事实正确的补充信息（例如相关的测试工具、背景知识等），**绝对不能扣分**，应视为逻辑详实的好答案。
    3. 【扣分红线】：仅当模型答案发生以下情况时才扣分：
       - 遗漏了标准答案中的关键性事实（扣 2-5 分）。
       - 包含了明显的常识性错误或事实性错误（扣 5-10 分）。
       - 答非所问，与提问无关（打 0 分）。

    【用户问题】: {query}
    【标准答案】: {ground_truth}
    【模型答案】: {generated_answer}
    
    必须严格输出 JSON 格式，包含 "score" (整数) 和 "reason" (简短解释)。
    """
    try:
        response = client_ai.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": 0, "reason": f"评估发生异常: {str(e)}"}

# ==========================================
# 5. UI 界面层
# ==========================================
st.title("基于通义千问的 RAG 检索增强生成系统")
st.markdown("---")

# 侧边栏：模型配置与文件解析
with st.sidebar:
    st.header("系统模型配置")
    st.markdown("请选择相应模型：")
    
    # 用户自由选择模型，提供默认最佳实践的 index
    selected_rag_model = st.selectbox("答题模型 (被测对象)", AVAILABLE_MODELS, index=0)
    selected_gen_model = st.selectbox("出题模型 (标准生成)", AVAILABLE_MODELS, index=1)
    selected_judge_model = st.selectbox("裁判模型 (智能评分)", AVAILABLE_MODELS, index=2)
    
    st.markdown("---")
    st.header("文件解析")
    st.markdown("支持格式：`.pdf`, `.docx`, `.jpg`, `.png`")
    
    uploaded_file = st.file_uploader("拖拽文件至此区域自动解析", type=["pdf", "docx", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_hash = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_hash") != file_hash:
            with st.spinner("系统正在读取并向量化文档..."):
                chunks = process_uploaded_file(uploaded_file)
                if chunks:
                    build_index(chunks)
                    st.session_state['chunks'] = chunks
                    st.session_state['current_file_hash'] = file_hash
                    st.session_state['messages'] = []
                    st.session_state.pop('test_cases', None)
                else:
                    st.warning("未能在文档中提取到有效文本。")
                    
        if 'chunks' in st.session_state:
            st.success("知识库已就绪")
            st.info(f"当前文档: {uploaded_file.name}\n片段数量: {len(st.session_state['chunks'])}")
    else:
        st.session_state.pop("current_file_hash", None)
        st.session_state.pop("chunks", None)

# 主界面：功能分签
if 'chunks' in st.session_state and collection.count() > 0:
    tab1, tab2 = st.tabs(["交互式文档问答", "系统准确度测评"])
    
    # ---------------- 第一部分：交互问答 ----------------
    with tab1:
        st.subheader("文档内容问答")
        st.caption(f"当前使用的生成模型：{selected_rag_model}")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("请输入您关于文档的问题"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("模型思考中..."):
                    history_for_rag = st.session_state.messages[:-1] 
                    # 将用户选择的模型传入
                    response = rag_pipeline(prompt, history=history_for_rag, answer_model=selected_rag_model)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # ---------------- 第二部分：准确度测评 ----------------
    with tab2:
        st.subheader("RAG 回答准确度自动化测评")
        
        col_input, col_action = st.columns([1, 1])
        with col_input:
            num_cases = st.number_input("设定评测样本数量", min_value=1, max_value=50, value=5)
        with col_action:
            st.write("")
            st.write("")
            generate_btn = st.button("生成评测数据集", use_container_width=True)
            
        if generate_btn:
            with st.spinner(f"正在调用 {selected_gen_model} 构建标准参考题库..."):
                # 将用户选择的出题模型传入
                test_cases = generate_evaluation_dataset(st.session_state['chunks'], num_cases, gen_model=selected_gen_model)
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
                    
                    # 使用答题模型生成答案，使用裁判模型打分
                    rag_ans = rag_pipeline(q, answer_model=selected_rag_model)
                    eval_result = evaluate_answer(q, rag_ans, gt, judge_model=selected_judge_model)
                    
                    score = eval_result.get("score", 0)
                    reason = eval_result.get("reason", "未提供理由")
                    total_score += score
                    
                    with st.expander(f"测试集 {i+1}: {q}", expanded=False):
                        st.markdown(f"**基准答案:** {gt}")
                        st.markdown(f"**模型输出 ({selected_rag_model}):** {rag_ans}")
                        st.markdown(f"**系统评分 ({selected_judge_model}):** {score} / 10")
                        st.markdown(f"**裁决依据:** {reason}")
                    
                    progress_bar.progress((i + 1) / len(st.session_state['test_cases']))
                    
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