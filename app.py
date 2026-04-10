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

# 1. 页面与全局配置
st.set_page_config(page_title="智能文档解析与评测系统", layout="wide")

# 注入自定义 CSS 以缩减页面和侧边栏顶部空白
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem; 
            padding-bottom: 2rem;
        }
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

# 阿里云百炼常用模型清单 (移除了报错的 thinking 模型)
AVAILABLE_MODELS = [
    "qwen-max",             # 默认裁判 (index 0)
    "qwen3.5-plus",         # 默认出题 (index 1)
    "qwen-plus",
    "qwen3.5-flash",        # 默认答题 (index 3)
    "qwen-turbo",
    "qwen-long",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct"
]
# 2. 模型与知识库初始化
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
    rerank_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=device)
    db_client = chromadb.EphemeralClient()
    collection = db_client.create_collection(name="rag_docs_standard")
    return embed_model, rerank_model, collection

embed_model, rerank_model, collection = load_models()

# 3. 多格式文档处理模块
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
    chunk_size = 120
    overlap_size = 30

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

# 4. RAG 与 动态模型调度机制
def rag_pipeline(query: str, history: List[Dict] = None, answer_model: str = "qwen-turbo") -> str:
    """模块一：答题 (返回回答和上下文)"""
    search_query = query
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
        return "知识库为空或未检索到相关内容。", ""
        
    retrieved_docs = results["documents"][0]
    pairs = [[search_query, doc] for doc in retrieved_docs]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    final_context = [item[0] for item in ranked[:3]]
    
    system_prompt = """你是一个智能文档助手。请遵循以下规则回答：
    1. 【核心规则】：如果用户的问题是关于文档内容的，请务必严格且仅根据提供的【参考内容】进行回答。
    2. 【历史对话】：如果用户询问的是你们之间的对话历史（例如“我上一个问题是什么”、“根据上文…”），请直接基于对话历史回答，此时无需受参考内容的限制。
    3. 如果既无法在参考内容中找到答案，也无法在历史记录中找到答案，请诚实告知。"""
    context_text = chr(10).join(final_context)
    user_prompt = f"参考内容：\n{context_text}\n\n当前问题：{query}"
    
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-4:]) 
    messages.append({"role": "user", "content": user_prompt})

    response = client_ai.chat.completions.create(
        model=answer_model, 
        messages=messages,
    )
    # 返回答案和使用的上下文，供裁判打分使用
    return response.choices[0].message.content, context_text

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

def evaluate_answer(query: str, generated_answer: str, ground_truth: str, context: str, judge_model: str = "qwen-max") -> Dict:
    """模块三：多维度裁判 (严格遵循 RAGAS 标准 + 增强防崩溃 JSON 解析)"""
    prompt = f"""
    你是一个专业的 RAG (检索增强生成) 系统评测专家。请严格按照业界权威的 RAGAS 评测标准，对大模型的回答进行多维度打分。
    注意：请严格使用 **0-100 的整数**进行百分制打分。你必须展现出细腻的评分颗粒度，不要只打 0 或 100 分。

    【评测数据】
    - 用户问题 (Query): {query}
    - 检索到的上下文 (Context): {context}
    - 标准答案 (Ground Truth): {ground_truth}
    - 模型生成的回答 (Answer): {generated_answer}

    【评分梯度参考 (重要)】
    - 90-100分：完美，精确，毫无瑕疵。
    - 70-89分：基本正确或有用，但存在轻微瑕疵、轻微啰嗦或遗漏了次要细节。
    - 40-69分：部分正确，但遗漏了关键信息，或者包含了一半的无关/冗余信息。
    - 1-39分：质量很差，存在严重错误、严重幻觉或严重答非所问，但勉强沾边。
    - 0分：完全错误，彻底脱离上下文，或完全没有回答问题。

    【RAGAS 评测四大核心维度】
    1. 忠实性 (faithfulness): 模型的 Answer 是否基于 Context？(满分100：严格基于上下文。若引入外部正确但不包含在 Context 中的信息，扣10-20分；若产生捏造幻觉，扣50-100分)
    2. 回答相关性 (answer_relevancy): 模型的 Answer 是否直接回答了 Query？(满分100：直接切题。若开头有废话但后续答对，扣10-20分；若兜圈子且核心不明确，扣40-60分；答非所问0分)
    3. 上下文召回率 (context_recall): Context 是否包含了 Ground Truth 的所有事实点？(根据包含的事实点比例打分，包含 80% 的事实点就打 80 分)
    4. 上下文精确率 (context_precision): 检索到的 Context 中是否有用信息占主导？【终极豁免规则】：由于 RAG 系统文本切分（Chunking）的物理限制，召回的文本块中附带同一段落内的“相邻主题”（属于正常的工程现象。只要该文本块精准命中了核心答案，就【绝对不要】因为包含相邻的无关主题而严苛扣分。(满分100：只要包含核心事实且非大面积乱码/跨文章串台，应直接打 90-100 分；若通篇废话只有一两个字沾边，扣 50-80 分)。
    必须严格输出 JSON 格式，不要包含其他解释文字。格式要求如下：
    {{
        "faithfulness": {{"score": 85, "reason": "简短理由..."}},
        "answer_relevancy": {{"score": 92, "reason": "简短理由..."}},
        "context_recall": {{"score": 60, "reason": "简短理由..."}},
        "context_precision": {{"score": 35, "reason": "简短理由..."}}
    }}
    """
    try:
        response = client_ai.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # 获取大模型的原始回答
        content = response.choices[0].message.content.strip()
        
        # 【增强解析核心】：使用正则表达式，暴力提取 { 到 } 之间的所有内容
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
            
        return json.loads(content)
        
    except Exception as e:
        # 如果还是失败，把具体错误打印出来，不再是笼统的解析失败
        err_dict = {"score": 0, "reason": f"模型未按规范输出或解析异常: {str(e)}"}
        return {
            "faithfulness": err_dict, 
            "answer_relevancy": err_dict, 
            "context_recall": err_dict, 
            "context_precision": err_dict
        }

# 5. UI 界面层
st.title("基于通义千问的 RAG 检索增强生成系统")
st.markdown("---")

# 侧边栏：模型配置与文件解析
with st.sidebar:
    st.header("系统模型配置")
    st.markdown("请选择相应模型：")
    
    # 默认选项已对齐：答题 qwen3.5-flash, 出题 qwen3.5-plus, 裁判 qwen-max
    selected_rag_model = st.selectbox("答题模型 (被测对象)", AVAILABLE_MODELS, index=3)
    selected_gen_model = st.selectbox("出题模型 (标准生成)", AVAILABLE_MODELS, index=1)
    selected_judge_model = st.selectbox("裁判模型 (智能评分)", AVAILABLE_MODELS, index=0)
    
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
    tab1, tab2 = st.tabs(["交互式文档问答", "系统多维准确度测评"])
    
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
                    # 解包两个返回值，在聊天界面我们只展示 answer，忽略 context
                    response_ans, _ = rag_pipeline(prompt, history=history_for_rag, answer_model=selected_rag_model)
                    st.markdown(response_ans)
            st.session_state.messages.append({"role": "assistant", "content": response_ans})

    # ---------------- 第二部分：准确度测评 ----------------
    with tab2:
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
                test_cases = generate_evaluation_dataset(st.session_state['chunks'], num_cases, gen_model=selected_gen_model)
                st.session_state['test_cases'] = test_cases
                st.success("测试用例构建完成。")
                
        if 'test_cases' in st.session_state and st.session_state['test_cases']:
            st.markdown("#### 当前评测数据集")
            st.dataframe(st.session_state['test_cases'], use_container_width=True)
            
            if st.button("执行 RAGAS 标准评测", type="primary", use_container_width=True):
                st.markdown("---")
                # 初始化 RAGAS 4 个维度的总分
                scores_total = {"fa": 0, "ar": 0, "cr": 0, "cp": 0}
                progress_bar = st.progress(0)
                
                for i, case in enumerate(st.session_state['test_cases']):
                    q = case.get('query')
                    gt = case.get('ground_truth')
                    
                    # 获取答案和上下文
                    rag_ans, retrieved_ctx = rag_pipeline(q, answer_model=selected_rag_model)
                    # 将上下文传给裁判
                    eval_result = evaluate_answer(q, rag_ans, gt, retrieved_ctx, judge_model=selected_judge_model)
                    
                    # 解析 RAGAS 4 个维度的分数和理由 (带安全回退)
                    default_eval = {"score": 0, "reason": "解析失败"}
                    fa = eval_result.get("faithfulness", default_eval)
                    ar = eval_result.get("answer_relevancy", default_eval)
                    cr = eval_result.get("context_recall", default_eval)
                    cp = eval_result.get("context_precision", default_eval)
                    
                    scores_total["fa"] += fa.get("score", 0)
                    scores_total["ar"] += ar.get("score", 0)
                    scores_total["cr"] += cr.get("score", 0)
                    scores_total["cp"] += cp.get("score", 0)
                    
                    with st.expander(f"测试集 {i+1}: {q}", expanded=False):
                        st.markdown(f"**【基准答案】** {gt}")
                        st.markdown(f"**【模型输出】 ({selected_rag_model})** {rag_ans}")
                        st.markdown("---")
                        
                        # 展示 RAGAS 四大指标
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("忠实性", f"{fa.get('score', 0)}%")
                        sc2.metric("回答相关性", f"{ar.get('score', 0)}%")
                        sc3.metric("上下文召回率", f"{cr.get('score', 0)}%")
                        sc4.metric("上下文精确率", f"{cp.get('score', 0)}%")
                        
                        st.markdown(f"> **裁判评语 ({selected_judge_model}):**\n"
                                    f"> - 忠实性 (Faithfulness): {fa.get('reason')}\n"
                                    f"> - 回答相关性 (Answer Relevancy): {ar.get('reason')}\n"
                                    f"> - 上下文召回率 (Context Recall): {cr.get('reason')}\n"
                                    f"> - 上下文精确率 (Context Precision): {cp.get('reason')}")
                    
                    progress_bar.progress((i + 1) / len(st.session_state['test_cases']))
                    
                # 计算平均分
                total_cases = len(st.session_state['test_cases'])
                avg_fa = scores_total["fa"] / total_cases
                avg_ar = scores_total["ar"] / total_cases
                avg_cr = scores_total["cr"] / total_cases
                avg_cp = scores_total["cp"] / total_cases
                
                st.markdown("### RAGAS 核心指标大盘")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("忠实性\n(Faithfulness)", f"{avg_fa:.1f}%", help="模型是否产生幻觉？得分越高说明回答越贴近检索到的原文。")
                col_m2.metric("回答相关性\n(Answer Relevancy)", f"{avg_ar:.1f}%", help="回答是否切题？得分越高说明废话越少，直击痛点。")
                col_m3.metric("上下文召回率\n(Context Recall)", f"{avg_cr:.1f}%", help="检索系统是否漏掉了关键信息？得分越高说明找得越全。")
                col_m4.metric("上下文精确率\n(Context Precision)", f"{avg_cp:.1f}%", help="检索到的内容是否全是干货？得分越高说明噪音/废话越少。")
else:
    st.info("请先在左侧区域上传并解析文档，系统将自动解锁问答与测评功能。")