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
import base64
import io
from PIL import Image

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
        import io
        import base64
        from PIL import Image
        import docx

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # 1. 提取当前页面的纯文本
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
                
                # 2. 提取并转化 Markdown 表格
                tables = page.extract_tables()
                for table in tables:
                    if not table: continue
                    md_table = "\n[文档结构化表格数据]\n"
                    for i, row in enumerate(table):
                        clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                        md_table += "| " + " | ".join(clean_row) + " |\n"
                        if i == 0:
                            md_table += "|" + "|".join(["---"] * len(clean_row)) + "|\n"
                    full_text += md_table + "\n\n"

                # 核心升级：自动抠取 PDF 插图并调用 VLM 解析
        
                # 遍历当前页面的所有图片对象
                for img_idx, img_obj in enumerate(page.images):
                    # 获取图片在 PDF 页面上的物理坐标
                    x0, y0, x1, y1 = img_obj['x0'], img_obj['top'], img_obj['x1'], img_obj['bottom']
                    
                    # 【防噪过滤】：过滤掉小于 100x100 像素的图片（通常是小 Logo、页脚装饰线条等废图）
                    if (x1 - x0) < 100 or (y1 - y0) < 100:
                        continue
                        
                    try:
                        # 1. 物理裁剪：从 PDF 页面上把这张图片“截图”下来
                        cropped_page = page.crop((x0, y0, x1, y1))
                        # 渲染为 PIL Image 对象 (分辨率设为 150 足够 VLM 看清了)
                        pil_img = cropped_page.to_image(resolution=150).original
                        
                        # 2. 图片压缩与 Base64 编码 (复用我们之前写过的防爆破逻辑)
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        buffered = io.BytesIO()
                        pil_img.save(buffered, format="JPEG", quality=85)
                        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        # 3. 编写多模态 RAG 专属 Prompt
                        vlm_prompt = """
                        你是一个专业的企业文档结构化解析引擎。这是一张从PDF中提取的插图。
                        请详细、客观地描述该图片的核心内容，以便用于后续的文字检索系统。
                        【规则】：
                        1. 如果是“流程图/架构图”：请用文本逻辑描述出先后步骤或层级关系。
                        2. 如果是“数据图表(柱状/折线/饼图)”：请提取出核心的数值、趋势或结论。
                        3. 如果是“产品/设备示意图”：请说明设备的名称及图中标注的各部件信息。
                        4. 忽略毫无意义的背景装饰。
                        """
                        
                        # 4. 调用通义千问视觉大模型
                        response = client_ai.chat.completions.create(
                            model="qwen-vl-max", 
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": vlm_prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                    ]
                                }
                            ]
                        )
                        vlm_desc = response.choices[0].message.content.strip()
                        
                        # 5. 【绝杀点】：将图片的文字描述作为独立段落，缝合进原文档文本流中！
                        full_text += f"\n[第{page_num+1}页-重要插图解析]\n{vlm_desc}\n\n"
                        
                    except Exception as e:
                        # 遇到个别破损的图片直接跳过，保证系统不崩溃
                        print(f"提取第 {page_num+1} 页图片失败，跳过: {e}")
                        continue

    elif file_extension == 'docx':
        doc = docx.Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    elif file_extension in ['jpg', 'jpeg', 'png']:
        try:
            # 1. 使用 PIL 打开图片
            img = Image.open(file_path)
            
            # 2. 转换为 RGB 模式（丢弃可能存在的透明通道，减小体积）
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 3. 智能等比缩放：限制图片最大边长为 2048 像素
            max_size = 2048
            if max(img.size) > max_size:
                # 使用 LANCZOS 算法进行高质量缩放
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            # 4. 将压缩后的图片存入内存缓冲区
            buffered = io.BytesIO()
            # 统一保存为 JPEG 格式，并设置 quality=85 进行适度体积压缩
            img.save(buffered, format="JPEG", quality=85) 
            
            # 5. 转换为 Base64 字符串
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            mime_type = "image/jpeg" # 既然强制存为 JPEG，MIME 类型也固定为 jpeg
            
            # 6. 编写极其严格的“排版还原 Prompt”
            ocr_prompt = """
            你是一个企业级的文档排版与内容还原专家。请精准提取这张图片中的所有文字内容。
            【严格要求】：
            1. 必须100%忠实于原图，不要遗漏、不要总结、不要瞎编（产生幻觉）。
            2. 【重中之重】：精准识别文章的层级结构。遇到各级标题（如“第一章”、“第一节”、“1.”、“1.1”、“步骤一”等），必须【独立成行】。
            3. 遇到正文的自然段落，请使用换行符分隔，不要把不同段落的内容揉成一段。
            """
            
            # 7. 调用通义千问视觉大模型
            response = client_ai.chat.completions.create(
                model="qwen-vl-max", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ocr_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                        ]
                    }
                ]
            )
            full_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"视觉大模型解析图片失败，请检查网络或模型权限: {str(e)}")
            raise e
    else:
        raise ValueError("不支持的文件格式")
        
    return full_text
    
def process_uploaded_file(uploaded_file) -> List[Dict]:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        full_text = extract_text_from_file(tmp_path, file_extension)
    except Exception as e:
        st.error(f"提取文本错误：{str(e)}")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 1. 清理异常空白，但必须保留换行符（因为换行是判断结构的依据）
    full_text = re.sub(r' +', ' ', full_text)
    raw_lines = [line.strip() for line in full_text.split('\n') if line.strip()]

    # 2.企业手册结构解析引擎 (Heading-Aware)
    # 针对中国企业常用公文排版设计的正则匹配
    h1_pattern = re.compile(r'^(第[一二三四五六七八九十百]+[章节部分篇]|[\d]{1,2}、|[A-Z]\.)') # 匹配: 第一章, 1、, A.
    h2_pattern = re.compile(r'^(\d{1,2}\.\d{1,2}|[\(（][一二三四五六七八九十][\)）])') # 匹配: 1.1, (一)
    
    current_h1 = "通用规范" # 默认全局上下文
    current_h2 = ""
    
    parents = []
    current_parent_lines = []

    # 逐行扫描，提取层级结构
    for line in raw_lines:
        if h1_pattern.match(line):
            # 遇到一级标题，先把之前攒的内容封箱
            if current_parent_lines:
                context_tag = f"{current_h1}{' > ' + current_h2 if current_h2 else ''}"
                parents.append({"context": context_tag, "text": "\n".join(current_parent_lines)})
            current_h1 = line # 更新一级标题
            current_h2 = ""   # 清空二级标题
            current_parent_lines = [line]
        elif h2_pattern.match(line):
            # 遇到二级标题，同样封箱
            if current_parent_lines:
                context_tag = f"{current_h1}{' > ' + current_h2 if current_h2 else ''}"
                parents.append({"context": context_tag, "text": "\n".join(current_parent_lines)})
            current_h2 = line # 更新二级标题
            current_parent_lines = [line]
        else:
            # 普通正文，继续攒入当前的层级箱子
            current_parent_lines.append(line)
            
    # 把最后剩余的内容收尾
    if current_parent_lines:
        context_tag = f"{current_h1}{' > ' + current_h2 if current_h2 else ''}"
        parents.append({"context": context_tag, "text": "\n".join(current_parent_lines)})


    # 终极绝杀：上下文注入法 (Contextual Injection)

    hierarchical_chunks = []
    window_size = 2  # 依然采用滑动窗口合并句子

    for p in parents:
        context_path = p["context"]
        parent_text = p["text"]
        
        # 【完美修复 1】：分离“标题行”和“正文内容”
        # 如果大块的第一行就是标题，我们在切子块时把它扔掉（因为 context_path 里已经有了）
        lines = parent_text.split('\n')
        if len(lines) > 1 and (lines[0].replace(' ', '') in context_path.replace(' ', '')):
            body_text = "\n".join(lines[1:]) # 只取真正的正文
        else:
            body_text = parent_text
            
        # 【完美修复 2】：中文排版清理，直接抹除物理换行，不加空格！
        clean_body = body_text.replace('\n', '')
        
        # 按句号进行原子切分
        sentences = re.split(r'(?<=[。！？；])', clean_body) 
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences: continue

        # 滑动窗口合并
        if len(sentences) < window_size:
            enriched_child = f"[{context_path}] {sentences[0]}"
            hierarchical_chunks.append({"child": enriched_child, "parent": parent_text})
        else:
            for i in range(len(sentences) - window_size + 1):
                child_text = " ".join(sentences[i:i + window_size])
                if len(child_text) > 10:
                    enriched_child = f"[{context_path}] {child_text}"
                    hierarchical_chunks.append({"child": enriched_child, "parent": parent_text})

    # 打印效果展示
    print("\n" + "="*50)
    print(f"企业规范结构化切分完成！(生成 {len(hierarchical_chunks)} 个带上下文的检索块)")
    print("="*50)
    for i, item in enumerate(hierarchical_chunks):
        print(f"\n【精准子块 Child {i+1}】(向量化目标：包含自动推断的结构上下文)")
        print(item['child'])
        print(f" └─▶ 【关联父块 Parent】(喂给大模型阅读的完整章节)")
        print(item['parent'][:80].replace('\n', ' ') + "......" if len(item['parent']) > 80 else item['parent'].replace('\n', ' '))
        print("-" * 50)

    return hierarchical_chunks
def build_index(chunks_data: List[Dict]):
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        
    # 提取子块文本进行向量化（用于被搜）
    child_texts = [item["child"] for item in chunks_data]
    # 将父块文本悄悄存入 ChromaDB 的 metadata 里（用于被大模型阅读）
    metadatas = [{"parent": item["parent"]} for item in chunks_data]
    ids = [f"id_{i}" for i in range(len(chunks_data))]
    
    embeddings = embed_model.encode(child_texts).tolist()
    # 存入数据库时，同时传入 metadatas
    collection.add(documents=child_texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

# 4. RAG 与 动态模型调度机制
def rag_pipeline(query: str, history: List[Dict] = None, answer_model: str = "qwen-turbo") -> str:
    """模块一：答题 (小块检索，大块喂入)"""
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

    # 1. 粗排：向量检索 Top-10 的“子块”
    query_emb = embed_model.encode([search_query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=10)
    
    # 注意这里：我们现在要取的是 metadatas，而不是 documents
    if not results["metadatas"] or not results["metadatas"][0]:
        return "知识库为空或未检索到相关内容。", ""
        
    retrieved_metadatas = results["metadatas"][0]
    
    # 2. 魔法时刻：从小块中提取关联的“大父块”，并去重！
    unique_parents = []
    seen = set()
    for meta in retrieved_metadatas:
        parent_text = meta["parent"]
        if parent_text not in seen:
            seen.add(parent_text)
            unique_parents.append(parent_text)
            
    # 3. 精排：让重排序模型（CrossEncoder）去阅读这些去重后的大父块
    pairs = [[search_query, p] for p in unique_parents]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(unique_parents, scores), key=lambda x: x[1], reverse=True)
    
    # 4. 选取最相关的前 2 个大父块喂给千问大模型
    final_context = []
    for doc, score in ranked[:2]:
        if score > -5.0: # 精排模型得分阈值
            final_context.append(doc)
            
    if not final_context and ranked:
        final_context.append(ranked[0][0])
    
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
    return response.choices[0].message.content, context_text

def rag_pipeline_stream(query: str, history: List[Dict] = None, answer_model: str = "qwen-turbo"):
    """支持 Stream 流式输出的 RAG 核心逻辑"""
    search_query = query
    
    # ================= 1. 补充缺失的意图识别与检索逻辑 =================
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

    # 粗排：向量检索 Top-10
    query_emb = embed_model.encode([search_query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=10)
    
    context_text = ""
    if results["metadatas"] and results["metadatas"][0]:
        retrieved_metadatas = results["metadatas"][0]
        
        # 父块去重
        unique_parents = []
        seen = set()
        for meta in retrieved_metadatas:
            parent_text = meta["parent"]
            if parent_text not in seen:
                seen.add(parent_text)
                unique_parents.append(parent_text)
                
        # 精排
        pairs = [[search_query, p] for p in unique_parents]
        scores = rerank_model.predict(pairs)
        ranked = sorted(zip(unique_parents, scores), key=lambda x: x[1], reverse=True)
        
        # 选取 Top 2
        final_context = []
        for doc, score in ranked[:2]:
            if score > -5.0:
                final_context.append(doc)
                
        if not final_context and ranked:
            final_context.append(ranked[0][0])
            
        context_text = chr(10).join(final_context)

    # ================= 2. 补充缺失的 Prompt 定义 =================
    system_prompt = """你是一个智能文档助手。请遵循以下规则回答：
    1. 【核心规则】：如果用户的问题是关于文档内容的，请务必严格且仅根据提供的【参考内容】进行回答。
    2. 【历史对话】：如果用户询问的是你们之间的对话历史（例如“我上一个问题是什么”、“根据上文…”），请直接基于对话历史回答，此时无需受参考内容的限制。
    3. 如果既无法在参考内容中找到答案，也无法在历史记录中找到答案，请诚实告知。"""
    
    user_prompt = f"参考内容：\n{context_text}\n\n当前问题：{query}"

    # ================= 3. 拼装消息并请求流式接口 =================
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-4:]) 
    messages.append({"role": "user", "content": user_prompt})

    # 开启流式请求
    response = client_ai.chat.completions.create(
        model=answer_model, 
        messages=messages,
        stream=True  # 关键点
    )
    
    # 构建一个 Python 生成器 (Generator)
    def stream_generator():
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    # 返回生成器和参考上下文
    return stream_generator(), context_text
    
def generate_evaluation_dataset(chunks_data: List[Dict], num_cases: int = 5, gen_model: str = "qwen-plus") -> List[Dict]:
    """模块二：出题 (基于大父块出题)"""
    # 从字典列表中提取所有去重后的父块
    unique_parents = list(set([item["parent"] for item in chunks_data]))
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
    
    # 开启多文件支持：accept_multiple_files=True
    uploaded_files = st.file_uploader("拖拽文件至此区域自动解析 (支持多文件)", type=["pdf", "docx", "jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files: # 如果上传了文件（列表不为空）
        # 将所有文件的信息拼接成一个唯一的 Hash，用于判断用户是否增加/删除了文件
        file_hash = "|".join([f"{f.name}_{f.size}" for f in uploaded_files])
        
        # 只要文件列表发生变动（新增或取消了某个文件），就重新构建知识库
        if st.session_state.get("current_file_hash") != file_hash:
            with st.spinner("系统正在读取并合并向量化文档..."):
                all_chunks = []
                for file in uploaded_files:
                    # 循环解析每一个上传的文件
                    chunks = process_uploaded_file(file)
                    if chunks:
                        all_chunks.extend(chunks) # 合并所有片段
                
                if all_chunks:
                    build_index(all_chunks) # 一次性将所有文件的片段写入数据库
                    st.session_state['chunks'] = all_chunks
                    st.session_state['current_file_hash'] = file_hash
                    st.session_state['messages'] = []
                    st.session_state.pop('test_cases', None)
                else:
                    st.warning("未能在文档中提取到有效文本。")
                    
        if 'chunks' in st.session_state:
            st.success("多文档知识库已就绪")
            st.info(f"当前已加载文档数: {len(uploaded_files)} 个\n融合片段总数: {len(st.session_state['chunks'])}")
    else:
        # 【关键修复】当用户点击 X 清空所有文件时，不仅要清理页面 Session，还要彻底清空底层向量数据库
        st.session_state.pop("current_file_hash", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("test_cases", None)
        st.session_state.pop("messages", None)
        
        # 提取当前数据库中残留的旧数据并强制删除
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

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
            # 1. 显示用户输入
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. 生成流式回答
            with st.chat_message("assistant"):
                history_for_rag = st.session_state.messages[:-1] 
                
                with st.spinner("正在检索知识库并思考..."):
            # 1. 先获取生成器（会执行检索）
                    stream_gen, _ = rag_pipeline_stream(prompt, history=history_for_rag, answer_model=selected_rag_model)
            
            # 2. 流式输出（spinner 会一直显示到输出结束）
                    full_response = st.write_stream(stream_gen)

    # 3. 存入历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})

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