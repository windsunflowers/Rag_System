import os
import time
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import torch
import pdfplumber

# 1. 环境与模型初始化
load_dotenv()

# 获取阿里云的 API KEY
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未在 .env 中找到 DASHSCOPE_API_KEY")

# 初始化通义千问客户端
client_ai = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载本地模型 (运行设备: {device})...")

# 基础向量模型
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
# 重排序模型 (Reranker)
rerank_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=device)

# 初始化内存数据库
db_client = chromadb.EphemeralClient()
collection = db_client.create_collection(name="rag_docs")

# 2. 功能函数定义
def load_docs(file_path: str) -> List[str]:
    """读取并切分文档"""
    if not os.path.exists(file_path):
        return []
    text = ""

    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 5]
    return chunks

def build_index(chunks: List[str]):
    """向量化并存入数据库"""
    print(f"正在索引 {len(chunks)} 个文档块...")
    embeddings = embed_model.encode(chunks).tolist()
    ids = [f"id_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print("索引构建完成！")

def ask_qwen(query: str, context: str):
    """调用通义千问模型生成回答"""
    prompt = f"你是一个专业的文档助手。请仅根据参考内容回答问题。如果参考内容不足以回答，请诚实告知。\n\n参考内容：\n{context}\n\n问题：{query}"
    response = client_ai.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个严谨的 RAG 问答助手。"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def rag_pipeline(query: str):
    try:
        # 1. 检索
        t1 = time.time()
        query_emb = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=10)
        retrieved_docs = results["documents"][0]
        
        # 2. 重排序
        t2 = time.time()
        pairs = [[query, doc] for doc in retrieved_docs]
        scores = rerank_model.predict(pairs)
        ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        final_context = [item[0] for item in ranked[:3]]
        
        # 3. 生成答案
        t3 = time.time()
        print(f"检索+重排耗时: {t3-t1:.2f}s | 正在调用通义千问...")
        answer = ask_qwen(query, "\n\n".join(final_context))
        print(f"问答总耗时: {time.time()-t1:.2f}s")
        
        return answer

    except Exception as e:
        if "402" in str(e) or "Quota" in str(e):
            return "免费额度已用完或余额不足。"
        return f"发生错误: {str(e)}"

# 3. 运行入口
if __name__ == "__main__":
# 这里把文件名改成文件路径

    doc_path = "doc.pdf"
    chunks = load_docs(doc_path)
    
    if chunks:
        build_index(chunks)
        print("\n阿里版 RAG 系统已启动 (Qwen 驱动)")
        while True:
            q = input("用户提问: ").strip()
            if q.lower() == 'exit':
                print("退出程序")
                break
            if not q:
                continue
            
            print("-" * 30)
            print(f"回答：\n{rag_pipeline(q)}")
            print("-" * 30)
    else:
        print(f"警告：找不到 {doc_path} 或文档内容为空，请检查。")