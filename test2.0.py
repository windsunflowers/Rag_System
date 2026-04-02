import os
import time
import json
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import torch
import pdfplumber
import re

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

# 2. 核心 RAG 功能函数定义
def split_pdf_into_chunks(pdf_file: str, chunk_size=500, overlap_size=100) -> list[str]:
    """
    修复版的 PDF 语义分块：按真实的“句子”进行滑动窗口切分
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    except Exception as e:
        print(f"读取PDF失败：{e}")
        return []

    # 1. 基础清理
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    # 2. 正确的中文分句（不依赖标点后的空格）
    # 使用捕获组保留标点符号，或者使用前瞻断言
    sentences = re.split(r'(?<=[。！？；])', full_text)
    
    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)

        # 如果当前块 + 新句子 没有超长，正常加入
        if current_length + sentence_len <= chunk_size:
            current_sentences.append(sentence)
            current_length += sentence_len
        else:
            # 超长了，先把当前的块打包存起来
            if current_sentences:
                chunks.append("".join(current_sentences))
            
            # --- 核心修复：按句子执行 Overlap ---
            # 从刚才的块中，从后往前捞取句子，直到达到 overlap_size 限制
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_sentences):
                if overlap_length + len(s) > overlap_size:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)
            
            # 开启新的块：包含 Overlap 留下的句子 + 当前的新句子
            current_sentences = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_sentences)

    # 3. 处理最后剩下的一点尾巴
    if current_sentences:
        chunks.append("".join(current_sentences))

    # 去除太短的垃圾块（比如只解析出页码的块）
    chunks = [c for c in chunks if len(c) > 20]
    
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
        query_emb = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=10)
        retrieved_docs = results["documents"][0]
        
        # 2. 重排序
        pairs = [[query, doc] for doc in retrieved_docs]
        scores = rerank_model.predict(pairs)
        ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        final_context = [item[0] for item in ranked[:3]]
        
        # 3. 生成答案
        answer = ask_qwen(query, "\n\n".join(final_context))
        return answer

    except Exception as e:
        if "402" in str(e) or "Quota" in str(e):
            return "免费额度已用完或余额不足。"
        return f"发生错误: {str(e)}"

# 3. 评测模块 (LLM-as-a-Judge)
def evaluate_with_qwen(query: str, generated_answer: str, ground_truth: str) -> Dict:
    """
    使用 Qwen 模型作为裁判，评估 RAG 生成答案的准确率。
    返回包含打分(0-10)和评价理由的字典。
    """
    prompt = f"""
    你是一个严谨的阅卷专家。请根据提供的【用户问题】和【标准答案】，对【模型生成的答案】进行评分。
    
    评分标准：
    - 满分为 10 分。
    - 核心事实一致即可得高分，语言表述可以不同。
    - 若生成答案包含严重事实错误或完全偏离标准答案，给 0-3 分。
    - 若生成答案部分正确，但遗漏了关键信息，给 4-7 分。
    - 若生成答案准确、完整地覆盖了标准答案的核心内容，给 8-10 分。

    【用户问题】: {query}
    【标准答案】: {ground_truth}
    【模型生成的答案】: {generated_answer}

    请以 JSON 格式输出，必须包含 "score" (整数) 和 "reason" (简短解释) 两个字段。
    """
    
    try:
        # 建议评测时使用能力更强的模型，比如 qwen-max
        response = client_ai.chat.completions.create(
            model="qwen-max", 
            messages=[
                {"role": "system", "content": "你是一个客观公正的答案评估助手。只能输出严格的 JSON 格式数据。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"} # 强制输出 JSON 格式
        )
        result_str = response.choices[0].message.content
        return json.loads(result_str)
    except Exception as e:
        return {"score": 0, "reason": f"评估过程发生错误: {str(e)}"}

def run_accuracy_test(test_cases: List[Dict]):
    """批量运行测试并计算平均准确率"""
    print("\n" + "="*40)
    print("开始进行模型准确率自动测试...")
    print("="*40)
    
    total_score = 0
    results = []

    for i, test in enumerate(test_cases):
        query = test["query"]
        ground_truth = test["ground_truth"]
        
        print(f"\n测试用例 {i+1}: {query}")
        
        # 1. 获取 RAG 答案
        t1 = time.time()
        generated_answer = rag_pipeline(query)
        t2 = time.time()
        
        # 2. 进行评估
        eval_result = evaluate_with_qwen(query, generated_answer, ground_truth)
        score = eval_result.get("score", 0)
        reason = eval_result.get("reason", "无")
        
        total_score += score
        
        print(f"-> 耗时: {t2-t1:.2f}s")
        print(f"-> 模型回答: {generated_answer}")
        print(f"-> 裁判打分: {score}/10")
        print(f"-> 裁判评语: {reason}")
        
    avg_score = total_score / len(test_cases)
    print("\n" + "="*40)
    print(f"测试完成！总用例数: {len(test_cases)} | 准确率: {avg_score * 10:.2f}%")
    print("="*40)

# 4. 运行入口
if __name__ == "__main__":
    doc_path = "doc.pdf"
    chunks = split_pdf_into_chunks(doc_path)
    
    if chunks:
        build_index(chunks)
        print("\n系统已启动！")
        
        # 选择运行模式
        mode = input("请选择模式 (1: 聊天问答模式, 2: 准确率测试模式): ").strip()
        
        if mode == '2':
            # 这里需要你根据你的 doc.pdf 编写对应的测试集
            test_dataset = [
    {
        "query": "故事中提到的神秘水域叫什么名字？",
        "ground_truth": "这片神秘水域名为“琉璃海马域” [cite: 3]。"
    },
    {
        "query": "海马族群由谁来统领？",
        "ground_truth": "由年长且智慧的深海海马王统领 [cite: 3]。"
    },
    {
        "query": "琉璃海马域孕育了哪些珍稀的海马族群？",
        "ground_truth": "孕育了黄金海马、冰晶海马、烈焰海马等珍稀族群 [cite: 3]。"
    },
    {
        "query": "海马族群擅长什么技能？",
        "ground_truth": "擅长操控水流、净化海水 [cite: 3]。"
    },
    {
        "query": "与琉璃海马域隔海相望的森林叫什么？",
        "ground_truth": "是一片漂浮在海面之上的“云顶喵喵森林” [cite: 4]。"
    },
    {
        "query": "云顶喵喵森林是由什么构建的？",
        "ground_truth": "由千年灵木构建 [cite: 4]。"
    },
    {
        "query": "喵喵喵部族由谁执掌？",
        "ground_truth": "由九尾灵猫族长执掌 [cite: 4]。"
    },
    {
        "query": "喵喵喵部族擅长哪些能力？",
        "ground_truth": "擅长攀爬、隐匿、释放幻术，还能操控林间风元素 [cite: 4]。"
    },
    {
        "query": "为什么喵喵喵部族对深海族群抱有戒备之心？",
        "ground_truth": "因为它们领地意识极强，觉得海马族群的水流波动会扰乱森林的灵气运转 [cite: 4]。"
    },
    {
        "query": "引发双方矛盾的导火索是什么？",
        "ground_truth": "是深海海底的上古灵珠发生能量外泄 [cite: 5]。"
    },
    {
        "query": "灵珠能量外泄给喵喵森林造成了什么破坏？",
        "ground_truth": "导致云顶喵喵森林的边缘灵木折断、雾气紊乱，部分幼崽灵猫被水流波及受伤 [cite: 5]。"
    },
    {
        "query": "对于森林受损，喵喵喵部族是怎么认为的？",
        "ground_truth": "误以为是海马族群故意发动攻击，破坏森林安宁 [cite: 5]。"
    },
    {
        "query": "海马族群得知灵猫即将进犯后，采取了什么初步行动？",
        "ground_truth": "深海海马王召集各分支首领，召开族群议事大会，制定防御与反击战术 [cite: 7]。"
    },
    {
        "query": "黄金海马分支在防御中负责什么？",
        "ground_truth": "负责组建防御阵型，利用发光珊瑚构建水下屏障 [cite: 7]。"
    },
    {
        "query": "冰晶海马分支如何限制灵猫的移动？",
        "ground_truth": "通过冻结海面表层水流来限制灵猫的移动范围 [cite: 7]。"
    },
    {
        "query": "烈焰海马分支在战斗中有什么作用？",
        "ground_truth": "释放温和的热能光束，用于破解幻术、击退近身的灵猫士兵 [cite: 7]。"
    },
    {
        "query": "海马族群启动守护阵法后，入口发生了什么变化？",
        "ground_truth": "将琉璃海马域的入口隐匿，只留下一处狭窄通道作为攻防枢纽 [cite: 7]。"
    },
    {
        "query": "成年海马在战前筹备中做了什么？",
        "ground_truth": "搬运海底礁石加固防线，收集深海珍珠储存能量 [cite: 8]。"
    },
    {
        "query": "海马王对族群反复告诫了什么战术原则？",
        "ground_truth": "若非必要绝不主动出击，力求以防御化解冲突，保留和解的余地 [cite: 8]。"
    },
    {
        "query": "海马在防御时是如何固定身形的？",
        "ground_truth": "尾部缠绕珊瑚礁固定身形 [cite: 8]。"
    },
    {
        "query": "喵喵喵部族将精锐灵猫分成了哪三支小队？",
        "ground_truth": "突袭小队、幻术小队和后勤小队 [cite: 10]。"
    },
    {
        "query": "灵猫幻术小队的任务是什么？",
        "ground_truth": "在后方释放迷幻雾气，干扰海马的判断与协作 [cite: 10]。"
    },
    {
        "query": "灵猫后勤小队携带了什么物资？",
        "ground_truth": "疗伤草药与灵气果实 [cite: 10]。"
    },
    {
        "query": "喵喵喵部族的战术核心是什么？",
        "ground_truth": "速战速决，利用空中优势和幻术压制海马族群，避免陷入水下持久战 [cite: 11]。"
    },
    {
        "query": "为什么灵猫要避免水下持久战？",
        "ground_truth": "因为灵猫不适应水下环境，一旦进入深海，行动力会大幅减弱 [cite: 11]。"
    },
    {
        "query": "灵猫族长在战前宣言中的目标是什么？",
        "ground_truth": "守护家园，彻底击退海马族群，夺回被扰乱的领地灵气 [cite: 10]。"
    },
    {
        "query": "决战之日的海面天气如何？",
        "ground_truth": "晴空万里，海面风平浪静 [cite: 13]。"
    },
    {
        "query": "幻术小队在战斗中释放了什么颜色的雾气？",
        "ground_truth": "粉色迷幻雾气 [cite: 13]。"
    },
    {
        "query": "突袭小队是如何发动攻击的？",
        "ground_truth": "借着雾气的掩护，如箭一般俯冲而下，直扑海马的水流屏障 [cite: 13]。"
    },
    {
        "query": "冰晶海马是如何阻挡雾气渗透的？",
        "ground_truth": "将屏障周边的海水冻结成一层薄冰 [cite: 14]。"
    },
    {
        "query": "黄金海马如何打乱俯冲灵猫的进攻节奏？",
        "ground_truth": "摆动尾部，催动水流形成漩涡，将灵猫包裹其中 [cite: 14]。"
    },
    {
        "query": "烈焰海马是如何破解幻术攻势的？",
        "ground_truth": "精准释放热能光束，灼烧迷幻雾气 [cite: 14]。"
    },
    {
        "query": "当灵猫跃入浅海区域后，为什么会逐渐陷入被动？",
        "ground_truth": "因为深海环境让动作变得迟缓，难以命中灵活的海马，反而被水流困住 [cite: 15]。"
    },
    {
        "query": "海马在近身战中如何反击灵猫？",
        "ground_truth": "利用小巧身形躲避，同时用尾部抽打灵猫的四肢，让其失去平衡 [cite: 15]。"
    },
    {
        "query": "九尾灵猫族长是如何让局势扭转的？",
        "ground_truth": "亲自上阵，释放九尾之力，掀起强劲风刃，劈砍海马的水流屏障使其出现裂痕 [cite: 15]。"
    },
    {
        "query": "海马王如何修补被风刃劈开的屏障裂痕？",
        "ground_truth": "调动全部灵气，凝聚成一道巨大的水幕进行修补 [cite: 16]。"
    },
    {
        "query": "海马王召唤了什么生物来抵挡风刃攻击？",
        "ground_truth": "召唤了深海中的巨型海龟 [cite: 16]。"
    },
    {
        "query": "是什么最终让战场瞬间安静下来并化解了攻击？",
        "ground_truth": "上古灵珠散发微光，一股温和的能量席卷战场，化解了攻击 [cite: 16]。"
    },
    {
        "query": "上古灵珠投射的光影影像还原了什么真相？",
        "ground_truth": "海底地壳轻微运动触发了灵珠的能量外泄，并非海马族群故意攻击 [cite: 18]。"
    },
    {
        "query": "影像展示了双方族群各自的初心是什么？",
        "ground_truth": "海马族群守护深海生态，喵喵喵部族守护森林安宁，本质都是为了守护海域平衡 [cite: 18, 19]。"
    },
    {
        "query": "看到影像后，九尾灵猫族长有什么反应？",
        "ground_truth": "瞬间面露愧色，意识到自己误会了海马族群，贸然发动战争是鲁莽之举 [cite: 20]。"
    },
    {
        "query": "深海海马王如何表达友好的态度？",
        "ground_truth": "主动游向海面，释放出友好的水流信号 [cite: 20]。"
    },
    {
        "query": "战后，冰晶海马如何帮助喵喵森林？",
        "ground_truth": "用纯净的海水滋养喵喵森林的折断灵木，帮助其快速复苏 [cite: 21]。"
    },
    {
        "query": "战后，喵喵喵部族如何帮助海马域？",
        "ground_truth": "利用风元素，吹散海域中多余的雾气，让琉璃海马域的水质更加澄澈 [cite: 21]。"
    },
    {
        "query": "双方最终签订了什么？",
        "ground_truth": "签订了和平契约 [cite: 21]。"
    },
    {
        "query": "和平契约中约定了哪些内容？",
        "ground_truth": "定期互通消息，共同守护生态平衡，遇到自然异象时携手应对，不再轻易发动战争 [cite: 21]。"
    },
    {
        "query": "深海海马族群与喵喵喵部族最后的关系变成了什么？",
        "ground_truth": "从对手变成了挚友 [cite: 22]。"
    },
    {
        "query": "和平之后，海马在阳光下会有什么表现？",
        "ground_truth": "在水下欢快游动 [cite: 22]。"
    },
    {
        "query": "和平之后，灵猫在林间会有什么表现？",
        "ground_truth": "在林间悠闲嬉戏 [cite: 22]。"
    },
    {
        "query": "这部小说或者文档的名字是什么？",
        "ground_truth": "《海马大战喵喵喵:深海秘境与灵猫部族的终极对决》 [cite: 1]。"
    }
]
            print("注意：当前使用的是示例测试集，请确保测试用例与你的 PDF 内容匹配。")
            run_accuracy_test(test_dataset)
            
        else:
            print("\n进入聊天模式 (输入 exit 退出)")
            while True:
                q = input("\n用户提问: ").strip()
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