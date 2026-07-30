import os

PAGE_TITLE = "智能文档解析与评测系统"
PAGE_LAYOUT = "wide"

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
    /* 限制聊天消息内标题字号，避免模型输出 # 标题时首段过大 */
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        font-size: 1.15em !important;
        font-weight: 600 !important;
        margin: 0.4em 0 0.2em 0 !important;
    }
    [data-testid="stChatMessage"] p {
        margin: 0.2em 0 !important;
    }
    [data-testid="stChatMessage"] li {
        margin: 0.15em 0 !important;
    }
    /* 模型选择旁的灰色感叹号 + 悬停气泡 */
    .model-tip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #9aa0a6;
        font-size: 17px;
        line-height: 38px;
        margin-left: 4px;
    }
    .model-tip .model-tip-box {
        visibility: hidden;
        opacity: 0;
        transition: opacity .15s ease-in-out;
        position: absolute;
        right: 0;
        top: 40px;
        z-index: 1000;
        width: 280px;
        background: #f8f9fa;
        color: #202124;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 12px;
        line-height: 1.6;
        box-shadow: 0 2px 10px rgba(0,0,0,.18);
        text-align: left;
        white-space: normal;
        font-weight: 400;
    }
    .model-tip:hover .model-tip-box {
        visibility: visible;
        opacity: 1;
    }
    .model-tip-box b { color: #1a73e8; }
</style>
"""

# DashScope (Qwen) OpenAI-compatible API
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Model roles on DashScope (Qwen)
QUERY_REWRITE_MODEL_NAME = "qwen-turbo"
DEFAULT_ANSWER_MODEL_NAME = "qwen-plus"
DEFAULT_GENERATOR_MODEL_NAME = "qwen-plus"
DEFAULT_JUDGE_MODEL_NAME = "qwen-max"

# 可调用模型目录（价格单位：元 / 百万 tokens，取自阿里云百炼官方计费页，
# 华北2（北京）地域、非思考模式、0~128K 标准阶梯；实际以官方实时计费为准）
MODEL_CATALOG = [
    {
        "id": "qwen-plus",
        "name": "通义千问 Plus",
        "desc": "综合能力强、性价比高，适合绝大多数医疗问答、出题与生成任务。",
        "context": "128K（思考/长上下文可达 256K）",
        "input_price": "0.8",
        "output_price": "2.0",
        "modality": "文本",
    },
    {
        "id": "qwen-max",
        "name": "通义千问 Max",
        "desc": "效果最强，擅长复杂推理与多步判断，推荐作为评测裁判模型。",
        "context": "32K（思考模式 128K）",
        "input_price": "2.4",
        "output_price": "9.6",
        "modality": "文本",
    },
    {
        "id": "qwen-turbo",
        "name": "通义千问 Turbo",
        "desc": "极速、低成本，适合 Query 改写、简单分类等高频轻量任务。",
        "context": "1M",
        "input_price": "0.3",
        "output_price": "0.6",
        "modality": "文本",
    },
    {
        "id": "qwen-long",
        "name": "通义千问 Long",
        "desc": "超长上下文（1M tokens），适合长文档、长病历的整体理解与检索。",
        "context": "1M",
        "input_price": "0.5",
        "output_price": "2.0",
        "modality": "文本",
    },
    {
        "id": "qwen-vl-plus",
        "name": "通义千问 VL Plus",
        "desc": "多模态视觉模型，可理解图片/图表，适合带图医疗资料问答。",
        "context": "32K（图文）",
        "input_price": "0.8",
        "output_price": "2.0",
        "modality": "视觉/文本",
    },
    {
        "id": "qwen-vl-max",
        "name": "通义千问 VL Max",
        "desc": "视觉旗舰，图文理解最强，适合复杂医学影像/检查单解析。",
        "context": "32K（图文）",
        "input_price": "1.6",
        "output_price": "4.0",
        "modality": "视觉/文本",
    },
]

AVAILABLE_MODELS = [m["id"] for m in MODEL_CATALOG]
MODEL_INFO = {m["id"]: m for m in MODEL_CATALOG}

DEFAULT_ANSWER_MODEL_INDEX = 0    # qwen-plus
DEFAULT_GENERATOR_MODEL_INDEX = 0 # qwen-plus
DEFAULT_JUDGE_MODEL_INDEX = 1     # qwen-max

SUPPORTED_FILE_TYPES = ["pdf", "docx", "jpg", "jpeg", "png"]

# Vector & retrieval config
VECTOR_COLLECTION_NAME = "rag_docs_standard"
MEDICAL_COLLECTION_NAME = "medical_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"
VISION_MODEL_NAME = "qwen-vl-plus"

# Medical RAG retrieval parameters
RETRIEVAL_TOP_K = 8
RERANK_TOP_N = 8
MAX_PARENT_TEXT_LENGTH = 20000  # 父块全字段保留，但设置一个安全上限防止极端异常
CHILD_SLIDE_WINDOW = 600        # 长字段内部滑窗切子块（仅影响检索粒度）

# Persistence paths (relative to project root)
PERSIST_DIR = "./chroma_db"
CACHE_DIR = "./cache"
MEDICAL_JSON_PATH = "./medical.json"
