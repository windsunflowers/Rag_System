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
</style>
"""

AVAILABLE_MODELS = [
    "qwen-max",
    "qwen3.5-plus",
    "qwen-plus",
    "qwen3.5-flash",
    "qwen-turbo",
    "qwen-long",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct",
]

SUPPORTED_FILE_TYPES = ["pdf", "docx", "jpg", "jpeg", "png"]

DEFAULT_ANSWER_MODEL_INDEX = 3
DEFAULT_GENERATOR_MODEL_INDEX = 1
DEFAULT_JUDGE_MODEL_INDEX = 0

VECTOR_COLLECTION_NAME = "rag_docs_standard"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
VISION_MODEL_NAME = "qwen-vl-max"
QUERY_REWRITE_MODEL_NAME = "qwen-turbo"
