import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag_system.config import CACHE_DIR, MEDICAL_COLLECTION_NAME
from rag_system.core.resources import get_medical_collection, load_resources
from rag_system.services.indexing import save_knowledge_base
from rag_system.services.medical_loader import load_medical_chunks
from rag_system.core.state import KnowledgeBase
import jieba
from rank_bm25 import BM25Okapi


def add_batch(collection, embed_model, chunks, start_index: int, batch_size: int = 5000):
    end_index = min(start_index + batch_size, len(chunks))
    batch = chunks[start_index:end_index]
    child_texts = [item["child"] for item in batch]
    metadatas = [{"parent": item["parent"]} for item in batch]
    ids = [f"id_{index}" for index in range(start_index, end_index)]

    print(f"  encoding {start_index}~{end_index}...", flush=True)
    embeddings = embed_model.encode(child_texts, batch_size=32, show_progress_bar=False).tolist()

    print(f"  adding to chroma...", flush=True)
    collection.add(
        documents=child_texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"  done {end_index}/{len(chunks)}", flush=True)
    return end_index


def main():
    start = time.time()
    try:
        print("[1/3] 加载模型与资源...", flush=True)
        resources = load_resources()

        print("[2/3] 解析 medical.json...", flush=True)
        t0 = time.time()
        chunks = load_medical_chunks()
        print(f"  子块数: {len(chunks)}, 耗时: {time.time() - t0:.1f}s", flush=True)

        medical_collection = get_medical_collection()
        current_count = medical_collection.count()
        print(f"[3/3] 当前 collection 数量: {current_count}", flush=True)

        if current_count < len(chunks):
            next_index = add_batch(medical_collection, resources.embed_model, chunks, current_count, batch_size=3000)
            print(f"  已索引到 {next_index}/{len(chunks)}", flush=True)
        else:
            print("  所有向量已存在，开始构建 BM25 缓存...", flush=True)
            child_texts = [item["child"] for item in chunks]
            tokenized_corpus = [list(jieba.cut(text)) for text in child_texts]
            bm25_engine = BM25Okapi(tokenized_corpus)
            kb = KnowledgeBase(
                chunks_data=chunks,
                child_texts=child_texts,
                bm25_engine=bm25_engine,
            )
            save_knowledge_base(kb, MEDICAL_COLLECTION_NAME, CACHE_DIR)
            print("  BM25 缓存保存完成", flush=True)

        print(f"总耗时: {time.time() - start:.1f}s", flush=True)
    except Exception as exc:
        print(f"错误: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
