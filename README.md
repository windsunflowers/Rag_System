# 医疗知识库问答与评测系统

基于大语言模型（通义千问）与 RAG 的本地医疗知识库问答系统，配套 RAGAS 四维评测。定位为专业医疗知识学习，非就医导诊。

## 文件结构

rag-system/
├── rag_system/                    # 核心源码包
│   ├── config.py                  # 全局配置：模型选型、检索/重排参数、模型目录
│   ├── core/
│   │   ├── resources.py           # 资源初始化（embedding、重排器、LLM 客户端）
│   │   └── state.py               # 会话状态管理
│   ├── services/
│   │   ├── document_processing.py # 文档解析（PDF / 图片 / 文本）
│   │   ├── indexing.py            # 父子分块与向量索引构建
│   │   ├── medical_loader.py      # 医疗 JSON 数据加载
│   │   ├── rag_pipeline.py        # 检索主流程：查询改写、双路召回、重排、上下文拼装
│   │   └── evaluation.py          # RAGAS 评测：自动出题与四维指标
│   └── ui/
│       └── app.py                 # Streamlit 交互问答与测评界面
├── build_medical_index.py         # 索引构建脚本（从 medical.json 生成 chroma 索引）
├── check.py                       # 启动入口
├── requirements.txt               # Python 依赖
├── .env.example                   # 环境变量模板（复制为 .env 后填入密钥）
├── .gitignore
├── README.md
└── SECURITY.md

## 运行方式

1. 安装依赖：pip install -r requirements.txt
2. 配置密钥：cp .env.example .env，填入 DASHSCOPE_API_KEY 与 LLM_BASE_URL
3. 构建索引：python build_medical_index.py（需要本地 medical.json 原始数据）
4. 启动界面：streamlit run rag_system/ui/app.py

> 说明：medical.json（原始医疗数据）、chroma_db/（向量索引）、cache/ 等因体积与隐私原因不纳入版本库，请本地保留。

## 优化概述

相对初始版本的主要优化：

- 重排模型升级：mmarco-mMiniLMv2 换为 BAAI/bge-reranker-large，重排质量显著提升。
- 检索/重排参数调优：RETRIEVAL_TOP_K=8、RERANK_TOP_N=8 为最终稳定配置。
- 父子分块 + 完整父块上下文：以完整疾病卡片作为上下文，不做跨维度裁剪，避免事实丢失。
- 维度感知排序增强：按查询意图维度优先匹配相关父子块，但不硬裁剪。
- 免责声明剥离评测：固定免责声明不计入 context_precision 计算。
- 评测出题去重与多样性：子块采样、疾病互异、维度打散，答案保留完整不精简。
- 多轮对话上下文记忆修复：输入框定位与历史上下文正确传递。
- 模型目录与价格提示：内置可调用 qwen 系列型号及输入输出价格悬停说明。

评测指标（RAGAS，真实基线 context_recall 76% / context_precision 45%）：

- 重排升级 + 完整父块（top10）：context_recall 86%、context_precision 59%、faithfulness 96%、answer_relevancy 87%
- 最终稳定版（top8）：context_precision 55%、context_recall 76%、faithfulness 89%、answer_relevancy 92%

核心配置：bge-reranker-large + bge-small-zh-v1.5 + topK8 + topN8 + 完整父块上下文。
