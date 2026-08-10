"""
RAG 核心链路模块
负责：Embedding 向量化 → FAISS 索引构建 → 语义检索 → Prompt 构造 → LLM 生成

Embedding: 使用本地 HuggingFace 模型（免费、无需 API Key）
LLM: 通过 OpenAI 兼容接口调用 DeepSeek 等大模型
"""
import os
from typing import List, Tuple, Optional

# ⚠️ config 必须在 langchain_huggingface 之前导入，
# 以确保 HF_ENDPOINT 镜像环境变量先生效
from config import config

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── 中文优化 RAG Prompt 模板 ──
RAG_PROMPT_TEMPLATE = """你是一个智能知识库助手，请根据以下参考资料回答用户的问题。

## 规则
1. 如果参考资料中有相关信息，请基于资料内容进行回答。
2. 如果参考资料中没有相关信息，请如实告知用户"该问题在已有文档中未找到相关信息"。
3. 回答时请尽量引用原文，保持准确性和完整性。
4. 如果问题涉及多个方面，请分点作答。

## 参考资料
{context}

## 用户问题
{question}

## 回答
"""


def _create_embeddings() -> HuggingFaceEmbeddings:
    """
    创建本地 HuggingFace Embedding 模型实例
    优先使用中文优化模型 bge-small-zh-v1.5，体积小、速度快
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class RAGChain:
    """
    RAG 检索增强生成链路

    使用方式:
        # 1. 初始化
        rag = RAGChain()

        # 2. 构建向量索引（每次上传新文档后调用一次）
        rag.build_index(all_chunks)

        # 3. 查询
        answer, sources = rag.query("什么是RAG？")
        print(answer)
    """

    def __init__(self):
        """初始化 Embedding 模型 & LLM"""
        print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
        self.embeddings = _create_embeddings()

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.OPENAI_API_BASE,
        )

        self.vector_store: Optional[FAISS] = None
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        # 尝试加载已有索引
        self._load_index()

    def build_index(self, documents: List[Document]) -> int:
        """
        构建 FAISS 向量索引（全量重建）

        参数:
            documents: 文档块列表

        返回:
            int: 索引中的文档块数量
        """
        if not documents:
            raise ValueError("文档列表为空，无法构建索引")

        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self._save_index()
        return len(documents)

    def add_documents(self, documents: List[Document]) -> int:
        """
        增量添加文档到现有索引（不从零重建）

        参数:
            documents: 新增文档块列表

        返回:
            int: 添加后索引中的文档块总数
        """
        if not documents:
            return self.index_size

        if self.vector_store is None:
            return self.build_index(documents)

        self.vector_store.add_documents(documents)
        self._save_index()
        return self.index_size

    def _save_index(self):
        """持久化向量索引到磁盘"""
        if self.vector_store is not None:
            os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
            self.vector_store.save_local(config.VECTOR_STORE_DIR)

    def _load_index(self) -> bool:
        """从磁盘加载已持久化的向量索引"""
        index_path = os.path.join(config.VECTOR_STORE_DIR, "index.faiss")
        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    config.VECTOR_STORE_DIR,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return True
            except Exception as e:
                print(f"加载索引失败: {e}")
        return False

    def query(
        self, question: str, top_k: int = None
    ) -> Tuple[str, List[dict]]:
        """
        执行 RAG 问答

        参数:
            question: 用户问题
            top_k:   检索文档数量

        返回:
            Tuple[str, List[dict]]: (回答文本, 参考来源列表)
        """
        if self.vector_store is None:
            return "尚未构建知识库，请先上传文档。", []

        k = top_k or config.TOP_K

        # 1. 相似度检索
        retrieved_docs = self.vector_store.similarity_search(question, k=k)

        # 2. 拼接上下文
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "")
            page_info = f" (第{page + 1}页)" if page != "" else ""
            context_parts.append(f"[参考{i}] 来源: {source}{page_info}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # 3. 构造 Prompt
        prompt_text = self.prompt.format(context=context, question=question)

        # 4. 调用 LLM (DeepSeek) 生成回答
        response = self.llm.invoke(prompt_text)
        answer = response.content

        # 5. 组装来源信息
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page"),
            })

        return answer, sources

    @property
    def is_ready(self) -> bool:
        """检查知识库是否已就绪"""
        return self.vector_store is not None

    @property
    def index_size(self) -> int:
        """返回索引中向量数量"""
        if self.vector_store is not None:
            return self.vector_store.index.ntotal
        return 0


# ── 全局单例 ──
rag_chain = RAGChain()
