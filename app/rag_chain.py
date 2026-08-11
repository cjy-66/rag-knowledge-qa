"""
RAG 核心链路模块
负责：Embedding 向量化 → FAISS 索引构建 → 语义检索 → Prompt 构造 → LLM 生成

Embedding: 使用本地 HuggingFace 模型（免费、无需 API Key）
LLM: 通过 OpenAI 兼容接口调用 DeepSeek 等大模型
多用户: 每用户独立 FAISS 索引，索引目录为 vector_store/{user_id}/
"""
import os
import warnings
import re
from typing import Dict, Generator, List, Tuple, Optional

from config import config

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


RAG_PROMPT_TEMPLATE = """你是一个知识增强型智能助手。你的核心能力是**结合用户上传的文档资料与你自身的广泛知识**来回答问题。

## 核心原则
- 📄 **文档优先**：涉及具体数据、事实、数字时，必须严格基于参考资料
- 🧠 **知识补充**：对专有名词、行业术语、背景概念，如果参考资料中没有解释，请用你的知识补充说明
- 🏷️ **明确标注**：让用户清楚知道哪些信息来自文档、哪些是你的补充

## 回答结构
1. **基于文档的回答**：引用参考资料中的数据和事实来回答问题
2. **知识补充**（如果问题涉及参考资料中未详细解释的专有名词/术语/背景，请用「💡 补充」标记进行解释）
3. **来源引用**：涉及文档数据时注明出处

## 重要
- 不要只因为参考资料没有详细解释某个名词就说"未找到信息"
- 表格中的数据点（如公司名、产品名）可能没有附带解释，请用你的知识补充
- 如果参考资料完全不相关，如实说明后再基于你的知识提供帮助

## 参考资料
{context}

## 用户问题
{question}

## 回答
"""


def _create_embeddings() -> HuggingFaceEmbeddings:
    """创建本地 HuggingFace Embedding 模型实例"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class RAGChain:
    """RAG 检索增强生成链路，支持多用户独立索引"""

    def __init__(self):
        """初始化 Embedding 模型 & LLM（所有用户共享）"""
        print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
        self.embeddings = _create_embeddings()

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.OPENAI_API_BASE,
        )

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        # 多用户向量存储缓存: user_id → FAISS
        self._stores: Dict[str, FAISS] = {}

    # ── 路径工具 ──
    def _store_dir(self, user_id: str) -> str:
        return os.path.join(config.VECTOR_STORE_DIR, user_id)

    # ── 用户索引加载/保存 ──
    def _load_user_store(self, user_id: str) -> Optional[FAISS]:
        """从磁盘加载指定用户的 FAISS 索引"""
        if user_id in self._stores:
            return self._stores[user_id]

        store_dir = self._store_dir(user_id)
        index_path = os.path.join(store_dir, "index.faiss")
        if not os.path.exists(index_path):
            return None

        if not config.FAISS_ALLOW_DANGEROUS_DESERIALIZATION:
            warnings.warn("FAISS 索引加载已禁用，跳过。")
            return None

        try:
            store = FAISS.load_local(
                store_dir, self.embeddings, allow_dangerous_deserialization=True,
            )
            self._stores[user_id] = store
            return store
        except Exception as e:
            print(f"加载用户 {user_id} 索引失败: {e}")
            return None

    def _save_user_store(self, user_id: str):
        """持久化指定用户的向量索引"""
        store = self._stores.get(user_id)
        if store is not None:
            store_dir = self._store_dir(user_id)
            os.makedirs(store_dir, exist_ok=True)
            store.save_local(store_dir)

    def _get_or_create_store(self, user_id: str) -> Optional[FAISS]:
        """获取用户索引：先查缓存，再查磁盘"""
        if user_id in self._stores:
            return self._stores[user_id]
        return self._load_user_store(user_id)

    # ── 公开 API ──

    def build_index(self, user_id: str, documents: List[Document]) -> int:
        """构建用户的 FAISS 向量索引（全量重建）"""
        if not documents:
            raise ValueError("文档列表为空，无法构建索引")

        store = FAISS.from_documents(documents, self.embeddings)
        self._stores[user_id] = store
        self._save_user_store(user_id)
        return len(documents)

    def add_documents(self, user_id: str, documents: List[Document]) -> int:
        """增量添加文档到用户的索引"""
        if not documents:
            return self._user_index_size(user_id)

        store = self._get_or_create_store(user_id)
        if store is None:
            return self.build_index(user_id, documents)

        store.add_documents(documents)
        self._stores[user_id] = store
        self._save_user_store(user_id)
        return self._user_index_size(user_id)

    def clear_index(self, user_id: str):
        """删除用户的整个向量索引（磁盘 + 缓存）"""
        self._stores.pop(user_id, None)
        store_dir = self._store_dir(user_id)
        if os.path.exists(store_dir):
            import shutil
            shutil.rmtree(store_dir)

    def _user_index_size(self, user_id: str) -> int:
        """获取用户索引中的向量数量"""
        store = self._get_or_create_store(user_id)
        if store is not None:
            return store.index.ntotal
        return 0

    def _expand_query(self, question: str) -> str:
        """轻量级查询扩展"""
        cn_words = re.findall(r'[一-鿿]{2,}', question)
        en_words = re.findall(r'[A-Za-z0-9.%]{2,}', question)

        expansions = set()
        for w in cn_words + en_words:
            if len(w) >= 2:
                expansions.add(w)

        if expansions:
            return question + " " + " ".join(sorted(expansions))
        return question

    def query(
        self, user_id: str, question: str, top_k: int = None
    ) -> Tuple[str, List[dict]]:
        """执行 RAG 问答"""
        store = self._get_or_create_store(user_id)
        if store is None:
            return "尚未构建知识库，请先上传文档。", []

        k = top_k or config.TOP_K
        expanded_query = self._expand_query(question)

        try:
            retrieved_docs = store.max_marginal_relevance_search(
                expanded_query, k=k, fetch_k=max(k * 3, 12), lambda_mult=0.6,
            )
        except Exception:
            retrieved_docs = store.similarity_search(expanded_query, k=k)

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page")
            if page is not None and isinstance(page, int):
                page_info = f" (第{page + 1}页)"
            else:
                page_info = ""
            context_parts.append(
                f"[参考{i}] 来源: {source}{page_info}\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)
        prompt_text = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_text)
        answer = response.content

        sources = []
        for doc in retrieved_docs:
            sources.append({
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page"),
            })

        return answer, sources

    def query_stream(
        self, user_id: str, question: str, top_k: int = None
    ) -> Tuple[Generator[str, None, None], List[dict]]:
        """流式 RAG 问答"""
        store = self._get_or_create_store(user_id)
        if store is None:
            def _empty():
                yield "尚未构建知识库，请先上传文档。"
                return
            return _empty(), []

        k = top_k or config.TOP_K
        expanded_query = self._expand_query(question)

        try:
            retrieved_docs = store.max_marginal_relevance_search(
                expanded_query, k=k, fetch_k=max(k * 3, 12), lambda_mult=0.6,
            )
        except Exception:
            retrieved_docs = store.similarity_search(expanded_query, k=k)

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page")
            if page is not None and isinstance(page, int):
                page_info = f" (第{page + 1}页)"
            else:
                page_info = ""
            context_parts.append(
                f"[参考{i}] 来源: {source}{page_info}\n{doc.page_content}"
            )
        context = "\n\n".join(context_parts)
        prompt_text = self.prompt.format(context=context, question=question)

        sources = []
        for doc in retrieved_docs:
            sources.append({
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page"),
            })

        def _token_gen() -> Generator[str, None, None]:
            for chunk in self.llm.stream(prompt_text):
                token = chunk.content
                if token:
                    yield token

        return _token_gen(), sources

    # ── 兼容旧的通用属性 ──
    def is_ready_for(self, user_id: str) -> bool:
        """检查指定用户的知识库是否就绪"""
        return self._get_or_create_store(user_id) is not None

    def index_size_for(self, user_id: str) -> int:
        """返回指定用户的索引大小"""
        return self._user_index_size(user_id)


# ── 全局单例 ──
rag_chain = RAGChain()
