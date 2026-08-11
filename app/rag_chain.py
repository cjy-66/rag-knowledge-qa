"""
RAG 核心链路模块
负责：Embedding 向量化 → FAISS 索引构建 → 智能迭代检索 → Prompt 构造 → LLM 生成

检索策略（Agentic RAG）：
  1. 首轮 MMR 检索 → LLM 评审资料是否充足
  2. 不充足 → LLM 生成追问 → 再搜一轮 → 合并去重
  3. 最多 N 轮，之后流式回答

Embedding: 使用本地 HuggingFace 模型（免费、无需 API Key）
LLM: 通过 OpenAI 兼容接口调用 DeepSeek 等大模型
多用户: 每用户独立 FAISS 索引，索引目录为 vector_store/{user_id}/
"""
import os
import warnings
import re
from typing import Dict, Generator, List, Set, Tuple, Optional

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

# 检索评估 Prompt：让 LLM 判断当前资料是否足够回答问题
QUERY_REFINER_PROMPT = """你是一个检索质量评估器。判断以下「参考资料」是否足够回答「用户问题」。

## 判断标准
- 如果资料包含问题所需的数据、事实、专有名词解释 → SUFFICIENT
- 如果资料不相关或明显缺少关键信息 → NEED_MORE: <缺失什么？换个什么角度搜？>

## 输出格式（严格二选一）
SUFFICIENT
或
NEED_MORE: <一句简洁的追问/搜索词>

## 用户问题
{question}

## 当前参考资料（共 {count} 条）
{context}

## 你的评估
"""


def _create_embeddings() -> HuggingFaceEmbeddings:
    """创建本地 HuggingFace Embedding 模型实例"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class RAGChain:
    """RAG 检索增强生成链路，支持多用户独立索引 + 智能迭代检索"""

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

        self.refiner_prompt = PromptTemplate(
            template=QUERY_REFINER_PROMPT,
            input_variables=["question", "context", "count"],
        )

        # 多用户向量存储缓存: user_id → FAISS
        self._stores: Dict[str, FAISS] = {}

    # ═══════════════════════════════════════════
    # 路径 & 索引管理
    # ═══════════════════════════════════════════

    def _store_dir(self, user_id: str) -> str:
        return os.path.join(config.VECTOR_STORE_DIR, user_id)

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
        store = self._stores.get(user_id)
        if store is not None:
            store_dir = self._store_dir(user_id)
            os.makedirs(store_dir, exist_ok=True)
            store.save_local(store_dir)

    def _get_or_create_store(self, user_id: str) -> Optional[FAISS]:
        if user_id in self._stores:
            return self._stores[user_id]
        return self._load_user_store(user_id)

    # ═══════════════════════════════════════════
    # 公开 API — 索引构建
    # ═══════════════════════════════════════════

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
        store = self._get_or_create_store(user_id)
        if store is not None:
            return store.index.ntotal
        return 0

    # ═══════════════════════════════════════════
    # 智能迭代检索核心
    # ═══════════════════════════════════════════

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

    def _retrieve_docs(
        self, store: FAISS, query: str, k: int
    ) -> List[Document]:
        """从 FAISS 检索 k 条文档（MMR 优先，失败回退相似度检索）"""
        try:
            return store.max_marginal_relevance_search(
                query, k=k, fetch_k=max(k * 3, 12), lambda_mult=0.6,
            )
        except Exception:
            return store.similarity_search(query, k=k)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """文档去重键：来源 + 内容前 80 字符"""
        return doc.metadata.get("source", "") + "|" + doc.page_content[:80]

    @staticmethod
    def _deduplicate(docs: List[Document]) -> List[Document]:
        """去重：保留首次出现的文档"""
        seen: Set[str] = set()
        result = []
        for d in docs:
            key = RAGChain._doc_key(d)
            if key not in seen:
                seen.add(key)
                result.append(d)
        return result

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        """将文档列表格式化为上下文字符串"""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page")
            if page is not None and isinstance(page, int):
                page_info = f" (第{page + 1}页)"
            else:
                page_info = ""
            parts.append(f"[参考{i}] 来源: {source}{page_info}\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_sources(docs: List[Document]) -> List[dict]:
        """将文档列表格式化为前端来源卡片"""
        sources = []
        for doc in docs:
            sources.append({
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page"),
            })
        return sources

    def _assess_sufficiency(
        self, question: str, docs: List[Document],
    ) -> Optional[str]:
        """让 LLM 评审当前资料是否足够。

        Returns:
            None  → SUFFICIENT，可以回答
            str   → NEED_MORE，返回追问/搜索词
        """
        context = self._format_context(docs)
        refiner_text = self.refiner_prompt.format(
            question=question, context=context, count=len(docs),
        )

        try:
            resp = self.llm.invoke(refiner_text)
            verdict = resp.content.strip()

            if verdict.upper().startswith("SUFFICIENT"):
                return None
            # 提取 NEED_MORE 后面的追问
            if "NEED_MORE:" in verdict.upper():
                idx = verdict.upper().index("NEED_MORE:") + len("NEED_MORE:")
                followup = verdict[idx:].strip()
                return followup[:200] if followup else question
            # 不明确的输出 → 视为 SUFFICIENT，避免无谓重试
            return None
        except Exception:
            return None

    def _iterative_retrieve(
        self, store: FAISS, question: str, k: int, max_rounds: int,
    ) -> List[Document]:
        """智能迭代检索：最多 max_rounds 轮，每轮让 LLM 评审是否足够。

        返回去重合并后的文档列表。
        """
        all_docs: List[Document] = []

        for rnd in range(max_rounds):
            # 决定搜索词：首轮用原问题 + 扩展，后续用 LLM 生成的追问
            search_query = (
                self._expand_query(question) if rnd == 0 else question
            )

            round_docs = self._retrieve_docs(store, search_query, k)
            all_docs.extend(round_docs)
            all_docs = self._deduplicate(all_docs)

            # 最后一轮不评估，直接用
            if rnd >= max_rounds - 1:
                break

            # 让 LLM 评估当前资料是否足够
            followup = self._assess_sufficiency(question, all_docs)
            if followup is None:
                # SUFFICIENT
                break
            # NEED_MORE → 用追问做下一轮搜索
            question = followup

        return all_docs

    # ═══════════════════════════════════════════
    # 公开 API — 问答
    # ═══════════════════════════════════════════

    def query(
        self, user_id: str, question: str, top_k: int = None,
    ) -> Tuple[str, List[dict]]:
        """智能迭代 RAG 问答（非流式）"""
        store = self._get_or_create_store(user_id)
        if store is None:
            return "尚未构建知识库，请先上传文档。", []

        k = top_k or config.TOP_K
        max_rounds = getattr(config, "MAX_RETRIEVAL_ROUNDS", 2)

        retrieved_docs = self._iterative_retrieve(store, question, k, max_rounds)

        context = self._format_context(retrieved_docs)
        prompt_text = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_text)
        answer = response.content
        sources = self._format_sources(retrieved_docs)

        return answer, sources

    def query_stream(
        self, user_id: str, question: str, top_k: int = None,
    ) -> Tuple[Generator[str, None, None], List[dict]]:
        """智能迭代 RAG 流式问答。

        检索阶段：多轮迭代直到 LLM 确认资料充足（非流式，用户无感知）。
        回答阶段：流式逐 token 输出（跟之前一样）。
        """
        store = self._get_or_create_store(user_id)
        if store is None:
            def _empty():
                yield "尚未构建知识库，请先上传文档。"
                return
            return _empty(), []

        k = top_k or config.TOP_K
        max_rounds = getattr(config, "MAX_RETRIEVAL_ROUNDS", 2)

        # ── 智能迭代检索（回答前的静默阶段）──
        retrieved_docs = self._iterative_retrieve(store, question, k, max_rounds)

        # ── 组装上下文 + 来源 ──
        context = self._format_context(retrieved_docs)
        prompt_text = self.prompt.format(context=context, question=question)
        sources = self._format_sources(retrieved_docs)

        # ── 流式生成 ──
        def _token_gen() -> Generator[str, None, None]:
            for chunk in self.llm.stream(prompt_text):
                token = chunk.content
                if token:
                    yield token

        return _token_gen(), sources

    # ═══════════════════════════════════════════
    # 兼容属性
    # ═══════════════════════════════════════════

    def is_ready_for(self, user_id: str) -> bool:
        """检查指定用户的知识库是否就绪"""
        return self._get_or_create_store(user_id) is not None

    def index_size_for(self, user_id: str) -> int:
        """返回指定用户的索引大小"""
        return self._user_index_size(user_id)


# ── 全局单例 ──
rag_chain = RAGChain()
