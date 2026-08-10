"""
Streamlit 独立应用 - 智能知识库问答系统
用于 Streamlit Community Cloud 免费部署（无需信用卡）
前后端合并，无需单独部署 FastAPI

本地运行: streamlit run streamlit_app.py
"""
import os
import sys

# 确保项目根目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# Streamlit Cloud: 将 secrets 注入环境变量（供 config.py 读取）
# 本地开发：.env 文件 > 环境变量；云端：Streamlit Secrets
try:
    for key in ["OPENAI_API_KEY", "OPENAI_API_BASE", "LLM_MODEL", "EMBEDDING_MODEL"]:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # st.secrets 在本地不可用时跳过

from config import config
from app.document_loader import process_file, get_all_uploaded_files
from app.ui_base import init_session, render_css, render_sidebar, render_main


# ═══════════════════════════════════════
# 页面设置（必须在所有 Streamlit 命令最前面）
# ═══════════════════════════════════════

st.set_page_config(
    page_title="Knowledge QA",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════
# 重量级资源（仅初始化一次）
# ═══════════════════════════════════════

@st.cache_resource
def get_rag_chain():
    """RAG 链路：Embedding 模型 + FAISS 索引 + LLM — 全局单例"""
    from app.rag_chain import rag_chain
    return rag_chain


@st.cache_resource
def get_conversation_store():
    """对话持久化存储 — 全局单例"""
    from app.conversation_store import conversation_store
    return conversation_store


# ═══════════════════════════════════════
# 数据层：业务函数（直接调用后端模块，无 HTTP）
# ═══════════════════════════════════════

def _check_health() -> dict:
    """检查知识库状态"""
    try:
        rag = get_rag_chain()
        return {
            "status": "ok",
            "index_size": rag.index_size,
            "index_ready": rag.is_ready,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _upload_file(uploaded_file) -> dict:
    """保存上传文件并加入向量索引"""
    save_dir = config.DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    chunks = process_file(file_path)
    if not chunks:
        os.remove(file_path)
        return {"error": "未能从文件中提取任何文本内容"}

    rag = get_rag_chain()
    try:
        total = rag.add_documents(chunks)
        return {
            "message": f"上传成功: {uploaded_file.name}，新增 {len(chunks)} 个文本块，索引共 {total} 条",
            "chunks": len(chunks),
            "total": total,
        }
    except Exception as e:
        return {"error": f"索引构建失败: {str(e)}"}


def _query_rag(question: str, top_k: int = 4) -> dict:
    """执行 RAG 问答"""
    rag = get_rag_chain()
    try:
        answer, sources = rag.query(question, top_k)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}


def _list_documents() -> list:
    """列出已上传文档"""
    return get_all_uploaded_files()


def _delete_document(filename: str) -> bool:
    """删除文档（从磁盘移除；需重建索引才能彻底清除 FAISS 中的旧向量）"""
    file_path = os.path.join(config.DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def _list_conversations() -> list:
    """列出所有对话"""
    return get_conversation_store().list_all()


def _create_conversation(title: str = "") -> dict:
    """新建对话"""
    return get_conversation_store().create(title)


def _get_conversation(conv_id: str) -> dict:
    """获取单个对话详情"""
    return get_conversation_store().get(conv_id) or {}


def _add_message(conv_id: str, role: str, content: str, sources: list = None):
    """向对话追加一条消息"""
    get_conversation_store().add_message(conv_id, role, content, sources)


def _delete_conversation(conv_id: str) -> bool:
    """删除对话"""
    return get_conversation_store().delete(conv_id)


# ── 组装数据层 ──

dl = {
    "health":      _check_health,
    "list_convs":  _list_conversations,
    "create_conv": _create_conversation,
    "get_conv":    _get_conversation,
    "delete_conv": _delete_conversation,
    "add_msg":     _add_message,
    "list_docs":   _list_documents,
    "upload_doc":  _upload_file,
    "delete_doc":  _delete_document,
    "query":       _query_rag,
}


# ═══════════════════════════════════════
# 运行
# ═══════════════════════════════════════

if __name__ == "__main__":
    init_session()
    render_css()
    render_sidebar(dl)
    render_main(dl)
