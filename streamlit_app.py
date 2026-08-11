"""
Streamlit 独立应用 - 智能知识库问答系统
用于 Streamlit Community Cloud 免费部署（无需信用卡）
前后端合并，无需单独部署 FastAPI
多用户: 每个浏览器会话自动生成 user_id，数据完全隔离

本地运行: streamlit run streamlit_app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

try:
    for key in ["OPENAI_API_KEY", "OPENAI_API_BASE", "LLM_MODEL", "EMBEDDING_MODEL"]:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from config import config
from app.document_loader import process_file, get_all_uploaded_files, cleanup_expired_files
from app.ui_base import init_session, render_css, render_sidebar, render_main

st.set_page_config(
    page_title="Knowledge QA",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_rag_chain():
    from app.rag_chain import rag_chain
    return rag_chain


@st.cache_resource
def get_conversation_store():
    from app.conversation_store import conversation_store
    return conversation_store


# ═══════════════════════════════════════
# 数据层（每个函数从 st.session_state 获取 user_id）
# ═══════════════════════════════════════

def _uid() -> str:
    return st.session_state.get("user_id", "")


def _collect_user_chunks(user_id: str) -> list:
    """收集指定用户目录下所有剩余文件的文档块"""
    user_dir = os.path.join(config.DATA_DIR, user_id)
    if not os.path.exists(user_dir):
        return []
    supported_exts = (".pdf", ".txt", ".md", ".markdown", ".csv", ".xlsx", ".xls")
    all_chunks = []
    for fname in os.listdir(user_dir):
        if fname.lower().endswith(supported_exts):
            try:
                chunks = process_file(os.path.join(user_dir, fname))
                all_chunks.extend(chunks)
            except Exception:
                pass
    return all_chunks


def _check_health() -> dict:
    try:
        rag = get_rag_chain()
        uid = _uid()
        return {
            "status": "ok",
            "index_size": rag.index_size_for(uid),
            "index_ready": rag.is_ready_for(uid),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _upload_file(uploaded_file) -> dict:
    uid = _uid()
    user_dir = os.path.join(config.DATA_DIR, uid)
    os.makedirs(user_dir, exist_ok=True)

    # 1) 先清理过期文件
    clean_result = cleanup_expired_files(uid)
    rag = get_rag_chain()
    if clean_result["deleted"]:
        # 有文件被删 → 用剩余文件重建索引
        all_chunks = _collect_user_chunks(uid)
        if all_chunks:
            rag.build_index(uid, all_chunks)
        else:
            rag.clear_index(uid)

    # 2) 保存新文件
    file_path = os.path.join(user_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    chunks = process_file(file_path)
    if not chunks:
        os.remove(file_path)
        return {"error": "未能从文件中提取任何文本内容"}

    try:
        total = rag.add_documents(uid, chunks)
        return {
            "message": f"上传成功: {uploaded_file.name}，新增 {len(chunks)} 个文本块，索引共 {total} 条",
            "chunks": len(chunks),
            "total": total,
        }
    except Exception as e:
        return {"error": f"索引构建失败: {str(e)}"}


def _query_rag(question: str, top_k: int = 4) -> dict:
    rag = get_rag_chain()
    try:
        answer, sources = rag.query(_uid(), question, top_k)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}


def _query_stream(question: str, top_k: int = 4):
    rag = get_rag_chain()
    try:
        return rag.query_stream(_uid(), question, top_k)
    except Exception as e:
        def _err():
            yield f"查询失败: {str(e)}"
        return _err(), []


def _list_documents() -> list:
    return get_all_uploaded_files(_uid())


def _delete_document(filename: str) -> bool:
    uid = _uid()
    file_path = os.path.join(config.DATA_DIR, uid, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def _list_conversations() -> list:
    return get_conversation_store().list_all(_uid())


def _create_conversation(title: str = "") -> dict:
    return get_conversation_store().create(_uid(), title)


def _get_conversation(conv_id: str) -> dict:
    return get_conversation_store().get(_uid(), conv_id) or {}


def _add_message(conv_id: str, role: str, content: str, sources: list = None):
    get_conversation_store().add_message(_uid(), conv_id, role, content, sources)


def _delete_conversation(conv_id: str) -> bool:
    return get_conversation_store().delete(_uid(), conv_id)


dl = {
    "health":       _check_health,
    "list_convs":   _list_conversations,
    "create_conv":  _create_conversation,
    "get_conv":     _get_conversation,
    "delete_conv":  _delete_conversation,
    "add_msg":      _add_message,
    "list_docs":    _list_documents,
    "upload_doc":   _upload_file,
    "delete_doc":   _delete_document,
    "query":        _query_rag,
    "query_stream": _query_stream,
}


if __name__ == "__main__":
    init_session()
    render_css()
    render_sidebar(dl)
    render_main(dl)
