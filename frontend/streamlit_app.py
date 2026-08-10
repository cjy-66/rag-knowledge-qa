"""
Streamlit 前端 - 智能知识库问答系统交互界面（客户端 / 服务器模式）
通过 HTTP API 与 FastAPI 后端通信。

本地运行: streamlit run frontend/streamlit_app.py
需要先启动后端: python run.py backend
"""
import os
import sys
import time
import requests
import streamlit as st

# ── 配置 ──
API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── 页面设置（必须在所有 Streamlit 命令最前面）──
st.set_page_config(
    page_title="Knowledge QA",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 确保项目根目录在 Python path 中（以便 import app.ui_base）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.ui_base import init_session, render_css, render_sidebar, render_main


# ═══════════════════════════════════════
# 数据层：通过 HTTP API 调用后端
# ═══════════════════════════════════════

def _api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.json() if r.ok else {"status": "error"}
    except Exception:
        return {"status": "unreachable"}


def _api_list_convs() -> list:
    try:
        r = requests.get(f"{API_BASE}/conversations", timeout=5)
        return r.json().get("conversations", []) if r.ok else []
    except Exception:
        return []


def _api_create_conv(title: str = "") -> dict:
    try:
        r = requests.post(f"{API_BASE}/conversations", json={"title": title}, timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _api_get_conv(conv_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/conversations/{conv_id}", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _api_add_msg(conv_id: str, role: str, content: str, sources: list = None):
    try:
        requests.post(
            f"{API_BASE}/conversations/{conv_id}/messages",
            json={"role": role, "content": content, "sources": sources or []},
            timeout=5,
        )
    except Exception:
        pass


def _api_delete_conv(conv_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/conversations/{conv_id}", timeout=5)
        return r.ok
    except Exception:
        return False


def _api_upload(file) -> dict:
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        r = requests.post(f"{API_BASE}/upload", files=files, timeout=300)
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def _api_query(question: str, top_k: int = None) -> dict:
    try:
        body = {"question": question}
        if top_k:
            body["top_k"] = top_k
        r = requests.post(f"{API_BASE}/query", json=body, timeout=120)
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def _api_list_docs() -> list:
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        return r.json().get("files", []) if r.ok else []
    except Exception:
        return []


def _api_delete_doc(filename: str) -> dict:
    try:
        r = requests.delete(f"{API_BASE}/documents/{filename}", timeout=10)
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ── 组装数据层 ──

dl = {
    "health":      _api_health,
    "list_convs":  _api_list_convs,
    "create_conv": _api_create_conv,
    "get_conv":    _api_get_conv,
    "delete_conv": _api_delete_conv,
    "add_msg":     _api_add_msg,
    "list_docs":   _api_list_docs,
    "upload_doc":  _api_upload,
    "delete_doc":  _api_delete_doc,
    "query":       _api_query,
}


# ═══════════════════════════════════════
# 运行
# ═══════════════════════════════════════

if __name__ == "__main__":
    init_session()
    render_css()
    render_sidebar(dl)
    render_main(dl)
