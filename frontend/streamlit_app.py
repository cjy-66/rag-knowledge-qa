"""
Streamlit 前端 - 智能知识库问答系统（客户端 / 服务器模式）
通过 HTTP API 与 FastAPI 后端通信，带重试机制和流式输出。
多用户: 通过 X-User-ID 请求头区分用户，每人独立数据和索引。

本地运行: streamlit run frontend/streamlit_app.py
需要先启动后端: python run.py backend
"""
import os
import sys
import json
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Knowledge QA",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.ui_base import init_session, render_css, render_sidebar, render_main


# ── 工具 ──

def _uid() -> str:
    """获取当前会话的 user_id"""
    return st.session_state.get("user_id", "")


def _headers() -> dict:
    return {"X-User-ID": _uid()}


# ── 重试 Session ──

RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT", "DELETE"],
)


def _make_session(timeout: int = 30) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.timeout = timeout
    return s


# ── 数据层 ──

def _api_health() -> dict:
    try:
        r = _make_session(2).get(f"{API_BASE}/health", headers=_headers())
        return r.json() if r.ok else {"status": "error"}
    except Exception:
        return {"status": "unreachable"}


def _api_list_convs() -> list:
    try:
        r = _make_session(5).get(f"{API_BASE}/conversations", headers=_headers())
        return r.json().get("conversations", []) if r.ok else []
    except Exception:
        return []


def _api_create_conv(title: str = "") -> dict:
    try:
        r = _make_session(5).post(f"{API_BASE}/conversations", json={"title": title}, headers=_headers())
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _api_get_conv(conv_id: str) -> dict:
    try:
        r = _make_session(5).get(f"{API_BASE}/conversations/{conv_id}", headers=_headers())
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _api_add_msg(conv_id: str, role: str, content: str, sources: list = None):
    try:
        _make_session(5).post(
            f"{API_BASE}/conversations/{conv_id}/messages",
            json={"role": role, "content": content, "sources": sources or []},
            headers=_headers(),
        )
    except Exception:
        pass


def _api_delete_conv(conv_id: str) -> bool:
    try:
        r = _make_session(5).delete(f"{API_BASE}/conversations/{conv_id}", headers=_headers())
        return r.ok
    except Exception:
        return False


def _api_upload(file) -> dict:
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        r = _make_session(300).post(f"{API_BASE}/upload", files=files, headers=_headers())
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def _api_query(question: str, top_k: int = None) -> dict:
    try:
        body = {"question": question}
        if top_k:
            body["top_k"] = top_k
        r = _make_session(120).post(f"{API_BASE}/query", json=body, headers=_headers())
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def _api_query_stream(question: str, top_k: int = None):
    body = {"question": question}
    if top_k:
        body["top_k"] = top_k

    try:
        resp = _make_session(120).post(
            f"{API_BASE}/query/stream", json=body, stream=True, headers=_headers(),
        )
        resp.raise_for_status()
        lines = resp.iter_lines()

        try:
            first = json.loads(next(lines))
            sources = first.get("data", []) if first.get("type") == "sources" else []
        except (StopIteration, json.JSONDecodeError):
            sources = []

        def _token_gen():
            for line in lines:
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    if evt.get("type") == "token":
                        yield evt["data"]
                    elif evt.get("type") == "done":
                        return
                except json.JSONDecodeError:
                    continue
            resp.close()

        return _token_gen(), sources
    except Exception as e:
        def _err():
            yield f"查询失败: {str(e)}"
        return _err(), []


def _api_list_docs() -> list:
    try:
        r = _make_session(5).get(f"{API_BASE}/documents", headers=_headers())
        return r.json().get("files", []) if r.ok else []
    except Exception:
        return []


def _api_delete_doc(filename: str) -> dict:
    try:
        r = _make_session(10).delete(f"{API_BASE}/documents/{filename}", headers=_headers())
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


dl = {
    "health":        _api_health,
    "list_convs":    _api_list_convs,
    "create_conv":   _api_create_conv,
    "get_conv":      _api_get_conv,
    "delete_conv":   _api_delete_conv,
    "add_msg":       _api_add_msg,
    "list_docs":     _api_list_docs,
    "upload_doc":    _api_upload,
    "delete_doc":    _api_delete_doc,
    "query":         _api_query,
    "query_stream":  _api_query_stream,
}


if __name__ == "__main__":
    init_session()
    render_css()
    render_sidebar(dl)
    render_main(dl)
