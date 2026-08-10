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

# ── 轻量级导入（不加载模型）──
from config import config
from app.document_loader import process_file, get_all_uploaded_files


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
# 页面设置
# ═══════════════════════════════════════

st.set_page_config(
    page_title="Knowledge QA",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════
# Premium Dark Theme CSS
# ═══════════════════════════════════════
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #161c2a;
        --bg-hover: #1e293b;
        --border-subtle: rgba(255,255,255,0.06);
        --border-medium: rgba(255,255,255,0.10);
        --text-primary: #e8edf5;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent: #c9a84c;
        --accent-soft: rgba(201,168,76,0.12);
        --accent-glow: rgba(201,168,76,0.25);
        --success: #34d399;
        --danger: #f87171;
        --info: #60a5fa;
    }

    .stApp { background: var(--bg-primary); }
    * { font-family: 'Inter', 'Noto Sans SC', sans-serif; }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1119 0%, #0f1724 100%);
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important; font-weight: 600; letter-spacing: 0.02em;
    }
    [data-testid="stSidebar"] hr { border-color: var(--border-subtle); margin: 1.2rem 0; }

    [data-testid="stSidebar"] .stButton > button {
        background: var(--bg-card); border: 1px solid var(--border-medium);
        color: var(--text-secondary); border-radius: 8px; font-size: 0.88rem;
        font-weight: 500; padding: 0.45rem 0.8rem; transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-hover); border-color: var(--accent); color: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--accent-soft); border-color: rgba(201,168,76,0.35);
        color: var(--accent) !important; font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: rgba(201,168,76,0.22); border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
    }
    [data-testid="stSidebar"] .stButton > button[help] {
        font-size: 0.72rem; padding: 0.35rem 0.3rem; border-radius: 6px;
        background: transparent; border: 1px solid transparent; color: var(--text-muted); min-width: 28px;
    }
    [data-testid="stSidebar"] .stButton > button[help]:hover {
        background: rgba(248,113,113,0.1); border-color: rgba(248,113,113,0.3); color: var(--danger);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        border: 1px dashed var(--border-medium); border-radius: 10px;
        background: transparent; transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover section {
        border-color: var(--accent); background: var(--accent-soft);
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBar"] {
        background: var(--border-subtle);
    }

    /* ===== 主区域 ===== */
    .main-header { padding: 1.5rem 0 0.5rem; }
    .main-header h1 {
        font-size: 1.8rem; font-weight: 700; color: var(--text-primary);
        letter-spacing: -0.01em; margin-bottom: 0.3rem;
    }
    .main-header .subtitle { color: var(--text-muted); font-size: 0.92rem; font-weight: 400; }
    .accent-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent); margin-right: 6px; box-shadow: 0 0 8px var(--accent-glow);
    }

    /* ===== 欢迎页 ===== */
    .welcome-container {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; padding: 5rem 2rem; text-align: center;
    }
    .welcome-icon {
        width: 88px; height: 88px; border-radius: 24px;
        background: linear-gradient(135deg, rgba(201,168,76,0.15) 0%, rgba(201,168,76,0.05) 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 2.4rem; margin-bottom: 2rem;
        border: 1px solid rgba(201,168,76,0.2); box-shadow: 0 0 40px rgba(201,168,76,0.08);
    }
    .welcome-title {
        font-size: 1.6rem; font-weight: 700; color: var(--text-primary);
        margin-bottom: 0.6rem; letter-spacing: -0.01em;
    }
    .welcome-subtitle {
        color: var(--text-muted); font-size: 0.95rem; max-width: 460px; line-height: 1.7;
    }
    .welcome-hints {
        margin-top: 2.5rem; display: flex; gap: 0.8rem; flex-wrap: wrap; justify-content: center;
    }
    .welcome-hint {
        background: var(--bg-card); border: 1px solid var(--border-medium);
        border-radius: 12px; padding: 0.7rem 1.2rem; color: var(--text-secondary);
        font-size: 0.85rem; transition: all 0.2s ease; cursor: default;
    }
    .welcome-hint:hover {
        border-color: rgba(201,168,76,0.35); color: var(--text-primary);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* ===== 聊天气泡 ===== */
    .chat-msg {
        display: flex; gap: 0.8rem; margin-bottom: 1.2rem; animation: fadeInUp 0.35s ease;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-msg .avatar {
        width: 38px; height: 38px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.9rem; flex-shrink: 0;
    }
    .chat-msg.user .avatar {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 100%);
        border: 1px solid rgba(96,165,250,0.3);
    }
    .chat-msg.assistant .avatar {
        background: linear-gradient(135deg, rgba(201,168,76,0.15) 0%, rgba(201,168,76,0.08) 100%);
        border: 1px solid rgba(201,168,76,0.25);
    }
    .chat-msg .body { flex: 1; min-width: 0; }
    .chat-msg .author {
        font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 0.35rem;
    }
    .chat-msg.user .author { color: var(--info); }
    .chat-msg.assistant .author { color: var(--accent); }
    .chat-msg .bubble {
        background: var(--bg-card); border: 1px solid var(--border-medium);
        border-radius: 14px; padding: 1rem 1.2rem; color: var(--text-primary);
        line-height: 1.75; font-size: 0.93rem; word-break: break-word;
    }
    .chat-msg.user .bubble {
        background: rgba(96,165,250,0.06); border-color: rgba(96,165,250,0.15);
        border-radius: 14px 14px 4px 14px;
    }
    .chat-msg.assistant .bubble {
        border-color: rgba(201,168,76,0.12); border-radius: 14px 14px 14px 4px;
    }

    /* ===== 来源引用 ===== */
    .source-card {
        margin-top: 0.6rem; background: rgba(201,168,76,0.04);
        border: 1px solid rgba(201,168,76,0.1); border-radius: 10px; padding: 0.8rem 1rem;
    }
    .source-card .source-header {
        font-size: 0.75rem; font-weight: 600; color: var(--accent);
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;
    }
    .source-card .source-item {
        font-size: 0.82rem; color: var(--text-muted);
        padding: 0.3rem 0; border-bottom: 1px solid var(--border-subtle); line-height: 1.5;
    }
    .source-card .source-item:last-child { border-bottom: none; }
    .source-card .source-badge {
        display: inline-block; background: var(--accent-soft); color: var(--accent);
        font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 4px;
        font-weight: 600; margin-right: 0.4rem;
    }
    .source-card .source-name { color: var(--text-secondary); font-weight: 500; }

    /* ===== 状态磁贴 ===== */
    .stat-tile {
        background: var(--bg-card); border: 1px solid var(--border-medium);
        border-radius: 12px; padding: 1rem 1.1rem; text-align: center; transition: all 0.2s ease;
    }
    .stat-tile:hover { border-color: rgba(201,168,76,0.25); }
    .stat-tile .stat-value {
        font-size: 1.8rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em;
    }
    .stat-tile .stat-label {
        font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;
        text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;
    }
    .stat-tile.online .stat-value { color: var(--success); }
    .stat-tile.online {
        border-color: rgba(52,211,153,0.15); background: rgba(52,211,153,0.04);
    }

    /* ===== 输入区域 ===== */
    div[data-testid="stTextInput"] input {
        background: transparent !important; border: none !important;
        color: var(--text-primary) !important; font-size: 0.95rem !important;
        padding: 0.6rem 0.4rem !important; outline: none !important; box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; }
    div[data-testid="stTextInput"] label { display: none !important; }
    [data-testid="stTextInput"] > div { border: none !important; background: transparent !important; }
    [data-testid="stTextInput"] > div:hover,
    [data-testid="stTextInput"] > div:focus-within { border: none !important; box-shadow: none !important; }

    .send-btn button {
        background: var(--accent) !important; color: #0a0e17 !important;
        border: none !important; border-radius: 10px !important; font-weight: 700 !important;
        font-size: 0.9rem !important; padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important; letter-spacing: 0.02em !important;
    }
    .send-btn button:hover {
        background: #d4b55a !important; box-shadow: 0 4px 24px rgba(201,168,76,0.35) !important;
        transform: translateY(-1px);
    }

    /* ===== 文档管理列表项 ===== */
    .doc-item {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.4rem 0.5rem; border-radius: 6px; font-size: 0.82rem; color: var(--text-secondary);
    }
    .doc-item:hover { background: var(--bg-hover); }

    .divider { height: 1px; background: var(--border-subtle); margin: 0.6rem 0; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }

    [data-testid="stAlert"] {
        background: var(--bg-card); border: 1px solid var(--border-medium);
        color: var(--text-primary); border-radius: 10px;
    }
    [data-testid="stAlert"][kind="success"] {
        border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.06);
    }
    [data-testid="stAlert"][kind="error"] {
        border-color: rgba(248,113,113,0.3); background: rgba(248,113,113,0.06);
    }

    div[data-testid="column"]:not([data-testid="stSidebar"] *) .stButton > button {
        background: var(--accent-soft); border: 1px solid rgba(201,168,76,0.3); color: var(--accent);
    }

    .chat-msg .bubble code {
        background: rgba(255,255,255,0.06); padding: 0.15rem 0.4rem;
        border-radius: 4px; font-size: 0.88em; color: var(--accent);
    }
    .chat-msg .bubble p { margin-bottom: 0.6em; }
    .chat-msg .bubble p:last-child { margin-bottom: 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════

def init_session():
    defaults = {
        "conv_id": None,
        "messages": [],
        "convs": [],
        "uploaded_files": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ═══════════════════════════════════════
# 业务函数（直接调用后端模块，无 HTTP）
# ═══════════════════════════════════════

def check_health() -> dict:
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


def upload_file(uploaded_file) -> dict:
    """保存上传文件并加入向量索引"""
    save_dir = config.DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)

    # 保存文件
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # 解析文档
    chunks = process_file(file_path)
    if not chunks:
        os.remove(file_path)
        return {"error": "未能从文件中提取任何文本内容"}

    # 加入向量索引（增量）
    rag = get_rag_chain()
    try:
        total = rag.add_documents(chunks)
        return {"message": f"上传成功: {uploaded_file.name}，新增 {len(chunks)} 个文本块，索引共 {total} 条", "chunks": len(chunks), "total": total}
    except Exception as e:
        return {"error": f"索引构建失败: {str(e)}"}


def query_rag(question: str, top_k: int = 4) -> dict:
    """执行 RAG 问答"""
    rag = get_rag_chain()
    try:
        answer, sources = rag.query(question, top_k)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}


def list_documents() -> list:
    """列出已上传文档"""
    return get_all_uploaded_files()


def delete_document(filename: str) -> bool:
    """删除文档（从磁盘移除；FAISS 索引中的旧向量需重建才能彻底清除）"""
    file_path = os.path.join(config.DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def list_conversations() -> list:
    """列出所有对话"""
    store = get_conversation_store()
    return store.list_all()


def create_conversation(title: str = "") -> dict:
    """新建对话"""
    store = get_conversation_store()
    return store.create(title)


def get_conversation(conv_id: str) -> dict:
    """获取单个对话详情"""
    store = get_conversation_store()
    return store.get(conv_id) or {}


def add_message_to_conv(conv_id: str, role: str, content: str, sources: list = None):
    """向对话追加一条消息"""
    store = get_conversation_store()
    store.add_message(conv_id, role, content, sources)


def delete_conversation(conv_id: str) -> bool:
    """删除对话"""
    store = get_conversation_store()
    return store.delete(conv_id)


# ═══════════════════════════════════════
# 对话管理逻辑
# ═══════════════════════════════════════

def ensure_conversation():
    """确保存在当前对话"""
    convs = list_conversations()
    st.session_state.convs = convs

    if st.session_state.conv_id is None:
        conv = create_conversation("新对话")
        if conv:
            st.session_state.conv_id = conv["id"]
            st.session_state.messages = []
            st.session_state.convs = list_conversations()


def load_messages(conv_id: str):
    """加载指定对话的消息"""
    conv = get_conversation(conv_id)
    st.session_state.messages = conv.get("messages", [])


def switch_conversation(conv_id: str):
    """切换到指定对话"""
    st.session_state.conv_id = conv_id
    load_messages(conv_id)


def create_new_conversation():
    """创建新对话并切换"""
    conv = create_conversation("新对话")
    if conv:
        st.session_state.conv_id = conv["id"]
        st.session_state.messages = []
        st.session_state.convs = list_conversations()


def save_message_pair(user_text: str, answer: str, sources: list):
    """保存一对用户/助手消息"""
    conv_id = st.session_state.conv_id
    if not conv_id:
        return
    add_message_to_conv(conv_id, "user", user_text)
    add_message_to_conv(conv_id, "assistant", answer, sources)
    st.session_state.convs = list_conversations()


# ═══════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """<div style="display:flex;align-items:center;gap:10px;padding:0.3rem 0 0.8rem;">
                <div style="width:36px;height:36px;border-radius:10px;
                    background:linear-gradient(135deg,#c9a84c 0%,#a68a3a 100%);
                    display:flex;align-items:center;justify-content:center;font-size:1rem;">✦</div>
                <div><div style="font-weight:700;color:#e8edf5;font-size:1rem;line-height:1.2;">Knowledge QA</div>
                <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.04em;">RAG &middot; DEEPSEEK</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── 系统状态 ──
        health = check_health()
        online = health.get("status") == "ok"

        if online:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""<div class="stat-tile">
                        <div class="stat-value">{health.get('index_size', 0)}</div>
                        <div class="stat-label">文档块</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                ready = health.get("index_ready")
                st.markdown(
                    f"""<div class="stat-tile {'online' if ready else ''}">
                        <div class="stat-value">{'●' if ready else '○'}</div>
                        <div class="stat-label">{'就绪' if ready else '空库'}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.error(f"系统初始化中... {health.get('detail', '')}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── 对话列表 ──
        st.markdown(
            """<div style="display:flex;align-items:center;justify-content:space-between;
                margin-bottom:0.4rem;">
                <span style="font-weight:600;color:#e8edf5;font-size:0.85rem;">历史对话</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if st.button("+ 新建对话", use_container_width=True):
            create_new_conversation()
            st.rerun()

        convs = st.session_state.convs
        if not convs:
            st.session_state.convs = list_conversations()
            convs = st.session_state.convs

        if convs:
            for c in convs[:20]:
                active = c["id"] == st.session_state.conv_id
                label = f"{c['title'][:30]}"
                if active:
                    label = f"✦ {label}"

                c1, c2 = st.columns([8, 1])
                with c1:
                    if st.button(
                        label,
                        key=f"conv_{c['id']}",
                        use_container_width=True,
                        type="primary" if active else "secondary",
                    ):
                        switch_conversation(c["id"])
                        st.rerun()
                with c2:
                    if st.button(
                        "✕",
                        key=f"delconv_{c['id']}",
                        help=f"删除「{c['title']}」",
                        use_container_width=True,
                    ):
                        delete_conversation(c["id"])
                        if active:
                            st.session_state.conv_id = None
                            st.session_state.messages = []
                        st.session_state.convs = list_conversations()
                        st.rerun()
        else:
            st.caption("—")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── 文档上传 ──
        st.markdown(
            """<div style="font-weight:600;color:#e8edf5;font-size:0.85rem;
                margin-bottom:0.3rem;">上传文档</div>""",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "PDF / TXT / MD / Excel / CSV",
            type=["pdf", "txt", "md", "markdown", "csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.button("开始上传", use_container_width=True, type="primary"):
                with st.spinner(f"正在解析 {uploaded_file.name} ..."):
                    result = upload_file(uploaded_file)
                if "error" in result:
                    st.error(f"上传失败: {result['error']}")
                else:
                    st.success(result.get("message", "OK"))
                    st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── 文档管理 ──
        st.markdown(
            """<div style="font-weight:600;color:#e8edf5;font-size:0.85rem;
                margin-bottom:0.3rem;">已上传文档</div>""",
            unsafe_allow_html=True,
        )

        docs = list_documents()
        st.session_state.uploaded_files = docs

        if not docs:
            st.caption("暂无文档")
        else:
            st.caption(f"{len(docs)} 个文件")
            for doc in docs[:15]:
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f"""<div class="doc-item" title="{doc}">
                            <span style="opacity:0.4;">▸</span> {doc[:32]}{'…' if len(doc)>32 else ''}
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("✕", key=f"del_{doc}", help=f"删除 {doc}"):
                        delete_document(doc)
                        st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── 设置 ──
        top_k = st.slider("检索数量", min_value=1, max_value=10, value=4)
        st.session_state.top_k = top_k


# ═══════════════════════════════════════
# 主区域
# ═══════════════════════════════════════

def render_main():
    st.markdown(
        """<div class="main-header">
            <h1><span class="accent-dot"></span>智能知识库问答</h1>
            <p class="subtitle">基于文档内容的 AI 问答 · 对话自动保存</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── 空状态欢迎页 ──
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-icon">✦</div>
                <div class="welcome-title">开始一段新的对话</div>
                <div class="welcome-subtitle">
                    上传 PDF、Excel 或任何文档后，向 AI 提问，
                    它将基于你的文档内容给出准确答案
                </div>
                <div class="welcome-hints">
                    <div class="welcome-hint">总结这份文档的主要内容</div>
                    <div class="welcome-hint">表格里利润率最高的是什么</div>
                    <div class="welcome-hint">列出关键数据</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 消息列表 ──
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(
                f"""<div class="chat-msg user">
                    <div class="avatar">◈</div>
                    <div class="body">
                        <div class="author">You</div>
                        <div class="bubble">{content}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            sources_html = ""
            if sources:
                items = ""
                for i, src in enumerate(sources, 1):
                    page = src.get("page")
                    page_info = f" · p.{page+1}" if page is not None else ""
                    items += (
                        f"<div class='source-item'>"
                        f"<span class='source-badge'>{i}</span>"
                        f"<span class='source-name'>{src['source']}{page_info}</span>"
                        f"<br><span style='color:#64748b;'>{src['content'][:100]}</span>"
                        f"</div>"
                    )
                sources_html = (
                    f"<div class='source-card'>"
                    f"<div class='source-header'>References</div>"
                    f"{items}"
                    f"</div>"
                )

            st.markdown(
                f"""<div class="chat-msg assistant">
                    <div class="avatar">✦</div>
                    <div class="body">
                        <div class="author">AI Assistant</div>
                        <div class="bubble">{content}</div>
                        {sources_html}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="divider" style="margin:1.5rem 0;"></div>', unsafe_allow_html=True)

    # ── 输入区域 ──
    col_input, col_btn = st.columns([7, 1])
    with col_input:
        user_input = st.text_input(
            "question_input",
            placeholder="输入你的问题，AI 将基于知识库文档回答…",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("发送", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if send and user_input.strip():
        ensure_conversation()

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner(""):
            top_k = st.session_state.get("top_k", 4)
            result = query_rag(user_input.strip(), top_k)

        if "error" in result:
            answer = f"查询失败: {result['error']}"
            sources_list = []
        else:
            answer = result.get("answer", "")
            sources_list = result.get("sources", [])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_list,
        })

        save_message_pair(user_input.strip(), answer, sources_list)
        st.rerun()


# ═══════════════════════════════════════
# 运行
# ═══════════════════════════════════════

if __name__ == "__main__":
    ensure_conversation()
    render_sidebar()
    render_main()
