"""
Shared Streamlit UI components for the Knowledge QA system.
Provides dark theme CSS, session initialization, sidebar & main area rendering.

Each entry point defines a ``DataLayer`` dict, then calls
``render_css()``, ``render_sidebar(dl)``, and ``render_main(dl)``.
"""
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════
# Premium Dark Theme CSS
# ═══════════════════════════════════════════════════════════════════════

DARK_THEME_CSS = """
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

    .stApp { background: #0a0e17 !important; }
    html, body, [data-testid="stAppViewContainer"] { background: #0a0e17; color: #e8edf5; }
    * { font-family: 'Inter', 'Noto Sans SC', sans-serif; }

    /* ── Aggressively hide Streamlit's dim overlay and status widget ── */
    .stSpinner, [data-testid="stSpinner"],
    [data-testid="stSpinner"] > div:first-child,
    iframe[title="streamlit_spinner"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    .st-emotion-cache-1dp5vir, .st-emotion-cache-ocsh0s,
    div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }

    /* ===== Sidebar ===== */
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

    /* ===== Chat bubbles ===== */
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

    /* ===== Source references ===== */
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

    /* ===== Stat tiles ===== */
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

    /* ===== Chat input ===== */
    [data-testid="stChatInput"] {
        background: linear-gradient(180deg, transparent 0%, #0a0e17 40%);
        padding: 0.8rem 1.5rem 1.2rem;
    }
    [data-testid="stChatInput"] textarea {
        background: #161c2a !important; border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e8edf5 !important; border-radius: 14px !important;
        padding: 0.7rem 1rem !important; caret-color: #c9a84c !important;
        font-size: 0.93rem !important; resize: none !important; min-height: 44px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(201,168,76,0.4) !important;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.12) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }

    /* ===== Typing indicator (bouncing dots) ===== */
    .thinking-bubble {
        display: flex; align-items: center; gap: 5px; padding: 1.2rem 1.5rem !important;
        min-height: 54px;
    }
    .typing-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--accent);
        animation: dotPulse 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.18s; }
    .typing-dot:nth-child(3) { animation-delay: 0.36s; }
    @keyframes dotPulse {
        0%, 60%, 100% { transform: scale(0.6); opacity: 0.3; }
        30% { transform: scale(1); opacity: 1; }
    }

    /* ===== Misc ===== */
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

    .chat-msg .bubble code {
        background: rgba(255,255,255,0.06); padding: 0.15rem 0.4rem;
        border-radius: 4px; font-size: 0.88em; color: var(--accent);
    }
    .chat-msg .bubble p { margin-bottom: 0.6em; }
    .chat-msg .bubble p:last-child { margin-bottom: 0; }
</style>
"""

THINKING_HTML = """<div class="chat-msg assistant">
    <div class="avatar">✦</div>
    <div class="body">
        <div class="author">AI Assistant</div>
        <div class="bubble thinking-bubble">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    </div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "conv_id": None,
        "messages": [],
        "convs": [],
        "uploaded_files": [],
        "_pending_text": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_css():
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Conversation helpers
# ═══════════════════════════════════════════════════════════════════════

def _ensure_conversation(dl):
    convs = dl["list_convs"]()
    st.session_state.convs = convs
    if st.session_state.conv_id is None:
        conv = dl["create_conv"]("新对话")
        if conv:
            st.session_state.conv_id = conv["id"]
            st.session_state.messages = []


def _switch_conversation(dl, conv_id: str):
    st.session_state.conv_id = conv_id
    conv = dl["get_conv"](conv_id)
    st.session_state.messages = conv.get("messages", [])


def _create_new_conversation(dl):
    conv = dl["create_conv"]("新对话")
    if conv:
        st.session_state.conv_id = conv["id"]
        st.session_state.messages = []
        st.session_state.convs = dl["list_convs"]()


def _persist(dl, user_text: str, answer: str, sources: list):
    conv_id = st.session_state.conv_id
    if not conv_id:
        return
    dl["add_msg"](conv_id, "user", user_text)
    dl["add_msg"](conv_id, "assistant", answer, sources)
    st.session_state.convs = dl["list_convs"]()


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════

def render_sidebar(dl: dict):
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

        health = dl["health"]()
        online = health.get("status") == "ok"
        if online:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""<div class="stat-tile">
                        <div class="stat-value">{health.get('index_size', 0)}</div>
                        <div class="stat-label">文档块</div></div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                ready = health.get("index_ready")
                st.markdown(
                    f"""<div class="stat-tile {'online' if ready else ''}">
                        <div class="stat-value">{'●' if ready else '○'}</div>
                        <div class="stat-label">{'就绪' if ready else '空库'}</div></div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.error(f"系统初始化中... {health.get('detail', '')}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown(
            """<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.4rem;">
                <span style="font-weight:600;color:#e8edf5;font-size:0.85rem;">历史对话</span></div>""",
            unsafe_allow_html=True,
        )
        if st.button("+ 新建对话", use_container_width=True):
            _create_new_conversation(dl)
            st.rerun()

        convs = st.session_state.convs
        if not convs:
            st.session_state.convs = dl["list_convs"]()
            convs = st.session_state.convs
        if convs:
            for c in convs[:20]:
                active = c["id"] == st.session_state.conv_id
                label = f"{c['title'][:30]}"
                if active:
                    label = f"✦ {label}"
                c1, c2 = st.columns([8, 1])
                with c1:
                    if st.button(label, key=f"conv_{c['id']}", use_container_width=True,
                                 type="primary" if active else "secondary"):
                        _switch_conversation(dl, c["id"])
                        st.rerun()
                with c2:
                    if st.button("✕", key=f"delconv_{c['id']}", help=f"删除「{c['title']}」",
                                 use_container_width=True):
                        dl["delete_conv"](c["id"])
                        if active:
                            st.session_state.conv_id = None
                            st.session_state.messages = []
                        st.session_state.convs = dl["list_convs"]()
                        st.rerun()
        else:
            st.caption("—")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            """<div style="font-weight:600;color:#e8edf5;font-size:0.85rem;margin-bottom:0.3rem;">上传文档</div>""",
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
                    result = dl["upload_doc"](uploaded_file)
                if "error" in result:
                    st.error(f"上传失败: {result['error']}")
                else:
                    st.success(result.get("message", "OK"))
                    dl.get("on_upload_ok", lambda: None)()
                    st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            """<div style="font-weight:600;color:#e8edf5;font-size:0.85rem;margin-bottom:0.3rem;">已上传文档</div>""",
            unsafe_allow_html=True,
        )
        docs = dl["list_docs"]()
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
                            <span style="opacity:0.4;">▸</span> {doc[:32]}{'…' if len(doc)>32 else ''}</div>""",
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("✕", key=f"del_{doc}", help=f"删除 {doc}"):
                        dl["delete_doc"](doc)
                        dl.get("on_doc_delete_ok", lambda: None)()
                        st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        top_k = st.slider("检索数量", min_value=1, max_value=10, value=4)
        st.session_state.top_k = top_k


# ═══════════════════════════════════════════════════════════════════════
# Bubble rendering helpers
# ═══════════════════════════════════════════════════════════════════════

def _render_user(msg: dict):
    st.markdown(
        f"""<div class="chat-msg user">
            <div class="avatar">◈</div>
            <div class="body">
                <div class="author">You</div>
                <div class="bubble">{msg['content']}</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_assistant(msg: dict):
    sources = msg.get("sources", [])
    sources_html = ""
    if sources:
        items = ""
        for i, src in enumerate(sources, 1):
            pg = src.get("page")
            pg_info = f" · p.{pg + 1}" if pg is not None and isinstance(pg, int) else ""
            items += (
                f"<div class='source-item'>"
                f"<span class='source-badge'>{i}</span>"
                f"<span class='source-name'>{src['source']}{pg_info}</span>"
                f"<br><span style='color:#64748b;'>{src['content'][:100]}</span>"
                f"</div>"
            )
        sources_html = (
            f"<div class='source-card'>"
            f"<div class='source-header'>References</div>{items}</div>"
        )
    st.markdown(
        f"""<div class="chat-msg assistant">
            <div class="avatar">✦</div>
            <div class="body">
                <div class="author">AI Assistant</div>
                <div class="bubble">{msg['content']}</div>
                {sources_html}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Main chat area
# ═══════════════════════════════════════════════════════════════════════

def render_main(dl: dict):
    """Main area: header, welcome, message history, chat input.

    Flow when user sends a message:
      1. Append user msg → render user bubble → rerun
      2. (re-run) Show history + user bubble, then render thinking dots,
         execute query (blocking), replace dots with answer, rerun
      3. (re-run) Show full history — conversation complete.

    No st.spinner used in the main flow → no dark overlay.
    st.chat_input auto-clears → no input residual.
    """
    # ── Header ──
    st.markdown(
        """<div class="main-header">
            <h1><span class="accent-dot"></span>智能知识库问答</h1>
            <p class="subtitle">基于文档内容的 AI 问答 · 对话自动保存</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Welcome (only when no messages) ──
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

    # ═════════════════════════════════════════════════════════════
    # Phase 2 — execute pending query (thinking → answer)
    # ═════════════════════════════════════════════════════════════
    pending = st.session_state.get("_pending_text", "")
    if pending:
        # Show all messages + user's latest message
        for m in st.session_state.messages:
            if m["role"] == "user":
                _render_user(m)
            else:
                _render_assistant(m)

        # ── Thinking animation (no spinner! no dim!) ──
        st.markdown(THINKING_HTML, unsafe_allow_html=True)
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # ── Execute LLM query (BLOCKING — but no dim overlay!) ──
        top_k = st.session_state.get("top_k", 4)
        result = dl["query"](pending, top_k)

        if "error" in result:
            answer = f"查询失败: {result['error']}"
            sources_list = []
        else:
            answer = result.get("answer", "")
            sources_list = result.get("sources", [])

        # Store answer
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources_list,
        })
        _persist(dl, pending, answer, sources_list)
        st.session_state._pending_text = ""
        st.rerun()  # → next run shows full history in normal idle state

    # ═════════════════════════════════════════════════════════════
    # Normal render — show message history
    # ═════════════════════════════════════════════════════════════
    for m in st.session_state.messages:
        if m["role"] == "user":
            _render_user(m)
        else:
            _render_assistant(m)

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # Chat input
    # ═════════════════════════════════════════════════════════════
    prompt = st.chat_input(
        placeholder="输入你的问题，AI 将基于知识库文档回答…",
    )

    if prompt and prompt.strip():
        _ensure_conversation(dl)
        user_text = prompt.strip()

        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state._pending_text = user_text

        # Set flag so the next run handles the query
        st.rerun()
