"""
FastAPI 后端服务
提供: 文档上传 / 问答查询 / 文档列表 / 知识库重建 等 REST API
"""
import os
import shutil
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import config
from app.models import QueryRequest, QueryResponse, SourceDocument, UploadResponse, ErrorResponse
from app.document_loader import process_file, get_all_uploaded_files
from app.rag_chain import rag_chain
from app.conversation_store import conversation_store
from pydantic import BaseModel

# ── 创建 FastAPI 应用 ──
app = FastAPI(
    title="智能知识库问答系统",
    description="基于 FastAPI + RAG 的智能知识库问答系统 API",
    version="1.0.0",
)

# ── CORS 中间件（允许 Streamlit 前端跨域）──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 健康检查 ──
@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "ok",
        "index_ready": rag_chain.is_ready,
        "index_size": rag_chain.index_size,
    }


# ── 文档上传 ──
@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="上传文档",
    description="上传 PDF / TXT / Markdown 文件，自动解析并构建向量索引",
)
async def upload_document(file: UploadFile = File(...)):
    # 校验文件类型
    allowed_exts = (".pdf", ".txt", ".md", ".markdown", ".csv", ".xlsx", ".xls")
    ext = os.path.splitext(file.filename or "unknown")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {list(allowed_exts)}",
        )

    # 保存文件
    os.makedirs(config.DATA_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(config.DATA_DIR, safe_name)

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 解析 & 切分 & 增量添加到索引
    try:
        chunks = process_file(file_path)
        if not chunks:
            raise ValueError("文档解析后无有效内容")

        # 增量添加，不再全量重建
        total = rag_chain.add_documents(chunks)

        return UploadResponse(
            filename=file.filename or "unknown",
            chunks=len(chunks),
            message=f"上传成功！知识库共 {total} 个文档块",
        )
    except Exception as e:
        # 回滚：删除已保存文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


# ── RAG 问答 ──
@app.post("/query", response_model=QueryResponse, summary="知识库问答")
async def query_knowledge_base(request: QueryRequest):
    if not rag_chain.is_ready:
        raise HTTPException(
            status_code=400,
            detail="知识库尚未构建，请先上传文档",
        )

    try:
        answer, raw_sources = rag_chain.query(
            question=request.question,
            top_k=request.top_k,
        )

        sources = [
            SourceDocument(
                content=s["content"],
                source=s["source"],
                page=s["page"],
            )
            for s in raw_sources
        ]

        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ── 获取已上传文档列表 ──
@app.get("/documents", summary="已上传文档列表")
async def list_documents():
    files = get_all_uploaded_files()
    return {"count": len(files), "files": files}


# ── 删除文档 ──
@app.delete("/documents/{filename}", summary="删除指定文档")
async def delete_document(filename: str):
    file_path = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    os.remove(file_path)

    # 重建索引
    all_chunks = _collect_all_chunks()
    if all_chunks:
        rag_chain.build_index(all_chunks)
    else:
        _clear_index()

    return {"message": f"已删除 {filename}，知识库已更新"}


# ── 清空知识库 ──
@app.delete("/documents", summary="清空所有文档")
async def clear_all_documents():
    _clear_index()

    if os.path.exists(config.DATA_DIR):
        shutil.rmtree(config.DATA_DIR)
        os.makedirs(config.DATA_DIR, exist_ok=True)

    return {"message": "知识库已清空"}


# ── 对话历史管理 ──

class CreateConversationRequest(BaseModel):
    title: str = ""


class AddMessageRequest(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str
    sources: list = []


@app.get("/conversations", summary="获取所有对话列表")
async def list_conversations():
    """返回所有历史对话摘要，按更新时间倒序"""
    return {"conversations": conversation_store.list_all()}


@app.post("/conversations", summary="创建新对话")
async def create_conversation(req: CreateConversationRequest = None):
    """创建一个新的空对话"""
    title = req.title if req else ""
    conv = conversation_store.create(title)
    return conv


@app.get("/conversations/{conv_id}", summary="获取对话详情")
async def get_conversation(conv_id: str):
    """获取某个对话的完整消息记录"""
    conv = conversation_store.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conv


@app.post("/conversations/{conv_id}/messages", summary="追加消息")
async def add_message(conv_id: str, req: AddMessageRequest):
    """给指定对话添加一条消息"""
    ok = conversation_store.add_message(
        conv_id, req.role, req.content, req.sources
    )
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "ok"}


@app.put("/conversations/{conv_id}/title", summary="修改对话标题")
async def update_conversation_title(conv_id: str, title: str):
    """修改对话标题"""
    ok = conversation_store.update_title(conv_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "ok"}


@app.delete("/conversations/{conv_id}", summary="删除对话")
async def delete_conversation(conv_id: str):
    """删除指定对话"""
    ok = conversation_store.delete(conv_id)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "ok"}


# ── 辅助函数 ──
def _collect_all_chunks() -> list:
    """收集所有文档并切分"""
    from app.document_loader import process_file

    all_chunks = []
    data_dir = config.DATA_DIR
    if not os.path.exists(data_dir):
        return all_chunks

    supported_exts = (".pdf", ".txt", ".md", ".markdown", ".csv", ".xlsx", ".xls")
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(supported_exts):
            file_path = os.path.join(data_dir, filename)
            try:
                chunks = process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    return all_chunks


def _clear_index():
    """清除向量索引"""
    import shutil

    if os.path.exists(config.VECTOR_STORE_DIR):
        shutil.rmtree(config.VECTOR_STORE_DIR)
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)

    rag_chain.vector_store = None
