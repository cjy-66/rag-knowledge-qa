"""
FastAPI 后端服务
提供: 文档上传 / 问答查询 / 文档列表 / 知识库重建 等 REST API
多用户: 通过 X-User-ID 请求头区分用户，每人独立数据和索引
"""
import os
import shutil
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from config import config
from app.models import QueryRequest, QueryResponse, SourceDocument, UploadResponse, ErrorResponse
from app.document_loader import process_file, get_all_uploaded_files, cleanup_expired_files
from app.rag_chain import rag_chain
from app.conversation_store import conversation_store
from pydantic import BaseModel

app = FastAPI(
    title="智能知识库问答系统",
    description="基于 FastAPI + RAG 的智能知识库问答系统 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials="*" not in config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 工具函数 ──
def _user_data_dir(user_id: str) -> str:
    d = os.path.join(config.DATA_DIR, user_id)
    os.makedirs(d, exist_ok=True)
    return d


# ═══════════════════════════════════════════
# 健康检查（全局，无需 user_id）
# ═══════════════════════════════════════════

@app.get("/health", summary="健康检查")
async def health_check(x_user_id: str = Header(None, alias="X-User-ID")):
    uid = x_user_id or ""
    return {
        "status": "ok",
        "index_ready": rag_chain.is_ready_for(uid) if uid else False,
        "index_size": rag_chain.index_size_for(uid) if uid else 0,
    }


# ═══════════════════════════════════════════
# 文档上传
# ═══════════════════════════════════════════

@app.post("/upload", summary="上传文档")
async def upload_document(
    file: UploadFile = File(...),
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    allowed_exts = (".pdf", ".txt", ".md", ".markdown", ".csv", ".xlsx", ".xls")
    ext = os.path.splitext(file.filename or "unknown")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # 1) 先清理该用户的过期文件
    clean_result = cleanup_expired_files(x_user_id)
    if clean_result["deleted"]:
        # 有文件被删 → 用剩余文件重建索引
        all_chunks = _collect_user_chunks(x_user_id)
        if all_chunks:
            rag_chain.build_index(x_user_id, all_chunks)
        else:
            rag_chain.clear_index(x_user_id)

    # 2) 保存新文件
    user_dir = _user_data_dir(x_user_id)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(user_dir, safe_name)

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    try:
        chunks = process_file(file_path)
        if not chunks:
            raise ValueError("文档解析后无有效内容")
        total = rag_chain.add_documents(x_user_id, chunks)
        return {
            "filename": file.filename or "unknown",
            "chunks": len(chunks),
            "message": f"上传成功！知识库共 {total} 个文档块",
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


# ═══════════════════════════════════════════
# RAG 问答（非流式）
# ═══════════════════════════════════════════

@app.post("/query", summary="知识库问答")
async def query_knowledge_base(
    request: QueryRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    if not rag_chain.is_ready_for(x_user_id):
        raise HTTPException(status_code=400, detail="知识库尚未构建，请先上传文档")

    try:
        answer, raw_sources = rag_chain.query(
            user_id=x_user_id,
            question=request.question,
            top_k=request.top_k,
        )
        sources = [
            SourceDocument(content=s["content"], source=s["source"], page=s["page"])
            for s in raw_sources
        ]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ═══════════════════════════════════════════
# RAG 流式问答（SSE）
# ═══════════════════════════════════════════

@app.post("/query/stream", summary="知识库问答（流式）")
async def query_knowledge_base_stream(
    request: QueryRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    if not rag_chain.is_ready_for(x_user_id):
        raise HTTPException(status_code=400, detail="知识库尚未构建，请先上传文档")

    token_gen, sources = rag_chain.query_stream(
        user_id=x_user_id,
        question=request.question,
        top_k=request.top_k,
    )

    def _event_stream():
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        for token in token_gen:
            yield json.dumps({"type": "token", "data": token}) + "\n"
        yield json.dumps({"type": "done", "data": ""}) + "\n"

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


# ═══════════════════════════════════════════
# 文档管理
# ═══════════════════════════════════════════

@app.get("/documents", summary="已上传文档列表")
async def list_documents(x_user_id: str = Header(..., alias="X-User-ID")):
    files = get_all_uploaded_files(x_user_id)
    return {"count": len(files), "files": files}


@app.delete("/documents/{filename}", summary="删除指定文档")
async def delete_document(
    filename: str,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    file_path = os.path.join(_user_data_dir(x_user_id), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    os.remove(file_path)

    # 重建该用户的索引
    all_chunks = _collect_user_chunks(x_user_id)
    if all_chunks:
        rag_chain.build_index(x_user_id, all_chunks)
    else:
        _clear_user_index(x_user_id)

    return {"message": f"已删除 {filename}，知识库已更新"}


@app.delete("/documents", summary="清空所有文档")
async def clear_all_documents(x_user_id: str = Header(..., alias="X-User-ID")):
    _clear_user_index(x_user_id)

    user_dir = _user_data_dir(x_user_id)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        os.makedirs(user_dir, exist_ok=True)

    return {"message": "知识库已清空"}


# ═══════════════════════════════════════════
# 对话历史管理
# ═══════════════════════════════════════════

class CreateConversationRequest(BaseModel):
    title: str = ""


class AddMessageRequest(BaseModel):
    role: str
    content: str
    sources: list = []


@app.get("/conversations", summary="获取所有对话列表")
async def list_conversations(x_user_id: str = Header(..., alias="X-User-ID")):
    return {"conversations": conversation_store.list_all(x_user_id)}


@app.post("/conversations", summary="创建新对话")
async def create_conversation(
    req: CreateConversationRequest = None,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    title = req.title if req else ""
    conv = conversation_store.create(x_user_id, title)
    return conv


@app.get("/conversations/{conv_id}", summary="获取对话详情")
async def get_conversation(
    conv_id: str,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    conv = conversation_store.get(x_user_id, conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conv


@app.post("/conversations/{conv_id}/messages", summary="追加消息")
async def add_message(
    conv_id: str,
    req: AddMessageRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    ok = conversation_store.add_message(x_user_id, conv_id, req.role, req.content, req.sources)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "ok"}


@app.put("/conversations/{conv_id}/title", summary="修改对话标题")
async def update_conversation_title(
    conv_id: str,
    title: str,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    ok = conversation_store.update_title(x_user_id, conv_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "ok"}


@app.delete("/conversations/{conv_id}", summary="删除对话")
async def delete_conversation(
    conv_id: str,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    ok = conversation_store.delete(x_user_id, conv_id)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "ok"}


# ═══════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════

def _collect_user_chunks(user_id: str) -> list:
    user_dir = _user_data_dir(user_id)
    if not os.path.exists(user_dir):
        return []

    all_chunks = []
    supported_exts = (".pdf", ".txt", ".md", ".markdown", ".csv", ".xlsx", ".xls")
    for filename in os.listdir(user_dir):
        if filename.lower().endswith(supported_exts):
            file_path = os.path.join(user_dir, filename)
            try:
                chunks = process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    return all_chunks


def _clear_user_index(user_id: str):
    store_dir = os.path.join(config.VECTOR_STORE_DIR, user_id)
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
        os.makedirs(store_dir, exist_ok=True)
    # 从缓存中移除
    rag_chain._stores.pop(user_id, None)
