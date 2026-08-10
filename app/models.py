"""
Pydantic 数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., description="用户提问内容", min_length=1)
    top_k: Optional[int] = Field(default=None, description="检索文档数量")


class SourceDocument(BaseModel):
    """检索到的源文档信息"""
    content: str = Field(..., description="文档片段内容")
    source: str = Field(..., description="来源文件名")
    page: Optional[int] = Field(default=None, description="页码")


class QueryResponse(BaseModel):
    """问答响应"""
    answer: str = Field(..., description="模型生成的回答")
    sources: List[SourceDocument] = Field(default_factory=list, description="参考来源")


class UploadResponse(BaseModel):
    """文件上传响应"""
    filename: str = Field(..., description="上传的文件名")
    chunks: int = Field(..., description="切分块数")
    message: str = Field(..., description="状态消息")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(default=None, description="详细错误")
