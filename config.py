"""
全局配置管理
优先级：环境变量 > .env 文件 > 代码默认值
"""
import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# ── HuggingFace 配置（本地开发用镜像+离线；云端部署不设这些变量即可）──
if os.getenv("HF_MIRROR", "0") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.getenv("HF_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _get_int(key: str, default: int) -> int:
    """安全获取整数环境变量（空字符串不会崩溃，回退到默认值）"""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(f"环境变量 {key}={raw!r} 不是合法整数，使用默认值 {default}")
        return default


def _get_float(key: str, default: float) -> float:
    """安全获取浮点环境变量"""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"环境变量 {key}={raw!r} 不是合法浮点数，使用默认值 {default}")
        return default


class Config:
    """集中管理所有配置项"""

    # ── LLM 配置 ──
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # ── Embedding 配置（本地 HuggingFace 模型，免费无需 API Key）──
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    )

    # ── 文本切分配置 ──
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 50)

    # ── 检索配置 ──
    TOP_K: int = _get_int("TOP_K", 4)

    # ── 路径配置 ──
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "vector_store")

    # ── LLM 温度 ──
    TEMPERATURE: float = 0.1

    # ── 安全配置 ──
    # FAISS 索引反序列化：默认信任本地索引文件（仅加载用户自己构建的索引）
    # 如果索引文件可能被第三方篡改，请设置 FAISS_TRUST_INDEX=0 禁用加载
    FAISS_ALLOW_DANGEROUS_DESERIALIZATION: bool = (
        os.getenv("FAISS_TRUST_INDEX", "1") == "1"
    )

    # ── 文件留存时间（小时），过期自动删除。0 表示永不自动删除 ──
    FILE_RETENTION_HOURS: int = _get_int("FILE_RETENTION_HOURS", 24)

    # ── 智能检索最大轮数 ──
    MAX_RETRIEVAL_ROUNDS: int = _get_int("MAX_RETRIEVAL_ROUNDS", 2)

    # CORS 允许的来源（逗号分隔，默认仅本地开发地址）
    CORS_ORIGINS: list = [
        origin.strip() for origin in
        os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    @classmethod
    def validate(cls) -> bool:
        """验证必要配置是否完整"""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "sk-your-api-key-here":
            raise ValueError(
                "⚠️ 请先配置 API Key！\n"
                "  方法1: 复制 .env.example 为 .env，填入真实 API Key\n"
                "  方法2: 设置环境变量 OPENAI_API_KEY=你的key"
            )
        return True


config = Config()
