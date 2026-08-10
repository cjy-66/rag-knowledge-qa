"""
全局配置管理
优先级：环境变量 > .env 文件 > 代码默认值
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── HuggingFace 配置（本地开发用镜像+离线；云端部署不设这些变量即可）──
if os.getenv("HF_MIRROR", "0") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.getenv("HF_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


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
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── 检索配置 ──
    TOP_K: int = int(os.getenv("TOP_K", "4"))

    # ── 路径配置 ──
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "vector_store")

    # ── LLM 温度 ──
    TEMPERATURE: float = 0.1

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
