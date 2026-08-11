"""
对话历史持久化存储
使用 JSON 文件存储，每个对话一个文件
目录结构: data/conversations/{user_id}/{conv_id}.json
"""
import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict


class ConversationStore:
    """管理多轮对话的增删查改，按 user_id 隔离"""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _user_dir(self, user_id: str) -> str:
        d = os.path.join(self.storage_dir, user_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _path(self, user_id: str, conv_id: str) -> str:
        return os.path.join(self._user_dir(user_id), f"{conv_id}.json")

    def create(self, user_id: str, title: str = "") -> dict:
        """新建一个对话"""
        conv_id = uuid.uuid4().hex[:12]
        conv = {
            "id": conv_id,
            "user_id": user_id,
            "title": title or "新对话",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
        }
        self._save(user_id, conv)
        return conv

    def get(self, user_id: str, conv_id: str) -> Optional[dict]:
        """获取一个对话"""
        path = self._path(user_id, conv_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_all(self, user_id: str) -> List[dict]:
        """列出指定用户的所有对话（按更新时间倒序，只返回摘要）"""
        convs = []
        user_dir = self._user_dir(user_id)
        for fname in os.listdir(user_dir):
            if fname.endswith(".json"):
                path = os.path.join(user_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        conv = json.load(f)
                    convs.append({
                        "id": conv["id"],
                        "title": conv["title"],
                        "created_at": conv["created_at"],
                        "updated_at": conv["updated_at"],
                        "message_count": len(conv.get("messages", [])),
                    })
                except Exception:
                    pass
        convs.sort(key=lambda c: c["updated_at"], reverse=True)
        return convs

    def add_message(self, user_id: str, conv_id: str, role: str, content: str, sources: list = None) -> bool:
        """给对话追加一条消息"""
        conv = self.get(user_id, conv_id)
        if conv is None:
            return False

        msg = {
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat(),
        }
        conv["messages"].append(msg)
        conv["updated_at"] = datetime.now().isoformat()

        if conv["title"] == "新对话" and role == "user":
            conv["title"] = content[:30] + ("..." if len(content) > 30 else "")

        self._save(user_id, conv)
        return True

    def update_title(self, user_id: str, conv_id: str, title: str) -> bool:
        """修改对话标题"""
        conv = self.get(user_id, conv_id)
        if conv is None:
            return False
        conv["title"] = title
        conv["updated_at"] = datetime.now().isoformat()
        self._save(user_id, conv)
        return True

    def delete(self, user_id: str, conv_id: str) -> bool:
        """删除一个对话"""
        path = self._path(user_id, conv_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    def _save(self, user_id: str, conv: dict):
        path = self._path(user_id, conv["id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(conv, f, ensure_ascii=False, indent=2)


# ── 全局单例 ──
from config import config

conversation_store = ConversationStore(
    os.path.join(config.BASE_DIR, "data", "conversations")
)
