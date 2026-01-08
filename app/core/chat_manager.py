import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
from .config import Config

class ChatManager:
    """Управление сохранением/загрузкой чатов"""
    
    def __init__(self, chats_dir: Optional[Path] = None):
        self.chats_dir = chats_dir or Config.CHATS_DIR
        self.chats_dir.mkdir(exist_ok=True)
    
    def create_chat(self, name: str) -> Dict[str, Any]:
        """Создаёт новый чат"""
        chat_id = str(uuid.uuid4())[:8]
        chat_data = {
            "id": chat_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        self._save_chat(chat_data)
        return chat_data
    
    def load_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Загружает чат по ID"""
        chat_file = self.chats_dir / f"{chat_id}.json"
        if chat_file.exists():
            return json.loads(chat_file.read_text())
        return None
    
    def save_chat(self, chat_data: Dict[str, Any]):
        """Сохраняет чат"""
        chat_data["updated_at"] = datetime.now().isoformat()
        self._save_chat(chat_data)
    
    def _save_chat(self, chat_data: Dict[str, Any]):
        """Внутреннее сохранение"""
        chat_file = self.chats_dir / f"{chat_data['id']}.json"
        chat_file.write_text(json.dumps(chat_data, ensure_ascii=False, indent=2))
    
    def delete_chat(self, chat_id: str) -> bool:
        """Удаляет чат"""
        chat_file = self.chats_dir / f"{chat_id}.json"
        if chat_file.exists():
            chat_file.unlink()
            return True
        return False
    
    def list_chats(self) -> List[Dict[str, Any]]:
        """Возвращает список всех чатов"""
        chats = []
        for chat_file in self.chats_dir.glob("*.json"):
            try:
                data = json.loads(chat_file.read_text())
                chats.append({
                    "id": data["id"],
                    "name": data["name"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "message_count": len(data["messages"])
                })
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {chat_file}: {e}")
        
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats
    
    def add_message(self, chat_id: str, role: str, content: str):
        """Добавляет сообщение в чат"""
        chat_data = self.load_chat(chat_id)
        if chat_data:
            chat_data["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            self.save_chat(chat_data)