# chat/storage.py
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class ChatStorage:
    """Класс для хранения и загрузки чатов в JSON формате"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_chat(self, chat_data: Dict[str, Any]) -> str:
        """
        Сохраняет чат в файл
        """
        chat_id = chat_data.get("id") or str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        if "created_at" not in chat_data:
            chat_data["created_at"] = timestamp
        chat_data["updated_at"] = timestamp
        
        file_path = self.storage_dir / f"{chat_id}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        return chat_id
    
    def load_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """
        Загружает чат по ID
        """
        file_path = self.storage_dir / f"{chat_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def delete_chat(self, chat_id: str) -> bool:
        """
        Удаляет чат по ID
        """
        file_path = self.storage_dir / f"{chat_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def list_chats(self) -> List[Dict[str, Any]]:
        """
        Возвращает список всех чатов с метаданными
        """
        chats = []
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    meta = {
                        "id": data.get("id", json_file.stem),
                        "title": data.get("title", "Без названия"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "message_count": len(data.get("messages", [])),
                    }
                    chats.append(meta)
            except Exception as e:
                print(f"⚠️ Ошибка чтения {json_file}: {e}")
        
        # Сортировка по дате обновления (новые первые)
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats
    
    def chat_exists(self, chat_id: str) -> bool:
        """Проверяет существование чата"""
        return (self.storage_dir / f"{chat_id}.json").exists()