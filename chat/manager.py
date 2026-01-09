# chat/manager.py
import uuid  # Добавили импорт
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from .storage import ChatStorage
from pathlib import Path

@dataclass
class Message:
    """Класс сообщения"""
    role: str  # system, user, assistant
    content: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class Chat:
    """Класс чата"""
    id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str
    model_config: Dict[str, Any] = None
    
    def add_message(self, role: str, content: str):
        """Добавляет сообщение в чат"""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model_config": self.model_config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chat":
        """Десериализация из словаря"""
        messages = [Message(**m) for m in data.get("messages", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            messages=messages,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            model_config=data.get("model_config"),
        )

class ChatManager:
    """Менеджер для управления чатами"""
    
    def __init__(self, storage_dir: Path):
        self.storage = ChatStorage(storage_dir)
        self.current_chat: Chat = None
    
    def create_chat(self, title: str = "Новый чат", model_config: Dict = None) -> Chat:
        """Создает новый чат"""
        chat = Chat(
            id=str(uuid.uuid4()),
            title=title,
            messages=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            model_config=model_config,
        )
        
        # Добавляем системное сообщение
        system_msg = model_config.get("system_message", "") if model_config else ""
        if system_msg:
            chat.add_message("system", system_msg)
        
        self.current_chat = chat
        return chat
    
    def load_chat(self, chat_id: str) -> bool:
        """Загружает чат по ID"""
        data = self.storage.load_chat(chat_id)
        if data:
            self.current_chat = Chat.from_dict(data)
            return True
        return False
    
    def save_current_chat(self):
        """Сохраняет текущий чат"""
        if self.current_chat:
            self.storage.save_chat(self.current_chat.to_dict())
    
    def delete_chat(self, chat_id: str) -> bool:
        """Удаляет чат"""
        if self.current_chat and self.current_chat.id == chat_id:
            self.current_chat = None
        return self.storage.delete_chat(chat_id)
    
    def get_chat_list(self) -> List[Dict[str, Any]]:
        """Получает список чатов"""
        return self.storage.list_chats()
    
    def add_message(self, role: str, content: str):
        """Добавляет сообщение в текущий чат"""
        if not self.current_chat:
            raise ValueError("Нет активного чата")
        
        self.current_chat.add_message(role, content)
        self.save_current_chat()  # Автосохранение
    
    def get_messages(self) -> List[Message]:
        """Получает сообщения текущего чата"""
        if not self.current_chat:
            return []
        return self.current_chat.messages
    
    def clear_current_chat(self):
        """Очищает текущий чат (удаляет сообщения)"""
        if self.current_chat:
            self.current_chat.messages = []
            self.save_current_chat()