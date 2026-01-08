import json
from pathlib import Path
from typing import Dict, List, Optional, Any

class Hyperprompt:
    """Система гиперпромптов с шаблонами и few-shot примерами"""
    
    def __init__(self, system_template: str = "", examples: Optional[List[Dict]] = None):
        self.system_template = system_template
        self.examples = examples or []
    
    def generate_prompt(self, user_message: str, context: Optional[List[Dict]] = None) -> str:
        """Генерирует финальный промпт"""
        parts = []
        
        # Системный промпт
        if self.system_template:
            parts.append(f"<system>\n{self.system_template}\n</system>")
        
        # Few-shot примеры
        for ex in self.examples:
            parts.append(
                f"<example>\nUser: {ex['user']}\nAssistant: {ex['assistant']}\n</example>"
            )
        
        # Контекст (последние 3 сообщения)
        if context:
            for msg in context[-3:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"<context>\n{role}: {msg['content']}\n</context>")
        
        # Текущее сообщение
        parts.append(f"<user>\n{user_message}\n</user>\n\n<assistant>\n")
        
        return "\n\n".join(parts)
    
    def save(self, path: Path):
        """Сохраняет в JSON"""
        data = {
            "system_template": self.system_template,
            "examples": self.examples
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "Hyperprompt":
        """Загружает из файла"""
        data = json.loads(path.read_text())
        return cls(**data)