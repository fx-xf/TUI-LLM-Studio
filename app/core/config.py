from pathlib import Path
from typing import Dict, Any
import torch

class Config:
    # Пути
    MODEL_NAME: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    CHATS_DIR: Path = Path("data/chats")
    
    # Параметры модели
    MAX_CONTEXT_LENGTH: int = 4096
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Параметры генерации
    GENERATION_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 1024,
        "do_sample": True,
    }
    
    # Настройки квантизации
    USE_4BIT_QUANTIZATION: bool = True  # Экономит память
    
    @classmethod
    def initialize(cls):
        """Создаёт необходимые директории"""
        cls.CHATS_DIR.mkdir(parents=True, exist_ok=True)