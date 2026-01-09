# config.py
import os
from pathlib import Path

# Пути к директориям
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
CHATS_DIR = DATA_DIR / "chats"
MODELS_DIR = DATA_DIR / "models"

# Создание директорий
DATA_DIR.mkdir(exist_ok=True)
CHATS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Настройки модели
MODEL_CONFIG = {
    "model_path": MODELS_DIR / "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
    "model_type": "deepseek",  # llama, deepseek, mistral
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Настройки UI
UI_CONFIG = {
    "title": "LLM TUI - DeepSeek Coder",
    "theme": "dark",  # dark, light
    "max_history_items": 100,
}

# Настройки гиперпромптов
PROMPT_CONFIG = {
    "system_role": "You are DeepSeek Coder, a helpful AI assistant specialized in programming and technical tasks.",
    "default_template": "default",
}