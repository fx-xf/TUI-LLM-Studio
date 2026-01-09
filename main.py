#!/usr/bin/env python3
# main.py
import sys
from pathlib import Path
import signal
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter  # Добавили импорт
import torch  # Добавили импорт torch

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from config import MODEL_CONFIG, UI_CONFIG, PROMPT_CONFIG, CHATS_DIR
from model.loader import load_model_simplified, create_dummy_model  # Изменили импорт
from model.inference import ModelInference
from chat.manager import ChatManager
from prompts.hyperprompt import get_hyperprompt
from ui.app import ChatUI
from ui.components import UIComponents

def check_model_file():
    """Проверяет наличие файла модели"""
    model_path = MODEL_CONFIG["model_path"]
    if not model_path.exists():
        print(f"⚠️  Файл модели не найден: {model_path}")
        print(f"📁 Будет использована тестовая модель")
        return False
    return True

def load_model():
    """Загрузка модели"""
    try:
        from config import MODEL_CONFIG
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Инициализация модели на {device}...")
        
        # Проверяем, если файл GGUF существует
        gguf_path = MODEL_CONFIG["model_path"]
        if gguf_path.exists() and gguf_path.suffix == ".gguf":
            print("⚠️  GGUF файл обнаружен, но используем упрощенную загрузку через transformers")
        
        # Используем упрощенную загрузку
        model, metadata = load_model_simplified(gguf_path, device)
        
        print(f"✅ Модель готова!")
        print(f"   Архитектура: {metadata.get('model_type', 'unknown')}")
        print(f"   Слои: {metadata['num_layers']}")
        print(f"   Параметры: {metadata['hidden_size']}")
        
        return model, metadata
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        print("🔄 Используется тестовая модель-заглушка...")
        
        # Используем dummy модель для тестирования
        model, metadata = create_dummy_model("cpu")
        return model, metadata

def setup_hyperprompt():
    """Настройка гиперпромпта"""
    template_name = PROMPT_CONFIG.get("default_template", "default")
    hyperprompt = get_hyperprompt(template_name)
    hyperprompt.system_template = hyperprompt.system_template.replace(
        "{{ system_message }}", 
        PROMPT_CONFIG["system_role"]
    )
    return hyperprompt

def main():
    """Главная функция"""
    # Проверка модели (не критично для dummy)
    check_model_file()
    
    # Загрузка модели
    model, metadata = load_model()
    inference = ModelInference(model, metadata)
    
    # Настройка гиперпромпта
    hyperprompt = setup_hyperprompt()
    
    # Инициализация менеджера чатов
    chat_manager = ChatManager(CHATS_DIR)
    
    # Создание и запуск UI
    ui = ChatUI(chat_manager, inference, hyperprompt)
    
    # Обработка сигналов
    def signal_handler(sig, frame):
        print("\n🚪 Выход из приложения...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Приветственное сообщение
    UIComponents.clear_screen()
    print(f"""
╔════════════════════════════════════════════════════════════╗
║               LLM TUI - DeepSeek Coder                     ║
║                                                             ║
║  Модель: {MODEL_CONFIG['model_path'].name[:30]:<30} ║
║  Устройство: {'CUDA' if torch.cuda.is_available() else 'CPU':>44} ║
║  Горячие клавиши: Ctrl+H - Справка                         ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Запросить действие при старте
    action = prompt("Начать [N]овый чат, [L]oad чат или [Q]uit? ", 
                   completer=WordCompleter(["n", "l", "q"]), default="n").lower()
    
    if action == "l":
        chats = chat_manager.get_chat_list()
        if chats:
            UIComponents.print_header("Доступные чаты")
            for i, chat in enumerate(chats, 1):
                print(f"{i}. {chat['title']} ({chat['message_count']} сообщений)")
            try:
                choice = int(prompt("Выберите номер: ")) - 1
                if 0 <= choice < len(chats):
                    chat_manager.load_chat(chats[choice]["id"])
            except:
                pass
    elif action == "q":
        return
    
    # Запуск UI
    try:
        ui.run()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 До свидания!")

if __name__ == "__main__":
    main()