# ui/components.py
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import radiolist_dialog, button_dialog, message_dialog
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from typing import List, Dict, Optional, Callable, Any
import sys

# Стиль TUI
style = Style.from_dict({
    "dialog": "bg:#282c34",
    "dialog frame.label": "bg:#282c34 #61afef",
    "dialog.body": "bg:#282c34 #abb2bf",
    "dialog shadow": "bg:#000000",
    "button": "bg:#3e4451 #abb2bf",
    "button.focused": "bg:#61afef #282c34",
    "text-area": "bg:#1e2127 #abb2bf",
})

class UIComponents:
    """UI компоненты для TUI"""
    
    @staticmethod
    def clear_screen():
        """Очищает экран терминала"""
        print("\033c", end="")
    
    @staticmethod
    def print_header(text: str):
        """Печатает заголовок"""
        print(f"\n{'='*60}")
        print(f" {text}")
        print('='*60)
    
    @staticmethod
    def print_message(role: str, content: str):
        """Печатает сообщение с цветовой кодировкой"""
        colors = {
            "user": "\033[94m",      # Синий
            "assistant": "\033[92m",  # Зеленый
            "system": "\033[93m",     # Желтый
        }
        reset = "\033[0m"
        
        prefix = {
            "user": "👤 You:",
            "assistant": "🤖 Assistant:",
            "system": "⚙️ System:",
        }
        
        color = colors.get(role, "")
        pref = prefix.get(role, "")
        
        print(f"\n{color}{pref}{reset}")
        print(f"{content}")
    
    @staticmethod
    def select_chat(chats: List[Dict[str, Any]]) -> Optional[str]:
        """Диалог выбора чата"""
        if not chats:
            message_dialog(
                title="Чаты",
                text="Нет сохраненных чатов",
                style=style
            ).run()
            return None
        
        values = [(c["id"], f"{c['title']} ({c['message_count']} сообщений)") for c in chats]
        
        result = radiolist_dialog(
            title="Выберите чат",
            text="Выберите чат для загрузки:",
            values=values,
            style=style
        ).run()
        
        return result
    
    @staticmethod
    def confirm_action(title: str, text: str) -> bool:
        """Диалог подтверждения действия"""
        return button_dialog(
            title=title,
            text=text,
            buttons=[
                ("Да", True),
                ("Нет", False),
            ],
            style=style
        ).run()
    
    @staticmethod
    def input_text(title: str, prompt_text: str, default: str = "") -> str:
        """Ввод текста"""
        completer = WordCompleter(["/exit", "/new", "/save", "/load", "/delete", "/clear", "/help"])
        
        return prompt(
            HTML(f"<b>{title}</b>\n{prompt_text}: "),
            default=default,
            completer=completer,
            style=style
        )
    
    @staticmethod
    def show_help():
        """Показывает помощь"""
        help_text = """
Доступные команды:
  /new     - Создать новый чат
  /load    - Загрузить существующий чат
  /save    - Сохранить текущий чат
  /delete  - Удалить чат
  /clear   - Очистить историю текущего чата
  /config  - Показать конфигурацию
  /exit    - Выйти из приложения
  /help    - Показать эту справку

Навигация:
  ↑/↓      - Прокрутка истории
  Tab      - Автодополнение команд
  Ctrl+C   - Прервать генерацию
"""
        message_dialog(
            title="Справка",
            text=help_text,
            style=style
        ).run()
    
    @staticmethod
    def show_config(config: Dict[str, Any]):
        """Показывает конфигурацию"""
        config_text = f"""
Модель: {config.get('model_path', 'Не указана')}
Устройство: {config.get('device', 'auto')}
Max tokens: {config.get('max_tokens', 2048)}
Temperature: {config.get('temperature', 0.7)}
Top-p: {config.get('top_p', 0.95)}
Top-k: {config.get('top_k', 40)}
"""
        message_dialog(
            title="Конфигурация",
            text=config_text,
            style=style
        ).run()