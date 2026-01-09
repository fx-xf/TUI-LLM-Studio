# ui/app.py
from prompt_toolkit import Application
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea, Frame, Label
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
import asyncio
import threading
from typing import Optional, Callable

class ChatUI:
    """Основной класс TUI интерфейса"""
    
    def __init__(self, chat_manager, model_inference, hyperprompt):
        self.chat_manager = chat_manager
        self.model_inference = model_inference
        self.hyperprompt = hyperprompt
        
        self.generating = False
        self.current_response = ""
        
        # История сообщений
        self.chat_history = TextArea(
            text="Добро пожаловать в LLM TUI!\nНачните диалог или загрузите существующий чат.",
            read_only=True,
            focusable=False,
            height=None,
            style="class:text-area"
        )
        
        # Поле ввода
        self.input_field = TextArea(
            height=3,
            prompt="> ",
            wrap_lines=True,
            multiline=False,
            style="class:input-field"
        )
        
        # Статусная строка
        self.status_bar = Label(
            text="Готов | Ctrl+H - Справка | Ctrl+N - Новый чат",
            style="class:status"
        )
        
        # Привязки клавиш
        self.kb = KeyBindings()
        self._setup_keybindings()
        
        # Layout
        self.layout = self._create_layout()
        
        # Application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self._get_style(),
            full_screen=True,
            mouse_support=True,
        )
        
        # Callback для генерации
        self.generation_callback: Optional[Callable] = None
    
    def _create_layout(self):
        """Создание layout"""
        return Layout(
            HSplit([
                Frame(
                    body=self.chat_history,
                    title="💬 Чат",
                    style="class:chat-frame"
                ),
                Window(height=1, char="─"),
                Frame(
                    body=self.input_field,
                    title="Ввод",
                    height=4,
                    style="class:input-frame"
                ),
                Window(height=1, char="─"),
                self.status_bar,
            ])
        )
    
    def _get_style(self):
        """Стилизация интерфейса"""
        return Style([
            ("chat-frame", "bg:#282c34 #abb2bf"),
            ("input-frame", "bg:#282c34 #abb2bf"),
            ("text-area", "bg:#1e2127 #abb2bf"),
            ("input-field", "bg:#1e2127 #abb2bf"),
            ("status", "bg:#3e4451 #abb2bf"),
        ])
    
    def _setup_keybindings(self):
        """Настройка горячих клавиш"""
        
        @self.kb.add("enter", filter=~self.input_field.buffer.read_only)
        def _(event):
            """Отправка сообщения"""
            text = self.input_field.text.strip()
            if text:
                if text.startswith("/"):
                    self._handle_command(text)
                else:
                    self._send_message(text)
                self.input_field.text = ""
        
        @self.kb.add("c-n")
        def _(event):
            """Новый чат"""
            self._handle_command("/new")
        
        @self.kb.add("c-l")
        def _(event):
            """Загрузить чат"""
            self._handle_command("/load")
        
        @self.kb.add("c-s")
        def _(event):
            """Сохранить чат"""
            self._handle_command("/save")
        
        @self.kb.add("c-h")
        def _(event):
            """Помощь"""
            self._handle_command("/help")
        
        @self.kb.add("c-c")
        def _(event):
            """Прервать генерацию"""
            if self.generating:
                self.generating = False
                self._update_status("Генерация прервана")
        
        @self.kb.add("c-q")
        def _(event):
            """Выход"""
            self._handle_command("/exit")
    
    def _handle_command(self, command: str):
        """Обработка команд"""
        cmd = command.lower().strip()
        
        if cmd == "/exit":
            self._exit_app()
        elif cmd == "/new":
            self._new_chat()
        elif cmd == "/load":
            self._load_chat()
        elif cmd == "/save":
            self._save_chat()
        elif cmd == "/delete":
            self._delete_chat()
        elif cmd == "/clear":
            self._clear_chat()
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/config":
            self._show_config()
        else:
            self._add_to_history("system", f"Неизвестная команда: {command}")
    
    def _send_message(self, message: str):
        """Отправка сообщения модели"""
        if not self.chat_manager.current_chat:
            # Автоматически создаем чат
            self.chat_manager.create_chat("Без названия", self.hyperprompt.__dict__)
            self._update_status("Создан новый чат")
        
        # Добавляем сообщение пользователя
        self._add_to_history("user", message)
        self.chat_manager.add_message("user", message)
        
        # Генерация ответа
        asyncio.create_task(self._generate_response())
    
    async def _generate_response(self):
        """Генерация ответа модели"""
        if self.generating:
            return
        
        self.generating = True
        self.current_response = ""
        
        # Формирование промпта
        messages = [{"role": m.role, "content": m.content} for m in self.chat_manager.get_messages()]
        prompt = self.hyperprompt.format_prompt(messages)
        
        self._update_status("🤖 Генерация...")
        
        try:
            # Генерация в отдельном потоке для неблокирующего UI
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._generate_sync, prompt)
            
            if self.current_response.strip():
                self.chat_manager.add_message("assistant", self.current_response)
        except Exception as e:
            self._add_to_history("system", f"Ошибка генерации: {e}")
        finally:
            self.generating = False
            self._update_status("Готов")
    
    def _generate_sync(self, prompt: str):
        """Синхронная генерация (выполняется в отдельном потоке)"""
        for token in self.model_inference.generate(
            prompt=prompt,
            max_tokens=self.model_inference.metadata.get("max_tokens", 2048),
            temperature=self.model_inference.metadata.get("temperature", 0.7),
            top_p=self.model_inference.metadata.get("top_p", 0.95),
            top_k=self.model_inference.metadata.get("top_k", 40),
            stop_sequences=self.hyperprompt.stop_sequences,
            stream=True
        ):
            if not self.generating:
                break
            
            self.current_response += token
            # Обновляем UI в главном потоке
            asyncio.run_coroutine_threadsafe(
                self._update_response(token), 
                asyncio.get_event_loop()
            )
    
    async def _update_response(self, token: str):
        """Обновление ответа в UI"""
        # Добавляем токен к последнему сообщению assistant
        lines = self.chat_history.text.split("\n")
        if lines and "Assistant:" in lines[-1]:
            lines[-1] += token
        else:
            lines.append(f"Assistant: {token}")
        
        self.chat_history.text = "\n".join(lines)
        self.chat_history.buffer.cursor_position = len(self.chat_history.text)
    
    def _add_to_history(self, role: str, content: str):
        """Добавление сообщения в историю"""
        prefix = {
            "user": "👤 You:",
            "assistant": "🤖 Assistant:",
            "system": "⚙️ System:",
        }
        
        text = self.chat_history.text
        if not text.endswith("\n"):
            text += "\n"
        
        text += f"\n{prefix.get(role, '')}\n{content}"
        self.chat_history.text = text
        self.chat_history.buffer.cursor_position = len(text)
    
    def _new_chat(self):
        """Создание нового чата"""
        title = prompt("Название чата: ", default=f"Чат {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if title:
            self.chat_manager.create_chat(title, self.hyperprompt.__dict__)
            self.chat_history.text = f"Создан новый чат: {title}\n"
    
    def _load_chat(self):
        """Загрузка чата"""
        chats = self.chat_manager.get_chat_list()
        if not chats:
            self._add_to_history("system", "Нет сохраненных чатов")
            return
        
        # Простой текстовый выбор (вместо radiolist)
        self._add_to_history("system", "Доступные чаты:")
        for i, chat in enumerate(chats[:10], 1):
            self._add_to_history("system", f"  {i}. {chat['title']} ({chat['message_count']} сообщений)")
        
        choice = prompt("Введите номер чата или ID: ")
        
        try:
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(chats):
                    chat_id = chats[idx]["id"]
                else:
                    self._add_to_history("system", "Неверный номер")
                    return
            else:
                chat_id = choice
            
            if self.chat_manager.load_chat(chat_id):
                # Обновляем историю UI
                self.chat_history.text = ""
                for msg in self.chat_manager.get_messages():
                    self._add_to_history(msg.role, msg.content)
                self._update_status(f"Чат загружен: {self.chat_manager.current_chat.title}")
            else:
                self._add_to_history("system", "Чат не найден")
        except Exception as e:
            self._add_to_history("system", f"Ошибка загрузки: {e}")
    
    def _save_chat(self):
        """Сохранение чата"""
        if not self.chat_manager.current_chat:
            self._add_to_history("system", "Нет активного чата для сохранения")
            return
        
        self.chat_manager.save_current_chat()
        self._update_status("Чат сохранен")
    
    def _delete_chat(self):
        """Удаление чата"""
        if not self.chat_manager.current_chat:
            self._add_to_history("system", "Нет активного чата")
            return
        
        if self.confirm_action("Удаление", f"Удалить чат '{self.chat_manager.current_chat.title}'?"):
            chat_id = self.chat_manager.current_chat.id
            if self.chat_manager.delete_chat(chat_id):
                self.chat_history.text = f"Чат '{self.chat_manager.current_chat.title}' удален\n"
                self.chat_manager.current_chat = None
    
    def _clear_chat(self):
        """Очистка чата"""
        if not self.chat_manager.current_chat:
            self._add_to_history("system", "Нет активного чата")
            return
        
        if self.confirm_action("Очистка", "Очистить историю текущего чата?"):
            self.chat_manager.clear_current_chat()
            self.chat_history.text = "История чата очищена\n"
    
    def _show_help(self):
        """Показ помощи"""
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

Горячие клавиши:
  Ctrl+N   - Новый чат
  Ctrl+L   - Загрузить чат
  Ctrl+S   - Сохранить чат
  Ctrl+H   - Помощь
  Ctrl+C   - Прервать генерацию
  Ctrl+Q   - Выход
"""
        self._add_to_history("system", help_text)
    
    def _show_config(self):
        """Показ конфигурации"""
        from config import MODEL_CONFIG
        config_text = f"""
Модель: {MODEL_CONFIG['model_path']}
Max tokens: {MODEL_CONFIG['max_tokens']}
Temperature: {MODEL_CONFIG['temperature']}
Top-p: {MODEL_CONFIG['top_p']}
Top-k: {MODEL_CONFIG['top_k']}
"""
        self._add_to_history("system", config_text)
    
    def _update_status(self, text: str):
        """Обновление статусной строки"""
        self.status_bar.text = f"{text} | Ctrl+H - Справка"
    
    def _exit_app(self):
        """Выход из приложения"""
        if self.generating:
            if not self.confirm_action("Выход", "Генерация в процессе. Выйти?"):
                return
        
        if self.chat_manager.current_chat:
            self.chat_manager.save_current_chat()
        
        get_app().exit()
    
    def run(self):
        """Запуск UI"""
        self.app.run()