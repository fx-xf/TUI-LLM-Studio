from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer
from textual.containers import Vertical
from textual.binding import Binding
from textual.reactive import reactive
import asyncio

from ..widgets.chat_widget import ChatWidget
from ..core.llm_engine_torch import LLMEngine
from ..core.chat_manager import ChatManager
from ..core.hyperprompt import Hyperprompt

class ChatScreen(Screen):
    """Главный экран чата"""
    
    BINDINGS = [
        Binding("ctrl+s", "save_chat", "Сохранить"),
        Binding("ctrl+n", "new_chat", "Новый чат"),
        Binding("ctrl+m", "manage_chats", "Мои чаты"),
        Binding("ctrl+h", "edit_hyperprompt", "Гиперпромпт"),
        Binding("ctrl+q", "quit", "Выход"),
    ]
    
    current_chat = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm = LLMEngine()
        self.chat_manager = ChatManager()
        self.hyperprompt = None
        self.is_generating = False
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(ChatWidget(id="chat-widget"))
        yield Footer()
    
    def on_mount(self):
        self.chat_widget = self.query_one("#chat-widget", ChatWidget)
        self.chat_widget.input.focus()
        self.action_new_chat()
    
    def action_new_chat(self):
        """Создаёт новый чат"""
        chat_data = self.chat_manager.create_chat("Новый чат")
        self.current_chat = chat_data
        self.chat_widget.set_chat(chat_data)
        self.notify(f"💬 Создан чат: {chat_data['name']}")
    
    def action_save_chat(self):
        """Сохраняет текущий чат"""
        if self.current_chat:
            self.chat_manager.save_chat(self.current_chat)
            self.notify("💾 Чат сохранён")
    
    def action_manage_chats(self):
        """Переход к управлению чатами"""
        self.app.push_screen("chat_manager")
    
    def action_edit_hyperprompt(self):
        """Редактирование гиперпромпта"""
        editor = self.app.get_hyperprompt_editor()
        self.app.push_screen(editor)
    
    def action_quit(self):
        self.app.exit()
    
    async def on_input_submitted(self, event):
        """Обработка отправки сообщения"""
        if event.input.id != "message-input" or self.is_generating:
            return
        
        message = event.value.strip()
        if not message:
            return
        
        self.is_generating = True
        self.chat_widget.input.disabled = True
        
        try:
            # Сохраняем сообщение пользователя
            self.chat_manager.add_message(self.current_chat["id"], "user", message)
            self.chat_widget.add_message("user", message)
            self.chat_widget.clear_input()
            
            # Создаём пустое сообщение для стриминга
            assistant_widget = self.chat_widget.add_message("assistant", "", stream=True)
            
            # Генерируем ответ
            messages = self.current_chat["messages"]
            response_text = ""
            
            async for token in self.llm.generate(messages, self.hyperprompt):
                response_text += token
                assistant_widget.update_content(response_text)
            
            # Сохраняем ответ
            self.chat_manager.add_message(self.current_chat["id"], "assistant", response_text)
            self.current_chat = self.chat_manager.load_chat(self.current_chat["id"])
            
        except Exception as e:
            self.notify(f"❌ Ошибка: {str(e)}", severity="error")
        finally:
            self.is_generating = False
            self.chat_widget.input.disabled = False
            self.chat_widget.input.focus()