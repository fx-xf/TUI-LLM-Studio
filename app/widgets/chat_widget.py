from textual.widgets import Input, Button
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.binding import Binding
from textual.reactive import reactive
from typing import List, Dict

from .message_widget import MessageWidget

class ChatWidget(Vertical):
    """Основной виджет чата"""
    
    BINDINGS = [
        Binding("escape", "blur_input", "Снять фокус"),
    ]
    
    messages = reactive([])
    
    def compose(self) -> ComposeResult:
        yield ScrollableContainer(
            Vertical(id="messages-container"),
            classes="messages-scroll"
        )
        yield Horizontal(
            Input(placeholder="Введите сообщение...", id="message-input"),
            Button("Send", id="send-btn", variant="primary"),
            classes="input-container"
        )
    
    def on_mount(self):
        self.input = self.query_one("#message-input", Input)
        self.messages_container = self.query_one("#messages-container", Vertical)
        self.send_btn = self.query_one("#send-btn", Button)
    
    def set_chat(self, chat_data):
        """Устанавливает текущий чат"""
        if chat_data:
            self.messages = chat_data["messages"]
        else:
            self.messages = []
    
    def watch_messages(self, messages: List[Dict]):
        """Обновляет UI при изменении сообщений"""
        container = self.messages_container
        container.remove_children()
        
        for msg in messages:
            container.mount(MessageWidget(msg["role"], msg["content"]))
        
        scroll = self.query_one(".messages-scroll")
        scroll.scroll_end(animate=False)
    
    def add_message(self, role: str, content: str, stream: bool = False):
        """Добавляет сообщение"""
        if stream:
            widget = MessageWidget(role, "")
            self.messages_container.mount(widget)
            return widget
        else:
            self.messages_container.mount(MessageWidget(role, content))
            scroll = self.query_one(".messages-scroll")
            scroll.scroll_end()
    
    def clear_input(self):
        self.input.value = ""
    
    def action_blur_input(self):
        self.input.blur()