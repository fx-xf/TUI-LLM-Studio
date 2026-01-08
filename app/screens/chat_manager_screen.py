from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, ListView, ListItem, Label, Button, Static
from textual.containers import Horizontal
from textual.binding import Binding

from ..core.chat_manager import ChatManager

class ChatManagerScreen(Screen):
    """Экран управления чатами"""
    
    BINDINGS = [
        Binding("escape", "back", "Назад"),
        Binding("delete", "delete_selected", "Удалить"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_manager = ChatManager()
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("📁 Управление чатами", classes="title")
        yield ListView(id="chats-list")
        yield Horizontal(
            Button("📂 Загрузить", id="load-btn", variant="primary"),
            Button("🗑️ Удалить", id="delete-btn", variant="error"),
            classes="buttons"
        )
        yield Footer()
    
    def on_mount(self):
        self.chats_list = self.query_one("#chats-list", ListView)
        self.refresh_list()
    
    def refresh_list(self):
        """Обновляет список чатов"""
        chats = self.chat_manager.list_chats()
        self.chats_list.clear()
        
        if not chats:
            self.chats_list.mount(
                ListItem(Label("Нет чатов. Создайте новый (Ctrl+N)"))
            )
        else:
            for chat in chats:
                item = ListItem(Label(
                    f"{chat['name']} | {chat['message_count']} сообщений | "
                    f"{chat['updated_at'][:10]}"
                ))
                item.chat_id = chat["id"]
                self.chats_list.mount(item)
    
    def on_button_pressed(self, event):
        if event.button.id == "load-btn":
            self.action_load_selected()
        elif event.button.id == "delete-btn":
            self.action_delete_selected()
    
    def action_load_selected(self):
        """Загружает выбранный чат"""
        selected = self.chats_list.highlighted_child
        if selected and hasattr(selected, "chat_id"):
            chat_data = self.chat_manager.load_chat(selected.chat_id)
            if chat_data:
                self.app.pop_screen()
                main_screen = self.app.screen
                main_screen.current_chat = chat_data
                main_screen.chat_widget.set_chat(chat_data)
                self.notify(f"📂 Чат загружен: {chat_data['name']}")
    
    def action_delete_selected(self):
        """Удаляет выбранный чат"""
        selected = self.chats_list.highlighted_child
        if selected and hasattr(selected, "chat_id"):
            success = self.chat_manager.delete_chat(selected.chat_id)
            if success:
                self.notify("🗑️ Чат удалён")
                self.refresh_list()
    
    def action_back(self):
        self.app.pop_screen()