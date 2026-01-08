from textual.widgets import Static
from textual.app import ComposeResult
from textual.containers import Horizontal

class MessageWidget(Static):
    """Виджет отдельного сообщения"""
    
    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
    
    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(
                f"[bold]{'User' if self.role == 'user' else 'Assistant'}:[/bold] ",
                classes="role-label"
            ),
            Static(self.content, classes="message-content", markup=True),
            classes=f"message {self.role}"
        )