from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, TextArea, Button, Static
from textual.containers import Horizontal
from textual.binding import Binding
from pathlib import Path
import json

from ..core.hyperprompt import Hyperprompt

class HyperpromptEditorScreen(Screen):
    """Редактор гиперпромпта"""
    
    BINDINGS = [
        Binding("escape", "back", "Назад"),
        Binding("ctrl+s", "save", "Сохранить"),
    ]
    
    def __init__(self, hyperprompt: Hyperprompt, **kwargs):
        super().__init__(**kwargs)
        self.hyperprompt = hyperprompt
        self.save_path = Path("data/hyperprompt.json")
        self.save_path.parent.mkdir(exist_ok=True)
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("⚙️ Редактор гиперпромпта", classes="title")
        yield Static("Системный промпт:", classes="label")
        yield TextArea(
            self.hyperprompt.system_template,
            id="system-prompt",
            classes="text-area"
        )
        yield Static("Few-shot примеры (JSON):", classes="label")
        yield TextArea(
            self._examples_to_json(),
            id="examples",
            classes="text-area"
        )
        yield Horizontal(
            Button("💾 Сохранить", id="save-btn", variant="primary"),
            Button("❌ Отмена", id="cancel-btn"),
            classes="buttons"
        )
        yield Footer()
    
    def _examples_to_json(self) -> str:
        return json.dumps(self.hyperprompt.examples, ensure_ascii=False, indent=2)
    
    def _json_to_examples(self, json_str: str) -> list:
        try:
            return json.loads(json_str)
        except:
            return []
    
    def on_button_pressed(self, event):
        if event.button.id == "save-btn":
            self.action_save()
        elif event.button.id == "cancel-btn":
            self.action_back()
    
    def action_save(self):
        system = self.query_one("#system-prompt", TextArea).text
        examples = self.query_one("#examples", TextArea).text
        
        self.hyperprompt.system_template = system
        self.hyperprompt.examples = self._json_to_examples(examples)
        self.hyperprompt.save(self.save_path)
        
        self.notify("✅ Гиперпромпт сохранён")
        self.action_back()
    
    def action_back(self):
        self.app.pop_screen()