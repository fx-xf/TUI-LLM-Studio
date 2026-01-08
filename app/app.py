from textual.app import App
from pathlib import Path
import logging  # добавили лог

from .screens.chat_screen import ChatScreen
from .screens.chat_manager_screen import ChatManagerScreen
from .screens.hyperprompt_editor_screen import HyperpromptEditorScreen
from .core.hyperprompt import Hyperprompt

log = logging.getLogger("main")  # общий логгер

class LLMTUIApp(App):
    CSS_PATH = "../styles.css"

    def __init__(self, **kwargs):
        log.info("LLMTUIApp.__init__ старт")
        super().__init__(**kwargs)
        self.hyperprompt = self._load_hyperprompt()
        log.info("LLMTUIApp.__init__ завершён")

    # ---------- гиперпромпт (без изменений) ----------
    def _load_hyperprompt(self) -> Hyperprompt:
        path = Path("data/hyperprompt.json")
        path.parent.mkdir(exist_ok=True)
        if path.exists() and path.stat().st_size:
            try:
                return Hyperprompt.load(path)
            except Exception:
                pass
        default = Hyperprompt(
            system_template="Ты — помощник-программист. Пиши кратко, в markdown.",
            examples=[{"user": "Привет", "assistant": "Привет! Чем помочь?"}],
        )
        default.save(path)
        return default

    def on_mount(self):
        log.info("on_mount: push ChatScreen")
        self.push_screen(ChatScreen())

    def get_hyperprompt_editor(self) -> HyperpromptEditorScreen:
        return HyperpromptEditorScreen(self.hyperprompt)