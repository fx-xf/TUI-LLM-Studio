# prompts/hyperprompt.py
from dataclasses import dataclass
from typing import Dict, List
from jinja2 import Template

@dataclass
class HyperPrompt:
    """Класс для управления гиперпромптами"""
    system_template: str
    user_template: str = "{{ message }}"
    assistant_template: str = "{{ message }}"
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
    
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Форматирует полный промпт из истории сообщений
        """
        formatted = ""
        
        # System prompt
        if messages and messages[0]["role"] == "system":
            sys_msg = messages[0]["content"]
            formatted += self.system_template.replace("{{ system_message }}", sys_msg)
            messages = messages[1:]
        
        # Добавление истории сообщений
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                formatted += self.user_template.replace("{{ message }}", content)
            elif role == "assistant":
                formatted += self.assistant_template.replace("{{ message }}", content)
        
        return formatted

# Предустановленные шаблоны
HYPERPROMPT_TEMPLATES = {
    "default": HyperPrompt(
        system_template="""<|im_start|>system
{{ system_message }}<|im_end|>
""",
        user_template="""<|im_start|>user
{{ message }}<|im_end|>
""",
        assistant_template="""<|im_start|>assistant
{{ message }}<|im_end|>
""",
        stop_sequences=["<|im_end|>", "<|endoftext|>"]
    ),
    
    "deepseek": HyperPrompt(
        system_template="""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
""",
        user_template="""### Instruction:
{{ message }}

### Response:
""",
        assistant_template="""{{ message }}

""",
        stop_sequences=["###", "<|endoftext|>"]
    ),
    
    "simple": HyperPrompt(
        system_template="""System: {{ system_message }}

""",
        user_template="""User: {{ message }}

""",
        assistant_template="""Assistant: {{ message }}

""",
        stop_sequences=["System:", "User:", "Assistant:"]
    )
}

def get_hyperprompt(template_name: str = "default") -> HyperPrompt:
    """Получает гиперпромпт по имени шаблона"""
    return HYPERPROMPT_TEMPLATES.get(template_name, HYPERPROMPT_TEMPLATES["default"])