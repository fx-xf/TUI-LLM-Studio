import torch
from typing import AsyncIterable, Dict, List, Optional
import asyncio
from threading import Thread
import queue
from .model_loader import ModelLoader
from .config import Config

class LLMEngine:
    """PyTorch-движок для генерации текста"""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
        self.model_name = model_name
        self.model_loader = ModelLoader(model_name)
        self.model, self.tokenizer = self.model_loader.load()
        
        # Параметры генерации
        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        hyperprompt: Optional['Hyperprompt'] = None,
        **kwargs
    ) -> AsyncIterable[str]:
        """Асинхронная генерация с потоковой передачей токенов"""
        
        # Формируем промпт
        prompt = self._format_messages(messages, hyperprompt)
        
        # Параметры генерации
        gen_params = {**self.generation_config, **kwargs}
        
        # Очередь для токенов
        token_queue = queue.Queue()
        generation_complete = False
        
        def generate_worker():
            """Рабочий поток для генерации"""
            try:
                # Токенизируем вход
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Генерация с callback'ом
                with torch.no_grad():
                    # Получаем эмбеддинги
                    input_ids = inputs.input_ids
                    
                    # Генерируем по одному токену
                    generated_ids = input_ids.clone()
                    
                    for _ in range(gen_params["max_new_tokens"]):
                        # Получаем logits
                        outputs = self.model(generated_ids)
                        logits = outputs.logits[:, -1, :] / gen_params["temperature"]
                        
                        # Применяем top-k и top-p
                        if gen_params.get("top_k"):
                            logits = self._apply_top_k(logits, gen_params["top_k"])
                        if gen_params.get("top_p"):
                            logits = self._apply_top_p(logits, gen_params["top_p"])
                        
                        # Сэмплируем следующий токен
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Декодируем токен
                        token_text = self.tokenizer.decode(
                            next_token[0], 
                            skip_special_tokens=True
                        )
                        
                        # Добавляем в очередь
                        token_queue.put(token_text)
                        
                        # Обновляем generated_ids
                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        
                        # Проверяем на конец последовательности
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                            
            except Exception as e:
                token_queue.put(f"[Ошибка: {str(e)}]")
            finally:
                generation_complete = True
                token_queue.put(None)  # Сигнал завершения
        
        # Запускаем генерацию в отдельном потоке
        thread = Thread(target=generate_worker, daemon=True)
        thread.start()
        
        # Потоковая передача токенов
        while True:
            try:
                token = token_queue.get(timeout=0.1)
                if token is None:  # Конец генерации
                    break
                yield token
                await asyncio.sleep(0)  # Неблокирующая пауза
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
        
        thread.join()
    
    def _format_messages(self, messages: List[Dict[str, str]], 
                        hyperprompt: Optional['Hyperprompt'] = None) -> str:
        """Форматирует сообщения в промпт DeepSeek Coder"""
        
        # Применяем гиперпромпт если есть
        if hyperprompt and messages:
            last_msg = messages[-1]["content"]
            context = messages[:-1]
            return hyperprompt.generate_prompt(last_msg, context)
        
        # Стандартный формат
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"### Instruction:\n{msg['content']}\n\n### Response:\n"
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']}\n"
        
        return prompt
    
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Применяет top-k фильтрацию"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Применяет top-p (nucleus) фильтрацию"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Удаляем токены с cumulative probability > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def get_memory_info(self):
        """Информация об использовании памяти"""
        return self.model_loader.get_memory_usage()