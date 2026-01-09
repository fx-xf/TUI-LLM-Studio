# model/inference.py
import torch
from transformers import AutoTokenizer
from typing import List, Dict, Generator
import random

class ModelInference:
    """Класс для инференса модели"""
    
    def __init__(self, model, metadata: Dict, device: str = None):
        self.model = model
        self.metadata = metadata
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка токенизатора
        print("🔤 Загрузка токенизатора...")
        
        # Проверяем, если это dummy модель
        if metadata.get("is_dummy"):
            self.tokenizer = DummyTokenizer()
        else:
            try:
                tokenizer_path = self.metadata.get("tokenizer_path", "deepseek-ai/deepseek-coder-6.7b-instruct")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            except:
                # Загрузка базового токенизатора Llama
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Токенизатор загружен!")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: List[str] = None,
        stream: bool = True
    ) -> Generator[str, None, None]:
        """
        Генерация текста с параметрами семплинга
        """
        if self.metadata.get("is_dummy"):
            # Тестовая генерация для dummy модели
            responses = [
                "Привет! Это тестовый ответ от dummy модели.",
                "Я понимаю ваш запрос, но пока работаю в тестовом режиме.",
                "Для полноценной работы нужно подключить реальную модель.",
                "Попробуйте установить transformers и bitsandbytes.",
            ]
            
            response = random.choice(responses)
            if stream:
                for char in response:
                    yield char
            else:
                yield response
            return
        
        # Здесь будет реальная генерация для настоящей модели
        # Пока используем упрощенную реализацию
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            input_ids = inputs["input_ids"].to(self.device)
            
            # Генерация через модель
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=min(max_tokens, 512),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Декодируем только новые токены
            new_tokens = outputs[0][input_ids.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            if stream:
                for char in response:
                    yield char
            else:
                yield response
                
        except Exception as e:
            # Запасной вариант - простой ответ
            response = f"Извините, произошла ошибка при генерации: {str(e)}"
            if stream:
                for char in response:
                    yield char
            else:
                yield response

class DummyTokenizer:
    """Токенизатор-заглушка для тестирования"""
    
    def __init__(self):
        self.vocab_size = 32000
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    
    def encode(self, text: str, add_special_tokens: bool = True):
        # Простая хэш-функция для токенизации
        return [hash(word) % self.vocab_size for word in text.split()]
    
    def decode(self, token_ids, skip_special_tokens: bool = True):
        # Возвращаем просто слова обратно
        return " ".join([f"token_{tid}" for tid in token_ids if tid < 100])
    
    def __call__(self, text: str, return_tensors=None, return_attention_mask=False):
        tokens = self.encode(text)
        return {"input_ids": torch.tensor([tokens])}