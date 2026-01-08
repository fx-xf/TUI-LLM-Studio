import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from pathlib import Path
from typing import Optional
import gc

class ModelLoader:
    """Загружает модель с квантизацией для экономии памяти"""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct", 
                 load_in_4bit: bool = True):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Загружает модель и токенизатор"""
        print(f"🤖 Загрузка модели: {self.model_name}...")
        
        # Настройки квантизации
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Модель загружена!")
        
        # Очистка памяти
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.model, self.tokenizer
    
    def get_memory_usage(self):
        """Возвращает использование памяти GPU"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,      # GB
            }
        return {"allocated": 0, "cached": 0}