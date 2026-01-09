# model/loader.py
import torch
from pathlib import Path
from typing import Dict, Any
import json

def load_model_simplified(model_path: Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Упрощенная загрузка модели через transformers
    Вместо GGUF используем прямую загрузку из HuggingFace
    """
    print(f"🚀 Загрузка модели через transformers...")
    
    try:
        # Попробуем загрузить как обычную модель transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Для начала используем модель из HuggingFace
        model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
        
        print(f"📦 Загрузка {model_name}...")
        
        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Загружаем модель в 4-bit квантизации для экономии памяти
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # Метаданные
        metadata = {
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "num_kv_heads": getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
            "max_seq_len": getattr(model.config, 'max_position_embeddings', 4096),
            "norm_eps": getattr(model.config, 'rms_norm_eps', 1e-6),
            "model_type": model.config.model_type,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        }
        
        print(f"✅ Модель загружена успешно!")
        print(f"   Архитектура: {metadata['model_type']}")
        print(f"   Параметры: {metadata['hidden_size']}")
        print(f"   Слои: {metadata['num_layers']}")
        
        return model, metadata
        
    except Exception as e:
        print(f"❌ Ошибка загрузки через transformers: {e}")
        print("Попробуйте альтернативный метод...")
        
        # Альтернативный метод - используем библиотеку bitsandbytes
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            metadata = {
                "vocab_size": tokenizer.vocab_size,
                "hidden_size": model.config.hidden_size,
                "intermediate_size": model.config.intermediate_size,
                "num_layers": model.config.num_hidden_layers,
                "num_heads": model.config.num_attention_heads,
                "num_kv_heads": getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
                "max_seq_len": getattr(model.config, 'max_position_embeddings', 4096),
                "norm_eps": getattr(model.config, 'rms_norm_eps', 1e-6),
                "model_type": model.config.model_type,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
            }
            
            return model, metadata
            
        except Exception as e2:
            print(f"❌ Альтернативный метод тоже не работает: {e2}")
            raise e2

def create_dummy_model(device: str = "cpu"):
    """
    Создает заглушку модели для тестирования без реальной загрузки
    """
    print("⚠️  Используется тестовая модель-заглушка")
    
    class DummyModel:
        def __init__(self):
            self.config = type('Config', (), {
                'vocab_size': 32000,
                'hidden_size': 4096,
                'intermediate_size': 11008,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'num_key_value_heads': 32,
                'max_position_embeddings': 4096,
                'rms_norm_eps': 1e-6,
                'model_type': 'llama',
                'pad_token_id': 0,
                'bos_token_id': 1,
                'eos_token_id': 2,
            })()
        
        def generate(self, *args, **kwargs):
            # Возвращаем dummy тензор
            import torch
            batch_size = kwargs.get('input_ids', torch.randn(1, 10)).shape[0]
            return torch.randn(batch_size, 50)  # dummy output
        
        def to(self, device):
            return self
        
        def eval(self):
            return self
    
    model = DummyModel()
    
    metadata = {
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "num_kv_heads": getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
        "max_seq_len": getattr(model.config, 'max_position_embeddings', 4096),
        "norm_eps": getattr(model.config, 'rms_norm_eps', 1e-6),
        "model_type": model.config.model_type,
        "pad_token_id": model.config.pad_token_id,
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "is_dummy": True,
    }
    
    return model, metadata


__all__ = ['load_model_simplified', 'create_dummy_model']