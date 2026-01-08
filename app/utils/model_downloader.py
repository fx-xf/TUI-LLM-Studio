import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model(model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
    """Скачивает модель в кэш"""
    print(f"📥 Скачивание модели: {model_name}")
    
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_files_only=False,
        resume_download=True
    )
    
    print("✅ Модель скачана в кэш HuggingFace")

if __name__ == "__main__":
    download_model()