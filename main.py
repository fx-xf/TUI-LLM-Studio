#!/usr/bin/env python3
import sys
import traceback
import logging
import torch
from pathlib import Path

# --------------------- 1.  Логирование --------------------- #
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("llm-tui.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("main")

# --------------------- 2.  Точка входа --------------------- #
def main() -> None:
    log.info("===  LLM-TUI  запускается  ===")
    try:
        log.info("Импорт модулей...")
        from app.app import LLMTUIApp
        from app.core.config import Config
        from app.utils.model_downloader import download_model

        log.info("Инициализация конфига...")
        Config.initialize()

        if torch.cuda.is_available():
            log.info("🚀 GPU обнаружен: %s", torch.cuda.get_device_name())
            log.info("📊 Доступно памяти: %.1f ГБ",
                     torch.cuda.get_device_properties(0).total_memory / 1024**3)
        else:
            log.info("⚠️ GPU не обнаружен, будет использоваться CPU (медленно)")

        log.info("Скачиваем модель при необходимости...")
        download_model()

        log.info("Создаём LLMTUIApp...")
        app = LLMTUIApp()
        log.info("LLMTUIApp создан успешно")

        log.info("Вызываем app.run()...")
        app.run()
        log.info("app.run() вернул управление")

    except Exception as exc:
        log.exception("КРИТИЧЕСКАЯ ОШИБКА – приложение упало")
        traceback.print_exc(file=sys.stdout)
        input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    main()