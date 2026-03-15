import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

LOGS_DIR = str(Path(__file__).resolve().parents[1] / "logs")
MAX_FOLDER_SIZE = 2 * 1024 * 1024
MAX_LOG_SIZE = 5 * 1024 * 1024

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, LOG_FILE)


def cleanup_logs():
    if not os.path.exists(LOGS_DIR):
        return
    files = [os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR) if f.endswith(".log")]
    files.sort(key=os.path.getmtime)
    total_size = sum(os.path.getsize(f) for f in files)
    while total_size > MAX_FOLDER_SIZE and files:
        oldest = files.pop(0)
        size = os.path.getsize(oldest)
        try:
            os.remove(oldest)
            total_size -= size
        except Exception as e:
            logging.error(f"Error deleting log {oldest}: {e}")
            break


def configure_logger():
    cleanup_logs()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=3)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


configure_logger()
logging.info(f"Logger initialized. Logging to {log_file_path}")