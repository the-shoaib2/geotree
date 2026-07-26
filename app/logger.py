import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_dir: str = "logs", log_file: str = "daily.log") -> logging.Logger:
    """Sets up a logger that logs to both the console and a daily rotating file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("sentinel_downloader")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs in case of re-import
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (rotating at 10MB, keeping 7 backups)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Shared logger instance
logger = setup_logger()
