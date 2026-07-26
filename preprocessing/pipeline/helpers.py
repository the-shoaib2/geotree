import logging
from pathlib import Path

# Setup preprocessing logger
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)
# Clear old handlers
logger.handlers = []

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"))
logger.addHandler(ch)

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(log_dir / "preprocessing.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(fh)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
