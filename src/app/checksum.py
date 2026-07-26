import hashlib
from pathlib import Path
from app.logger import logger

def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA-256 hash of a file using chunk-by-chunk stream processing."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating SHA-256 for {file_path}: {e}")
        raise
