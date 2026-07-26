import zipfile
import tarfile
import shutil
from pathlib import Path
from preprocessing.pipeline.helpers import logger

class ArchiveExtractor:
    def extract_all(self, archive_path: Path, output_dir: Path) -> bool:
        """Extracts ZIP, TAR, TGZ files automatically."""
        if not archive_path.exists():
            logger.error(f"Archive not found: {archive_path}")
            return False
            
        suffix = archive_path.suffix.lower()
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(output_dir)
                logger.info(f"Successfully extracted zip: {archive_path.name}")
                return True
            elif suffix in [".tar", ".gz", ".tgz"]:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_ref.extractall(output_dir)
                logger.info(f"Successfully extracted tar archive: {archive_path.name}")
                return True
            else:
                logger.warning(f"Unsupported archive suffix '{suffix}' for {archive_path.name}. Copying directly.")
                shutil.copy(archive_path, output_dir / archive_path.name)
                return True
        except Exception as e:
            logger.error(f"Failed to extract {archive_path.name}: {e}")
            return False
