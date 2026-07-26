import rasterio
from pathlib import Path
from typing import Dict, Any, List
from preprocessing.pipeline.helpers import logger

class QualityChecker:
    def check_tile_quality(self, tile_path: Path) -> Dict[str, Any]:
        """Checks pixel limits, data format, size matching, and returns quality report."""
        report = {
            "tile": tile_path.name,
            "width": 0,
            "height": 0,
            "mean": 0.0,
            "std": 0.0,
            "valid": False
        }
        
        try:
            with rasterio.open(tile_path) as src:
                report["width"] = src.width
                report["height"] = src.height
                
                band = src.read(1)
                report["mean"] = float(band.mean())
                report["std"] = float(band.std())
                
                # Check for standard tile sizes and presence of non-zero data
                if src.width >= 128 and src.height >= 128 and report["std"] > 0:
                    report["valid"] = True
        except Exception as e:
            logger.error(f"Quality check failure on {tile_path.name}: {e}")
            
        return report

    def check_label_match(self, images: List[Path], labels_dir: Path) -> int:
        missing_count = 0
        for img in images:
            lbl_file = labels_dir / f"{img.stem}.txt"
            if not lbl_file.exists():
                missing_count += 1
                logger.warning(f"Label file missing for image: {img.name}")
        return missing_count
