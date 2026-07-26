import os
import rasterio
from pathlib import Path
from typing import Dict, Any, List
from preprocessing.pipeline.helpers import logger

class DataVerifier:
    def __init__(self, crs: str = "EPSG:4326"):
        self.target_crs = crs

    def verify_dataset_item(self, filepath: Path) -> Dict[str, Any]:
        """Verifies if geospatial file exists, is readable, not corrupted, and checks CRS/Projection."""
        report = {
            "file": filepath.name,
            "exists": False,
            "readable": False,
            "corrupted": True,
            "crs": None,
            "projection": None,
            "errors": []
        }
        
        if not filepath.exists():
            report["errors"].append("File does not exist")
            logger.error(f"Verification failed for {filepath.name}: {report['errors']}")
            return report
            
        report["exists"] = True
        
        # Check readability and corruption using rasterio
        try:
            with rasterio.open(filepath) as src:
                report["readable"] = True
                report["corrupted"] = False
                report["crs"] = str(src.crs)
                report["projection"] = src.driver
                
                # Check CRS alignment
                if src.crs and str(src.crs).upper() != self.target_crs.upper():
                    report["errors"].append(f"CRS Mismatch: found {src.crs}, expected {self.target_crs}")
                    logger.warning(f"CRS Mismatch in {filepath.name}: {src.crs}")
        except Exception as e:
            report["errors"].append(f"Corruption or Read Error: {e}")
            logger.error(f"Read error for {filepath.name}: {e}")
            
        return report

    def verify_all_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        results = []
        for path in file_paths:
            results.append(self.verify_dataset_item(path))
        return results
