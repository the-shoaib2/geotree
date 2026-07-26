"""
GeoTree — Vegetation Health Analyzer
Classifies vegetation health zones from NDVI thresholds and produces health maps.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from preprocessing.pipeline.helpers import logger, ensure_dir


# Health classification thresholds based on NDVI ranges
HEALTH_ZONES = {
    "dead": {"range": (-1.0, 0.0), "color": "#6e1423", "emoji": "☠️"},
    "stressed": {"range": (0.0, 0.2), "color": "#f85149", "emoji": "🟥"},
    "moderate": {"range": (0.2, 0.4), "color": "#e3b341", "emoji": "🟨"},
    "healthy": {"range": (0.4, 0.6), "color": "#7ee787", "emoji": "🟩"},
    "vigorous": {"range": (0.6, 1.0), "color": "#2ea44f", "emoji": "💚"},
}


class VegetationHealthAnalyzer:
    """Classifies vegetation health from NDVI values into health zones."""

    def __init__(self):
        self.zones = HEALTH_ZONES

    def classify_ndvi(self, ndvi: np.ndarray) -> np.ndarray:
        """Classify each pixel into a health zone ID.

        Zone IDs: 0=dead, 1=stressed, 2=moderate, 3=healthy, 4=vigorous
        """
        health_map = np.zeros_like(ndvi, dtype=np.uint8)
        zone_list = list(self.zones.values())

        for i, zone in enumerate(zone_list):
            lo, hi = zone["range"]
            health_map[(ndvi >= lo) & (ndvi < hi)] = i

        return health_map

    def analyze(self, ndvi: np.ndarray, source_name: str = "image") -> Dict[str, Any]:
        """Perform vegetation health analysis on an NDVI array.

        Returns:
            Dict with health zone stats, overall health score, and recommendations.
        """
        health_map = self.classify_ndvi(ndvi)
        total_pixels = ndvi.size
        total_veg_pixels = int(np.sum(ndvi > 0.1))  # Vegetation pixels only

        zone_stats = []
        zone_names = list(self.zones.keys())

        for i, (name, info) in enumerate(self.zones.items()):
            count = int(np.sum(health_map == i))
            pct = round(count / max(total_pixels, 1) * 100, 2)
            veg_pct = round(count / max(total_veg_pixels, 1) * 100, 2) if total_veg_pixels > 0 else 0.0
            zone_stats.append({
                "zone": name,
                "emoji": info["emoji"],
                "color": info["color"],
                "ndvi_range": list(info["range"]),
                "pixel_count": count,
                "total_pct": pct,
                "vegetation_pct": veg_pct,
            })

        # Overall health score (0-100)
        # Weighted: dead=0, stressed=25, moderate=50, healthy=75, vigorous=100
        weights = [0, 25, 50, 75, 100]
        weighted_sum = sum(zone_stats[i]["pixel_count"] * weights[i] for i in range(5))
        health_score = round(weighted_sum / max(total_veg_pixels, 1), 1) if total_veg_pixels > 0 else 0.0
        health_score = min(100.0, health_score)

        # Health grade
        if health_score >= 80:
            grade = "A — Excellent"
        elif health_score >= 60:
            grade = "B — Good"
        elif health_score >= 40:
            grade = "C — Fair"
        elif health_score >= 20:
            grade = "D — Poor"
        else:
            grade = "F — Critical"

        # NDVI summary
        valid_ndvi = ndvi[np.isfinite(ndvi)]
        ndvi_stats = {
            "mean": round(float(np.mean(valid_ndvi)), 4),
            "std": round(float(np.std(valid_ndvi)), 4),
            "min": round(float(np.min(valid_ndvi)), 4),
            "max": round(float(np.max(valid_ndvi)), 4),
            "median": round(float(np.median(valid_ndvi)), 4),
        }

        result = {
            "source": source_name,
            "total_pixels": total_pixels,
            "vegetation_pixels": total_veg_pixels,
            "vegetation_coverage_pct": round(total_veg_pixels / max(total_pixels, 1) * 100, 2),
            "health_score": health_score,
            "health_grade": grade,
            "ndvi_statistics": ndvi_stats,
            "zone_breakdown": zone_stats,
        }

        logger.info(f"VegetationHealth: {source_name} — Score={health_score}, Grade={grade}")
        return result

    def analyze_from_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze vegetation health from an image (supports GeoTIFF and RGB)."""
        import numpy as np
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            red = arr[:, :, 0]
            green = arr[:, :, 1]
            nir_approx = green * 1.2 + red * 0.1
            ndvi = (nir_approx - red) / (nir_approx + red + 1e-8)
        except Exception:
            # Fallback: rasterio for GeoTIFF
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    if src.count >= 4:
                        red = src.read(1).astype(np.float32) / 10000.0
                        nir = src.read(4).astype(np.float32) / 10000.0
                    else:
                        band = src.read(1).astype(np.float32) / 10000.0
                        red = band
                        nir = band * 1.2
                    ndvi = (nir - red) / (nir + red + 1e-8)
                    ndvi = np.nan_to_num(ndvi, nan=0.0)
            except Exception as e:
                raise ValueError(f"Cannot open image {image_path}: {e}")
        return self.analyze(ndvi, source_name=str(image_path))
