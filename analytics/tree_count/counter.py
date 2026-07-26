"""
GeoTree — Tree Counter
Counts detected trees and computes spatial density metrics.
"""
import math
from typing import Dict, Any, List, Optional
from preprocessing.pipeline.helpers import logger


class TreeCounter:
    """Counts trees from detection results and computes density metrics."""

    def __init__(self, pixel_resolution_m: float = 10.0):
        """
        Args:
            pixel_resolution_m: Ground sampling distance in meters per pixel.
                                Default 10m for Sentinel-2.
        """
        self.pixel_res = pixel_resolution_m

    def count_from_detections(self, detection_results: List[Dict[str, Any]],
                              image_size_px: int = 640,
                              area_hectares: Optional[float] = None) -> Dict[str, Any]:
        """Count trees from detection engine output.

        Args:
            detection_results: List of per-image result dicts from TreeDetector
            image_size_px: Image tile dimension in pixels
            area_hectares: Total surveyed area in hectares (auto-computed if not provided)

        Returns:
            Tree count summary with density metrics
        """
        total_trees = 0
        per_tile_counts = []
        all_confidences = []
        all_crown_areas_px = []

        for result in detection_results:
            tile_count = result.get("num_detections", 0)
            total_trees += tile_count
            per_tile_counts.append(tile_count)

            for det in result.get("detections", []):
                all_confidences.append(det.get("confidence", 0))
                all_crown_areas_px.append(det.get("crown_area_px", 0))

        num_tiles = len(detection_results)

        # Compute area if not provided
        # Each tile covers (image_size_px * pixel_res)² square meters
        tile_area_m2 = (image_size_px * self.pixel_res) ** 2
        total_area_m2 = tile_area_m2 * num_tiles
        total_area_ha = total_area_m2 / 10000.0  # 1 ha = 10000 m²
        total_area_km2 = total_area_m2 / 1_000_000.0

        if area_hectares:
            total_area_ha = area_hectares
            total_area_km2 = area_hectares / 100.0

        # Density metrics
        trees_per_ha = round(total_trees / max(total_area_ha, 0.001), 2)
        trees_per_km2 = round(total_trees / max(total_area_km2, 0.000001), 0)

        # Crown area statistics
        avg_crown_area_px = sum(all_crown_areas_px) / max(len(all_crown_areas_px), 1)
        avg_crown_area_m2 = avg_crown_area_px * (self.pixel_res ** 2)

        # Confidence statistics
        avg_conf = sum(all_confidences) / max(len(all_confidences), 1)

        result = {
            "total_tree_count": total_trees,
            "tiles_processed": num_tiles,
            "trees_per_tile_avg": round(total_trees / max(num_tiles, 1), 2),
            "trees_per_hectare": trees_per_ha,
            "trees_per_km2": int(trees_per_km2),
            "survey_area": {
                "hectares": round(total_area_ha, 2),
                "km2": round(total_area_km2, 4),
                "m2": round(total_area_m2, 0),
            },
            "crown_statistics": {
                "avg_crown_area_px": round(avg_crown_area_px, 1),
                "avg_crown_area_m2": round(avg_crown_area_m2, 2),
            },
            "confidence": {
                "mean": round(avg_conf, 4),
                "min": round(min(all_confidences), 4) if all_confidences else 0.0,
                "max": round(max(all_confidences), 4) if all_confidences else 0.0,
            },
            "per_tile_counts": per_tile_counts,
        }

        logger.info(f"TreeCounter: {total_trees} trees detected, {trees_per_ha} trees/ha over {round(total_area_ha, 2)} ha")
        return result
