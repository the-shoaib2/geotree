"""
GeoTree — Land Cover Statistics Engine
Computes comprehensive land-cover area statistics from segmentation results.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from preprocessing.pipeline.helpers import logger


class LandCoverStatistics:
    """Computes area-based statistics from segmentation pixel counts."""

    def __init__(self, pixel_resolution_m: float = 10.0):
        self.pixel_res = pixel_resolution_m

    def compute(self, segmentation_results: List[Dict[str, Any]],
                area_hectares: Optional[float] = None) -> Dict[str, Any]:
        """Aggregate segmentation results into land-cover statistics.

        Args:
            segmentation_results: List of segmenter output dicts
            area_hectares: Override total area (auto-computed otherwise)

        Returns:
            Comprehensive land-cover statistics
        """
        # Aggregate pixel counts across all tiles
        agg_counts = {}
        total_pixels = 0

        for seg in segmentation_results:
            total_pixels += seg.get("total_pixels", 0)
            for cls_name, count in seg.get("pixel_counts", {}).items():
                agg_counts[cls_name] = agg_counts.get(cls_name, 0) + count

        # Compute percentages
        percentages = {}
        for cls_name, count in agg_counts.items():
            percentages[cls_name] = round(count / max(total_pixels, 1) * 100, 2)

        # Compute area in hectares and km²
        pixel_area_m2 = self.pixel_res ** 2
        areas_m2 = {k: v * pixel_area_m2 for k, v in agg_counts.items()}
        areas_ha = {k: round(v / 10000.0, 2) for k, v in areas_m2.items()}
        areas_km2 = {k: round(v / 1_000_000.0, 4) for k, v in areas_m2.items()}

        total_area_ha = sum(areas_ha.values())
        total_area_km2 = sum(areas_km2.values())

        # Key metrics
        forest_pct = percentages.get("forest", 0) + percentages.get("tree_crown", 0)
        water_pct = percentages.get("water", 0)
        built_up_pct = percentages.get("built_up", 0) + percentages.get("road", 0)
        vegetation_pct = forest_pct + percentages.get("grass", 0) + percentages.get("crop", 0)

        result = {
            "total_pixels": total_pixels,
            "total_area_ha": round(total_area_ha, 2),
            "total_area_km2": round(total_area_km2, 4),
            "pixel_counts": agg_counts,
            "percentages": percentages,
            "areas_hectares": areas_ha,
            "areas_km2": areas_km2,
            "key_metrics": {
                "forest_cover_pct": round(forest_pct, 2),
                "vegetation_cover_pct": round(vegetation_pct, 2),
                "water_coverage_pct": round(water_pct, 2),
                "built_up_pct": round(built_up_pct, 2),
                "bare_soil_pct": round(percentages.get("bare_soil", 0), 2),
                "cloud_pct": round(percentages.get("cloud", 0), 2),
            },
            "num_tiles_analyzed": len(segmentation_results),
        }

        logger.info(f"LandCoverStats: Forest={forest_pct:.1f}%, Water={water_pct:.1f}%, "
                     f"Built-up={built_up_pct:.1f}% over {round(total_area_ha, 2)} ha")
        return result
