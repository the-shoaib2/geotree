"""
GeoTree — Tree Density Mapper
Generates spatial density heatmaps from detection coordinates.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from preprocessing.pipeline.helpers import logger, ensure_dir


class TreeDensityMapper:
    """Generates tree density heatmaps from detection spatial coordinates."""

    def __init__(self, grid_size: int = 50):
        """
        Args:
            grid_size: Number of grid cells per axis for density computation
        """
        self.grid_size = grid_size

    def compute_density_grid(self, detection_results: List[Dict[str, Any]],
                             image_size_px: int = 640,
                             pixel_resolution_m: float = 10.0) -> Dict[str, Any]:
        """Compute a spatial density grid from detection coordinates.

        Args:
            detection_results: List of per-image results from TreeDetector
            image_size_px: Tile size in pixels
            pixel_resolution_m: Meters per pixel

        Returns:
            Dict with density_grid array, statistics, and hotspots
        """
        # Collect all tree center coordinates (normalized 0-1)
        centers = []
        for result in detection_results:
            for det in result.get("detections", []):
                bbox = det.get("bbox_xywh", [0.5, 0.5, 0, 0])
                centers.append((bbox[0], bbox[1]))

        if not centers:
            logger.warning("TreeDensityMapper: No detections to map")
            return {
                "density_grid": np.zeros((self.grid_size, self.grid_size)).tolist(),
                "total_trees": 0,
                "hotspots": [],
                "statistics": {},
            }

        # Build 2D histogram
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        density, x_edges, y_edges = np.histogram2d(
            xs, ys, bins=self.grid_size, range=[[0, 1], [0, 1]]
        )

        # Convert to density per hectare
        cell_width_m = (image_size_px * pixel_resolution_m) / self.grid_size
        cell_area_ha = (cell_width_m ** 2) / 10000.0
        density_per_ha = density / max(cell_area_ha, 0.001)

        # Find hotspots (top 5 densest cells)
        flat_indices = np.argsort(density.flatten())[::-1][:5]
        hotspots = []
        for idx in flat_indices:
            row, col = divmod(idx, self.grid_size)
            if density[row, col] > 0:
                hotspots.append({
                    "grid_cell": [int(row), int(col)],
                    "tree_count": int(density[row, col]),
                    "density_per_ha": round(float(density_per_ha[row, col]), 1),
                })

        result = {
            "density_grid": density.tolist(),
            "density_per_ha_grid": density_per_ha.tolist(),
            "grid_size": self.grid_size,
            "total_trees": len(centers),
            "hotspots": hotspots,
            "statistics": {
                "mean_density_per_ha": round(float(np.mean(density_per_ha)), 2),
                "max_density_per_ha": round(float(np.max(density_per_ha)), 2),
                "std_density_per_ha": round(float(np.std(density_per_ha)), 2),
                "cells_with_trees": int(np.sum(density > 0)),
                "cells_total": self.grid_size ** 2,
                "spatial_coverage_pct": round(float(np.sum(density > 0) / (self.grid_size ** 2) * 100), 1),
            },
        }

        logger.info(f"TreeDensityMapper: {len(centers)} trees mapped to {self.grid_size}x{self.grid_size} grid")
        return result
