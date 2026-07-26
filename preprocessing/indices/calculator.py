import numpy as np
import rasterio
from pathlib import Path
from preprocessing.pipeline.helpers import logger


class IndicesCalculator:
    """Computes multi-spectral vegetation indices from satellite imagery."""

    def calculate_indices(self, tiff_path: Path, output_dir: Path) -> dict:
        """Calculates multi-spectral indices (NDVI, NDWI, NDMI, EVI) and writes them out.

        Returns dict with index paths and statistics.
        """
        results = {}
        try:
            with rasterio.open(tiff_path) as src:
                # Read Sentinel-2 L2A bands
                if src.count >= 4:
                    red = src.read(1).astype(np.float32)
                    green = src.read(2).astype(np.float32)
                    blue = src.read(3).astype(np.float32)
                    nir = src.read(4).astype(np.float32)
                else:
                    nir = green = blue = red = src.read(1).astype(np.float32)

                np.seterr(divide='ignore', invalid='ignore')

                # 1. NDVI (Normalized Difference Vegetation Index)
                ndvi = (nir - red) / (nir + red + 1e-8)
                ndvi = np.nan_to_num(ndvi, nan=0.0)

                # 2. NDWI (Normalized Difference Water Index)
                ndwi = (green - nir) / (green + nir + 1e-8)
                ndwi = np.nan_to_num(ndwi, nan=0.0)

                # 3. NDMI (Normalized Difference Moisture Index)
                ndmi = (nir - blue) / (nir + blue + 1e-8)
                ndmi = np.nan_to_num(ndmi, nan=0.0)

                # 4. EVI (Enhanced Vegetation Index)
                # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
                evi_denom = nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8
                evi = 2.5 * (nir - red) / evi_denom
                evi = np.clip(np.nan_to_num(evi, nan=0.0), -1.0, 1.0)

                # Save calculated rasters
                meta = src.meta.copy()
                meta.update(dtype=rasterio.float32, count=1)

                indices_map = {"ndvi": ndvi, "ndwi": ndwi, "ndmi": ndmi, "evi": evi}
                output_dir.mkdir(parents=True, exist_ok=True)

                for name, arr in indices_map.items():
                    out_path = output_dir / f"{tiff_path.stem}_{name}.tif"
                    with rasterio.open(out_path, "w", **meta) as dst:
                        dst.write(arr, 1)

                    # Compute statistics for each index
                    valid = arr[np.isfinite(arr)]
                    stats = {
                        "path": str(out_path),
                        "mean": round(float(np.mean(valid)), 4) if len(valid) > 0 else 0.0,
                        "std": round(float(np.std(valid)), 4) if len(valid) > 0 else 0.0,
                        "min": round(float(np.min(valid)), 4) if len(valid) > 0 else 0.0,
                        "max": round(float(np.max(valid)), 4) if len(valid) > 0 else 0.0,
                        "median": round(float(np.median(valid)), 4) if len(valid) > 0 else 0.0,
                    }
                    results[name] = stats

            logger.info(f"Successfully calculated satellite indices for {tiff_path.name}")
        except Exception as e:
            logger.error(f"Failed to calculate indices for {tiff_path.name}: {e}")

        return results

    def calculate_from_arrays(self, red: np.ndarray, green: np.ndarray,
                              blue: np.ndarray, nir: np.ndarray) -> dict:
        """Calculate indices directly from numpy band arrays. Returns index arrays and stats."""
        np.seterr(divide='ignore', invalid='ignore')

        ndvi = np.nan_to_num((nir - red) / (nir + red + 1e-8), nan=0.0)
        ndwi = np.nan_to_num((green - nir) / (green + nir + 1e-8), nan=0.0)
        ndmi = np.nan_to_num((nir - blue) / (nir + blue + 1e-8), nan=0.0)
        evi = np.clip(np.nan_to_num(
            2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8), nan=0.0
        ), -1.0, 1.0)

        indices = {"ndvi": ndvi, "ndwi": ndwi, "ndmi": ndmi, "evi": evi}
        stats = {}
        for name, arr in indices.items():
            valid = arr[np.isfinite(arr)]
            stats[name] = {
                "mean": round(float(np.mean(valid)), 4) if len(valid) > 0 else 0.0,
                "std": round(float(np.std(valid)), 4) if len(valid) > 0 else 0.0,
                "min": round(float(np.min(valid)), 4) if len(valid) > 0 else 0.0,
                "max": round(float(np.max(valid)), 4) if len(valid) > 0 else 0.0,
            }

        return {"arrays": indices, "statistics": stats}


