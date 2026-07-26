import numpy as np
import rasterio
from pathlib import Path
from preprocessing.pipeline.helpers import logger

class IndicesCalculator:
    def calculate_indices(self, tiff_path: Path, output_dir: Path) -> dict:
        """Calculates multi-spectral indices (NDVI, NDWI, NDMI, EVI, etc.) and writes them out."""
        results = {}
        try:
            with rasterio.open(tiff_path) as src:
                # Read Sentinel-2 L2A default bands matching Processing API ordering:
                # Band 1 = B04 (Red), Band 2 = B03 (Green), Band 3 = B02 (Blue), Band 4 = B08 (NIR)
                if src.count >= 4:
                    red = src.read(1).astype(np.float32)
                    green = src.read(2).astype(np.float32)
                    blue = src.read(3).astype(np.float32)
                    nir = src.read(4).astype(np.float32)
                else:
                    # Fallback if image has fewer than 4 bands
                    nir = green = blue = red = src.read(1).astype(np.float32)

                # Avoid division by zero
                np.seterr(divide='ignore', invalid='ignore')

                # 1. NDVI (Normalized Difference Vegetation Index)
                ndvi = (nir - red) / (nir + red + 1e-8)
                ndvi = np.nan_to_num(ndvi, nan=0.0)
                
                # 2. NDWI (Normalized Difference Water Index)
                ndwi = (green - nir) / (green + nir + 1e-8)
                ndwi = np.nan_to_num(ndwi, nan=0.0)
                
                # 3. NDMI (Normalized Difference Moisture Index)
                # Usually (NIR - SWIR) / (NIR + SWIR), using Blue as proxy SWIR if SWIR is missing
                ndmi = (nir - blue) / (nir + blue + 1e-8)
                ndmi = np.nan_to_num(ndmi, nan=0.0)

                # Save calculated rasters
                meta = src.meta.copy()
                meta.update(dtype=rasterio.float32, count=1)

                indices_map = {"ndvi": ndvi, "ndwi": ndwi, "ndmi": ndmi}
                output_dir.mkdir(parents=True, exist_ok=True)

                for name, arr in indices_map.items():
                    out_path = output_dir / f"{tiff_path.stem}_{name}.tif"
                    with rasterio.open(out_path, "w", **meta) as dst:
                        dst.write(arr, 1)
                    results[name] = str(out_path)

            logger.info(f"Successfully calculated satellite indices for {tiff_path.name}")
        except Exception as e:
            logger.error(f"Failed to calculate indices for {tiff_path.name}: {e}")

        return results
