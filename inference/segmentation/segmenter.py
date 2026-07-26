"""
GeoTree — Land Cover Segmentation Engine
Performs pixel-level classification of satellite imagery into land-cover categories
using spectral index thresholds and vegetation analysis.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from preprocessing.pipeline.helpers import logger, ensure_dir

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# Land cover class definitions with spectral thresholds
LAND_COVER_CLASSES = {
    0: {"name": "background", "color": "#000000", "label": "Background"},
    1: {"name": "tree_crown", "color": "#2ea44f", "label": "🌲 Tree Crown"},
    2: {"name": "forest", "color": "#196127", "label": "🌿 Forest"},
    3: {"name": "grass", "color": "#7ee787", "label": "🌱 Grass"},
    4: {"name": "crop", "color": "#d4a373", "label": "🌾 Crop"},
    5: {"name": "water", "color": "#58a6ff", "label": "💧 Water"},
    6: {"name": "bare_soil", "color": "#d29922", "label": "🪨 Bare Soil"},
    7: {"name": "built_up", "color": "#f85149", "label": "🏠 Built-up"},
    8: {"name": "road", "color": "#8b949e", "label": "🛣️ Road"},
    9: {"name": "cloud", "color": "#f0f0f0", "label": "☁️ Cloud"},
    10: {"name": "shadow", "color": "#484f58", "label": "🌑 Shadow"},
}


class LandCoverSegmenter:
    """Spectral-index-based land cover segmentation for satellite imagery.
    
    Uses NDVI, NDWI, and brightness thresholds to classify each pixel into
    one of the land cover classes. This serves as a production baseline
    segmentation engine that works without requiring a dedicated segmentation
    model — it can be upgraded to a U-Net or DeepLabV3 when training data
    is available.
    """

    def __init__(self):
        self.classes = LAND_COVER_CLASSES

    def segment_from_bands(self, red: np.ndarray, green: np.ndarray,
                           blue: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Classify each pixel using spectral indices.

        Args:
            red, green, blue, nir: 2D float arrays of band reflectances

        Returns:
            2D int array of class IDs matching LAND_COVER_CLASSES
        """
        h, w = red.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Compute spectral indices
        eps = 1e-8
        ndvi = (nir - red) / (nir + red + eps)
        ndwi = (green - nir) / (green + nir + eps)
        brightness = (red + green + blue) / 3.0
        nir_ratio = nir / (red + green + blue + eps) * 3.0

        # Classification hierarchy (order matters — later overwrites earlier)
        # 1. Bare soil: low NDVI, moderate brightness
        mask[(ndvi < 0.2) & (brightness > 0.15) & (brightness < 0.55)] = 6  # bare_soil

        # 2. Built-up: low NDVI, high brightness, low NIR ratio
        mask[(ndvi < 0.15) & (brightness > 0.3) & (nir_ratio < 0.8)] = 7  # built_up

        # 3. Road: very low NDVI, moderate-high brightness, very low NIR
        mask[(ndvi < 0.1) & (brightness > 0.25) & (brightness < 0.65) & (nir_ratio < 0.6)] = 8  # road

        # 4. Water: positive NDWI or very low brightness with low NDVI
        mask[(ndwi > 0.0) & (ndvi < 0.2)] = 5  # water
        mask[(brightness < 0.08) & (ndvi < 0.1)] = 5  # dark water

        # 5. Grass: moderate NDVI
        mask[(ndvi >= 0.2) & (ndvi < 0.4)] = 3  # grass

        # 6. Crop: moderate NDVI with distinct pattern
        mask[(ndvi >= 0.25) & (ndvi < 0.45) & (nir_ratio > 1.0)] = 4  # crop

        # 7. Forest: high NDVI, dense canopy
        mask[(ndvi >= 0.4) & (ndvi < 0.65)] = 2  # forest

        # 8. Dense tree crowns: very high NDVI
        mask[ndvi >= 0.65] = 1  # tree_crown

        # 9. Cloud: very high brightness across all bands
        mask[(brightness > 0.7) & (ndvi < 0.15)] = 9  # cloud

        # 10. Shadow: very low brightness with low NDVI
        mask[(brightness < 0.05)] = 10  # shadow

        return mask

    def segment_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Segment a multi-band satellite image into land cover classes.

        Args:
            image_path: Path to a multi-band GeoTIFF (≥4 bands: R, G, B, NIR)
            output_dir: If provided, saves classification mask as GeoTIFF

        Returns:
            Dict with class_mask, pixel_counts, percentages, and class info
        """
        path = Path(image_path)

        # Try rasterio first (handles GeoTIFFs with any extension)
        if HAS_RASTERIO:
            try:
                import rasterio as _rio
                with _rio.open(path) as _src:
                    pass  # valid rasterio file
                return self._segment_geotiff(path, output_dir)
            except Exception:
                pass
        # Fallback to PIL RGB
        return self._segment_rgb_image(path, output_dir)

    def _segment_geotiff(self, tiff_path: Path, output_dir: Optional[str]) -> Dict[str, Any]:
        """Segment a GeoTIFF with at least 4 bands."""
        with rasterio.open(tiff_path) as src:
            if src.count >= 4:
                red = src.read(1).astype(np.float32) / 10000.0  # Sentinel-2 scale
                green = src.read(2).astype(np.float32) / 10000.0
                blue = src.read(3).astype(np.float32) / 10000.0
                nir = src.read(4).astype(np.float32) / 10000.0
            else:
                # Grayscale fallback
                band = src.read(1).astype(np.float32) / 10000.0
                red = green = blue = nir = band

            mask = self.segment_from_bands(red, green, blue, nir)
            result = self._compute_statistics(mask, tiff_path)

            # Save classification mask
            if output_dir:
                out_path = Path(output_dir)
                ensure_dir(out_path)
                mask_path = out_path / f"{tiff_path.stem}_landcover.tif"
                meta = src.meta.copy()
                meta.update(dtype=rasterio.uint8, count=1, nodata=0)
                with rasterio.open(mask_path, "w", **meta) as dst:
                    dst.write(mask, 1)
                result["mask_path"] = str(mask_path)
                logger.info(f"Segmentation mask saved to {mask_path}")

            return result

    def _segment_rgb_image(self, img_path: Path, output_dir: Optional[str]) -> Dict[str, Any]:
        """Segment an RGB image using synthetic NIR approximation."""
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0

        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        # Approximate NIR from visible bands (green channel as proxy)
        nir = green * 1.2 + red * 0.1

        mask = self.segment_from_bands(red, green, blue, nir)
        return self._compute_statistics(mask, img_path)

    def _compute_statistics(self, mask: np.ndarray, source_path: Path) -> Dict[str, Any]:
        """Compute per-class pixel counts and percentages from classification mask."""
        total_pixels = mask.size
        pixel_counts = {}
        percentages = {}
        class_details = []

        for class_id, info in self.classes.items():
            count = int(np.sum(mask == class_id))
            pct = round(count / max(total_pixels, 1) * 100, 2)
            pixel_counts[info["name"]] = count
            percentages[info["name"]] = pct
            if count > 0:
                class_details.append({
                    "class_id": class_id,
                    "class_name": info["name"],
                    "label": info["label"],
                    "color": info["color"],
                    "pixel_count": count,
                    "percentage": pct,
                })

        return {
            "source": str(source_path),
            "image_shape": list(mask.shape),
            "total_pixels": total_pixels,
            "pixel_counts": pixel_counts,
            "percentages": percentages,
            "class_details": sorted(class_details, key=lambda x: x["pixel_count"], reverse=True),
            "num_classes_present": len(class_details),
        }

    def get_class_info(self) -> Dict[int, Dict]:
        """Return the full class taxonomy."""
        return self.classes
