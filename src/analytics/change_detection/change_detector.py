"""
GeoTree — Forest Change Detector
Compares NDVI rasters from two dates to detect forest gain/loss.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from preprocessing.pipeline.helpers import logger, ensure_dir


class ForestChangeDetector:
    """Detects forest gain and loss by comparing NDVI rasters from two dates."""

    def __init__(self, gain_threshold: float = 0.15, loss_threshold: float = -0.15):
        """
        Args:
            gain_threshold: Minimum NDVI increase to classify as forest gain
            loss_threshold: Maximum NDVI decrease (negative) to classify as forest loss
        """
        self.gain_threshold = gain_threshold
        self.loss_threshold = loss_threshold

    def detect_change(self, ndvi_before: np.ndarray, ndvi_after: np.ndarray,
                      date_before: str = "T1", date_after: str = "T2") -> Dict[str, Any]:
        """Compare two NDVI rasters and compute forest gain/loss.

        Args:
            ndvi_before: 2D NDVI array from earlier date
            ndvi_after: 2D NDVI array from later date
            date_before: Label for the earlier date
            date_after: Label for the later date

        Returns:
            Dict with change map, gain/loss statistics, and burned area estimate
        """
        # Compute NDVI difference
        delta = ndvi_after - ndvi_before
        total_pixels = delta.size

        # Classification
        gain_mask = delta >= self.gain_threshold
        loss_mask = delta <= self.loss_threshold
        stable_mask = (~gain_mask) & (~loss_mask)

        # Burned area detection (severe NDVI drop + low post-fire NDVI)
        burned_mask = (delta <= -0.3) & (ndvi_after < 0.15)

        # Pixel counts
        gain_pixels = int(np.sum(gain_mask))
        loss_pixels = int(np.sum(loss_mask))
        stable_pixels = int(np.sum(stable_mask))
        burned_pixels = int(np.sum(burned_mask))

        # Percentages
        gain_pct = round(gain_pixels / max(total_pixels, 1) * 100, 2)
        loss_pct = round(loss_pixels / max(total_pixels, 1) * 100, 2)
        stable_pct = round(stable_pixels / max(total_pixels, 1) * 100, 2)
        burned_pct = round(burned_pixels / max(total_pixels, 1) * 100, 2)

        # NDVI change statistics
        valid_delta = delta[np.isfinite(delta)]
        delta_stats = {
            "mean": round(float(np.mean(valid_delta)), 4),
            "std": round(float(np.std(valid_delta)), 4),
            "min": round(float(np.min(valid_delta)), 4),
            "max": round(float(np.max(valid_delta)), 4),
        }

        # Net change direction
        if gain_pct > loss_pct + 5:
            trend = "📈 Net Forest Gain"
        elif loss_pct > gain_pct + 5:
            trend = "📉 Net Forest Loss"
        else:
            trend = "➡️ Stable"

        result = {
            "date_before": date_before,
            "date_after": date_after,
            "total_pixels": total_pixels,
            "trend": trend,
            "forest_gain": {
                "pixels": gain_pixels,
                "percentage": gain_pct,
                "threshold": self.gain_threshold,
                "color": "#2ea44f",
            },
            "forest_loss": {
                "pixels": loss_pixels,
                "percentage": loss_pct,
                "threshold": self.loss_threshold,
                "color": "#f85149",
            },
            "stable": {
                "pixels": stable_pixels,
                "percentage": stable_pct,
                "color": "#8b949e",
            },
            "burned_area": {
                "pixels": burned_pixels,
                "percentage": burned_pct,
                "color": "#d29922",
            },
            "ndvi_change_stats": delta_stats,
            "ndvi_before_mean": round(float(np.mean(ndvi_before[np.isfinite(ndvi_before)])), 4),
            "ndvi_after_mean": round(float(np.mean(ndvi_after[np.isfinite(ndvi_after)])), 4),
        }

        logger.info(f"ForestChange: {trend} — Gain={gain_pct}%, Loss={loss_pct}%, "
                     f"Burned={burned_pct}% ({date_before} → {date_after})")
        return result

    def detect_from_images(self, before_path: str, after_path: str) -> Dict[str, Any]:
        """Detect change from two RGB images using synthetic NDVI."""
        from PIL import Image

        def _img_to_ndvi(path):
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            red, green = arr[:, :, 0], arr[:, :, 1]
            nir_approx = green * 1.2 + red * 0.1
            return (nir_approx - red) / (nir_approx + red + 1e-8)

        ndvi_before = _img_to_ndvi(before_path)
        ndvi_after = _img_to_ndvi(after_path)

        # Resize to match if dimensions differ
        if ndvi_before.shape != ndvi_after.shape:
            from PIL import Image as PILImage
            h, w = min(ndvi_before.shape[0], ndvi_after.shape[0]), min(ndvi_before.shape[1], ndvi_after.shape[1])
            ndvi_before = ndvi_before[:h, :w]
            ndvi_after = ndvi_after[:h, :w]

        return self.detect_change(ndvi_before, ndvi_after,
                                  date_before=str(before_path),
                                  date_after=str(after_path))
