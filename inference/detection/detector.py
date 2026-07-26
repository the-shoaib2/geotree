"""
GeoTree — Core Detection Engine
Runs the trained TreeDetectorModel on images and produces structured detection results.
"""
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Union

from training.models.selector import TreeDetectorModel
from configs.train_config import train_config
from preprocessing.pipeline.helpers import logger


# Class label mapping for multi-class detection
CLASS_LABELS = {
    0: "tree",
    1: "building",
    2: "road",
    3: "water",
    4: "grass",
    5: "bare_soil",
    6: "cloud",
    7: "shadow",
}

CLASS_COLORS = {
    "tree": "#2ea44f",
    "building": "#f85149",
    "road": "#8b949e",
    "water": "#58a6ff",
    "grass": "#7ee787",
    "bare_soil": "#d29922",
    "cloud": "#c9d1d9",
    "shadow": "#484f58",
}


class TreeDetector:
    """Production detection engine — loads model weights and runs inference."""

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or train_config.device
        self.img_size = train_config.img_size
        self.weights_path = Path(weights_path) if weights_path else train_config.weights_dir / "best_model.pth"

        # Load model
        self.model = TreeDetectorModel().to(self.device)
        if self.weights_path.exists():
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"TreeDetector: Loaded weights from {self.weights_path}")
        else:
            logger.warning(f"TreeDetector: No weights found at {self.weights_path}")
        self.model.eval()

    def _preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess a single image to model input tensor."""
        if isinstance(image, (str, Path)):
            try:
                img = Image.open(image).convert("RGB")
            except Exception:
                # Fallback: try rasterio for GeoTIFF files with wrong extension
                try:
                    import rasterio
                    with rasterio.open(image) as src:
                        if src.count >= 3:
                            r = src.read(1).astype(np.float32)
                            g = src.read(2).astype(np.float32)
                            b = src.read(3).astype(np.float32)
                        else:
                            r = g = b = src.read(1).astype(np.float32)
                        # Normalize to 0-255 range for PIL
                        stack = np.stack([r, g, b], axis=-1)
                        smin = np.nanmin(stack)
                        smax = np.nanmax(stack)
                        if smax > smin:
                            stack = ((stack - smin) / (smax - smin) * 255.0)
                        stack = np.clip(np.nan_to_num(stack, nan=0.0), 0, 255).astype(np.uint8)
                        img = Image.fromarray(stack, mode="RGB")
                except Exception as e:
                    raise ValueError(f"Cannot open image {image}: {e}")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image.convert("RGB")

        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.tensor(arr).unsqueeze(0)

    def detect_single(self, image: Union[str, Path, Image.Image, np.ndarray],
                      confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Run detection on a single image.

        Returns dict with keys:
            - detections: list of {class_name, confidence, bbox_xywh, bbox_xyxy_pixels}
            - latency_ms: inference time in ms
            - image_path: original path (if applicable)
        """
        tensor = self._preprocess(image).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor).squeeze(0).cpu()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # output: [confidence_logit, x, y, w, h]
        confidence = torch.sigmoid(output[0]).item()
        x, y, w, h = output[1].item(), output[2].item(), output[3].item(), output[4].item()

        detections = []
        if confidence >= confidence_threshold and w > 0.01 and h > 0.01:
            # Convert normalized xywh to pixel xyxy
            x1 = max(0, (x - w / 2)) * self.img_size
            y1 = max(0, (y - h / 2)) * self.img_size
            x2 = min(1, (x + w / 2)) * self.img_size
            y2 = min(1, (y + h / 2)) * self.img_size

            # Crown area in pixels
            area_px = (x2 - x1) * (y2 - y1)

            detections.append({
                "class_id": 0,
                "class_name": "tree",
                "confidence": round(confidence, 4),
                "bbox_xywh": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
                "bbox_xyxy_pixels": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "crown_area_px": round(area_px, 1),
                "color": CLASS_COLORS["tree"],
            })

        image_path = str(image) if isinstance(image, (str, Path)) else "in_memory"

        return {
            "image_path": image_path,
            "image_size": self.img_size,
            "detections": detections,
            "num_detections": len(detections),
            "latency_ms": round(latency_ms, 2),
        }

    def detect_directory(self, tiles_dir: Union[str, Path],
                         confidence_threshold: float = 0.3,
                         extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                         ) -> List[Dict[str, Any]]:
        """Run detection on all images in a directory."""
        tiles_dir = Path(tiles_dir)
        image_files = sorted([f for f in tiles_dir.iterdir() if f.suffix.lower() in extensions])

        if not image_files:
            logger.warning(f"No images found in {tiles_dir}")
            return []

        logger.info(f"TreeDetector: Running detection on {len(image_files)} images from {tiles_dir}")
        results = []
        total_detections = 0

        for img_path in image_files:
            try:
                result = self.detect_single(img_path, confidence_threshold)
                results.append(result)
                total_detections += result["num_detections"]
            except Exception as e:
                logger.warning(f"Failed to process {img_path.name}: {e}")

        logger.info(f"TreeDetector: Completed — {total_detections} detections across {len(results)} images")
        return results

    def detect_image(self, image_path: str, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Convenience method — detect on a single image path."""
        path = Path(image_path)
        if path.is_dir():
            return {"results": self.detect_directory(path, confidence_threshold)}
        return self.detect_single(path, confidence_threshold)
