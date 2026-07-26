"""
GeoTree — Batch Detection Processor
Runs detection across all tiles in a directory and aggregates results to JSON.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from inference.detection.detector import TreeDetector
from preprocessing.pipeline.helpers import logger, ensure_dir


class BatchDetector:
    """Processes a directory of tiles through the detection engine and saves aggregated results."""

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.detector = TreeDetector(weights_path=weights_path, device=device)

    def process(self, tiles_dir: str, output_dir: str = "reports",
                confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Run batch detection on all tiles and save results.

        Args:
            tiles_dir: Directory containing image tiles
            output_dir: Directory to save detection_results.json
            confidence_threshold: Minimum confidence to count a detection

        Returns:
            Aggregated detection summary dict
        """
        tiles_path = Path(tiles_dir)
        output_path = Path(output_dir)
        ensure_dir(output_path)

        t0 = time.perf_counter()
        per_image_results = self.detector.detect_directory(tiles_path, confidence_threshold)
        elapsed = time.perf_counter() - t0

        # Aggregate statistics
        total_images = len(per_image_results)
        total_detections = sum(r["num_detections"] for r in per_image_results)
        total_latency = sum(r["latency_ms"] for r in per_image_results)
        avg_latency = total_latency / max(total_images, 1)

        # Collect all detections flat
        all_detections = []
        for r in per_image_results:
            for det in r["detections"]:
                all_detections.append({
                    "image": r["image_path"],
                    **det
                })

        # Class distribution
        class_counts = {}
        for det in all_detections:
            cls = det["class_name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Confidence distribution
        confidences = [d["confidence"] for d in all_detections]
        avg_conf = sum(confidences) / max(len(confidences), 1)
        min_conf = min(confidences) if confidences else 0.0
        max_conf = max(confidences) if confidences else 0.0

        summary = {
            "pipeline": "GeoTree Batch Detection",
            "tiles_dir": str(tiles_path),
            "total_images_processed": total_images,
            "total_detections": total_detections,
            "class_distribution": class_counts,
            "confidence_stats": {
                "mean": round(avg_conf, 4),
                "min": round(min_conf, 4),
                "max": round(max_conf, 4),
            },
            "performance": {
                "total_time_sec": round(elapsed, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "throughput_fps": round(total_images / max(elapsed, 0.001), 1),
            },
            "per_image_results": per_image_results,
        }

        # Save to JSON
        json_path = output_path / "detection_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"BatchDetector: {total_detections} detections across {total_images} tiles → {json_path}")
        return summary
