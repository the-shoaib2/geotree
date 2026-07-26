"""
GeoTree — Full Pipeline Orchestrator
End-to-end processing: Image tiles → Detection → Segmentation → Vegetation Analysis → GIS Analytics
Produces comprehensive analysis_results.json with all outputs.
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from preprocessing.pipeline.helpers import logger, ensure_dir


class GeoTreePipeline:
    """Orchestrates the complete GeoTree analysis pipeline.

    Pipeline stages:
        1. Detection — Tree detection on image tiles
        2. Segmentation — Land-cover classification
        3. Vegetation Analysis — NDVI/EVI health assessment
        4. GIS Analytics — Tree count, density, biomass, carbon, land-cover stats
    """

    def __init__(self, weights_path: Optional[str] = None,
                 pixel_resolution_m: float = 10.0,
                 forest_type: str = "tropical_broadleaf"):
        self.weights_path = weights_path
        self.pixel_res = pixel_resolution_m
        self.forest_type = forest_type

    def run(self, tiles_dir: str = "dataset/yolo/images/val",
            output_dir: str = "reports",
            confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Execute the full GeoTree analysis pipeline.

        Args:
            tiles_dir: Directory of image tiles to analyze
            output_dir: Directory to save all outputs
            confidence_threshold: Minimum detection confidence

        Returns:
            Comprehensive analysis results dict
        """
        tiles_path = Path(tiles_dir)
        output_path = Path(output_dir)
        ensure_dir(output_path)

        logger.info("=" * 70)
        logger.info("  🌳 GeoTree — Full Analysis Pipeline Starting")
        logger.info("=" * 70)
        pipeline_start = time.perf_counter()

        results = {
            "pipeline": "GeoTree Full Analysis",
            "tiles_dir": str(tiles_path),
            "output_dir": str(output_path),
        }

        # ── Stage 1: Detection ──────────────────────────────────────────────
        logger.info("\n[1/4] 🔍 Running Detection Engine...")
        t0 = time.perf_counter()
        try:
            from inference.detection.detector import TreeDetector
            detector = TreeDetector(weights_path=self.weights_path)
            detection_results = detector.detect_directory(tiles_path, confidence_threshold)
            total_detections = sum(r["num_detections"] for r in detection_results)
            results["detection"] = {
                "total_images": len(detection_results),
                "total_detections": total_detections,
                "per_image": detection_results,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            logger.info(f"    ✅ {total_detections} detections across {len(detection_results)} tiles")
        except Exception as e:
            logger.error(f"    ❌ Detection failed: {e}")
            results["detection"] = {"error": str(e)}
            detection_results = []

        # ── Stage 2: Segmentation ───────────────────────────────────────────
        logger.info("\n[2/4] 🗺️  Running Segmentation Engine...")
        t0 = time.perf_counter()
        try:
            from inference.segmentation.segmenter import LandCoverSegmenter
            segmenter = LandCoverSegmenter()
            seg_results = []
            image_files = sorted([f for f in tiles_path.iterdir()
                                   if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')])
            for img_path in image_files:
                try:
                    seg = segmenter.segment_image(str(img_path))
                    seg_results.append(seg)
                except Exception as e:
                    logger.warning(f"    Segmentation skip {img_path.name}: {e}")

            results["segmentation"] = {
                "tiles_segmented": len(seg_results),
                "per_tile": seg_results,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            logger.info(f"    ✅ {len(seg_results)} tiles segmented")
        except Exception as e:
            logger.error(f"    ❌ Segmentation failed: {e}")
            results["segmentation"] = {"error": str(e)}
            seg_results = []

        # ── Stage 3: Vegetation Analysis ────────────────────────────────────
        logger.info("\n[3/4] 🌿 Running Vegetation Health Analysis...")
        t0 = time.perf_counter()
        try:
            from analytics.health.health_analyzer import VegetationHealthAnalyzer
            health_analyzer = VegetationHealthAnalyzer()
            health_results = []
            for img_path in image_files:
                try:
                    health = health_analyzer.analyze_from_image(str(img_path))
                    health_results.append(health)
                except Exception as e:
                    logger.warning(f"    Health skip {img_path.name}: {e}")

            # Aggregate health scores
            if health_results:
                avg_score = sum(h["health_score"] for h in health_results) / len(health_results)
                avg_ndvi = sum(h["ndvi_statistics"]["mean"] for h in health_results) / len(health_results)
            else:
                avg_score = 0.0
                avg_ndvi = 0.0

            results["vegetation_health"] = {
                "tiles_analyzed": len(health_results),
                "avg_health_score": round(avg_score, 1),
                "avg_ndvi": round(avg_ndvi, 4),
                "per_tile": health_results,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            logger.info(f"    ✅ Avg Health Score: {avg_score:.1f}/100, Avg NDVI: {avg_ndvi:.4f}")
        except Exception as e:
            logger.error(f"    ❌ Health analysis failed: {e}")
            results["vegetation_health"] = {"error": str(e)}

        # ── Stage 4: GIS Analytics ──────────────────────────────────────────
        logger.info("\n[4/4] 📊 Running GIS Analytics Engine...")
        t0 = time.perf_counter()
        try:
            # 4a. Tree Count
            from analytics.tree_count.counter import TreeCounter
            counter = TreeCounter(pixel_resolution_m=self.pixel_res)
            tree_count = counter.count_from_detections(detection_results)
            results["tree_count"] = tree_count

            # 4b. Tree Density
            from analytics.density.density_mapper import TreeDensityMapper
            mapper = TreeDensityMapper(grid_size=25)
            density = mapper.compute_density_grid(detection_results)
            # Remove large grid arrays for JSON output
            density_summary = {k: v for k, v in density.items()
                               if k not in ("density_grid", "density_per_ha_grid")}
            results["tree_density"] = density_summary

            # 4c. Land Cover Statistics
            from analytics.statistics.statistics_engine import LandCoverStatistics
            stats_engine = LandCoverStatistics(pixel_resolution_m=self.pixel_res)
            if seg_results:
                land_stats = stats_engine.compute(seg_results)
                results["land_cover_statistics"] = land_stats
            else:
                results["land_cover_statistics"] = {"note": "No segmentation data available"}

            # 4d. Biomass & Carbon
            from analytics.biomass.biomass_estimator import BiomassEstimator
            biomass = BiomassEstimator(forest_type=self.forest_type,
                                       pixel_resolution_m=self.pixel_res)
            biomass_result = biomass.estimate_from_detections(detection_results)
            results["biomass_carbon"] = biomass_result

            results["gis_analytics_elapsed_sec"] = round(time.perf_counter() - t0, 2)
            logger.info(f"    ✅ Tree Count: {tree_count['total_tree_count']}, "
                         f"Biomass: {biomass_result['biomass']['total_tonnes']:.3f}t, "
                         f"Carbon: {biomass_result['carbon_storage']['total_tonnes']:.3f}t")
        except Exception as e:
            logger.error(f"    ❌ GIS Analytics failed: {e}")
            results["gis_analytics"] = {"error": str(e)}

        # ── Pipeline Summary ────────────────────────────────────────────────
        total_elapsed = time.perf_counter() - pipeline_start
        results["pipeline_elapsed_sec"] = round(total_elapsed, 2)

        # Save comprehensive JSON
        json_path = output_path / "analysis_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n" + "=" * 70)
        logger.info(f"  🌳 GeoTree Pipeline Complete — {round(total_elapsed, 1)}s")
        logger.info(f"  📄 Results saved to {json_path}")
        logger.info("=" * 70)

        return results
