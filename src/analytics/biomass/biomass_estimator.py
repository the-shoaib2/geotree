"""
GeoTree — Biomass & Carbon Estimator
Estimates above-ground biomass and carbon storage from tree crown detections.
"""
import math
from typing import Dict, Any, List
from preprocessing.pipeline.helpers import logger


# Allometric equation parameters for tropical broadleaf trees (pantropical model)
# Biomass (kg) = a * (Crown_Area_m2) ^ b
# Reference: Jucker et al. (2017) — pantropical crown-area allometry
ALLOMETRIC_PARAMS = {
    "tropical_broadleaf": {"a": 5.83, "b": 1.27},
    "temperate_mixed": {"a": 4.12, "b": 1.22},
    "mangrove": {"a": 3.45, "b": 1.35},
    "default": {"a": 5.0, "b": 1.25},
}

# Carbon conversion: biomass × 0.47 (IPCC default)
CARBON_FRACTION = 0.47

# CO2 equivalent: carbon × 3.67
CO2_FACTOR = 3.67


class BiomassEstimator:
    """Estimates above-ground biomass from tree crown area using allometric equations."""

    def __init__(self, forest_type: str = "tropical_broadleaf",
                 pixel_resolution_m: float = 10.0):
        self.params = ALLOMETRIC_PARAMS.get(forest_type, ALLOMETRIC_PARAMS["default"])
        self.pixel_res = pixel_resolution_m
        self.forest_type = forest_type

    def estimate_from_detections(self, detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate biomass and carbon from tree detection results.

        Uses crown area from each detection to apply allometric equations.

        Args:
            detection_results: List of per-image results from TreeDetector

        Returns:
            Dict with biomass, carbon, CO2 estimates and per-tree breakdown
        """
        a, b = self.params["a"], self.params["b"]
        per_tree = []
        total_biomass_kg = 0.0
        total_crown_area_m2 = 0.0

        for result in detection_results:
            for det in result.get("detections", []):
                crown_px = det.get("crown_area_px", 0)
                crown_m2 = crown_px * (self.pixel_res ** 2)
                total_crown_area_m2 += crown_m2

                if crown_m2 > 0:
                    biomass_kg = a * (crown_m2 ** b)
                else:
                    biomass_kg = 0.0

                total_biomass_kg += biomass_kg
                per_tree.append({
                    "crown_area_px": round(crown_px, 1),
                    "crown_area_m2": round(crown_m2, 2),
                    "biomass_kg": round(biomass_kg, 2),
                })

        total_trees = len(per_tree)
        total_carbon_kg = total_biomass_kg * CARBON_FRACTION
        total_co2_kg = total_carbon_kg * CO2_FACTOR

        # Convert to tonnes
        biomass_tonnes = total_biomass_kg / 1000.0
        carbon_tonnes = total_carbon_kg / 1000.0
        co2_tonnes = total_co2_kg / 1000.0

        # Per tree averages
        avg_biomass = total_biomass_kg / max(total_trees, 1)
        avg_crown_m2 = total_crown_area_m2 / max(total_trees, 1)

        result = {
            "forest_type": self.forest_type,
            "allometric_model": f"Biomass = {a} × CrownArea^{b}",
            "total_trees": total_trees,
            "total_crown_area_m2": round(total_crown_area_m2, 2),
            "biomass": {
                "total_kg": round(total_biomass_kg, 2),
                "total_tonnes": round(biomass_tonnes, 3),
                "per_tree_avg_kg": round(avg_biomass, 2),
            },
            "carbon_storage": {
                "total_kg": round(total_carbon_kg, 2),
                "total_tonnes": round(carbon_tonnes, 3),
                "conversion_factor": CARBON_FRACTION,
            },
            "co2_equivalent": {
                "total_kg": round(total_co2_kg, 2),
                "total_tonnes": round(co2_tonnes, 3),
                "conversion_factor": CO2_FACTOR,
            },
            "crown_statistics": {
                "avg_crown_area_m2": round(avg_crown_m2, 2),
                "total_crown_area_ha": round(total_crown_area_m2 / 10000.0, 4),
            },
        }

        logger.info(f"BiomassEstimator: {total_trees} trees → {round(biomass_tonnes, 2)} tonnes biomass, "
                     f"{round(carbon_tonnes, 2)} tonnes carbon, {round(co2_tonnes, 2)} tonnes CO₂")
        return result


class CarbonEstimator(BiomassEstimator):
    """Convenience wrapper that focuses on carbon storage output."""

    def estimate_carbon(self, detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Shorthand for carbon-focused estimation."""
        full = self.estimate_from_detections(detection_results)
        return {
            "total_trees": full["total_trees"],
            "carbon_tonnes": full["carbon_storage"]["total_tonnes"],
            "co2_tonnes": full["co2_equivalent"]["total_tonnes"],
            "biomass_tonnes": full["biomass"]["total_tonnes"],
        }
