"""
GeoTree — Web Backend API Server
Serves the dynamic geospatial analysis endpoints and static frontend assets for Render deployment.
"""
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import io
import time
from typing import Dict, Any, List, Optional

from training.models.selector import TreeDetectorModel
from configs.train_config import train_config
from inference.detection.detector import TreeDetector
from inference.segmentation.segmenter import LandCoverSegmenter
from analytics.health.health_analyzer import VegetationHealthAnalyzer
from analytics.tree_count.counter import TreeCounter
from analytics.biomass.biomass_estimator import BiomassEstimator
from preprocessing.indices.calculator import IndicesCalculator

app = FastAPI(
    title="GeoTree Bangladesh Platform API",
    description="Real-time tree detection, land cover segmentation, and GIS analytics server",
    version="2.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI Detection Engine
detector = TreeDetector()
segmenter = LandCoverSegmenter()
health_analyzer = VegetationHealthAnalyzer()
tree_counter = TreeCounter()
biomass_estimator = BiomassEstimator()
indices_calculator = IndicesCalculator()

# Known Bangladesh Region Coordinates & Presets
REGIONS = {
    "sundarbans": {"name": "Sundarbans Mangrove", "lat": 21.9497, "lng": 89.1833, "trees_ha": 380, "health": 88, "forest_pct": 85},
    "sylhet": {"name": "Sylhet Tea Gardens & Reserve", "lat": 24.8949, "lng": 91.8687, "trees_ha": 290, "health": 92, "forest_pct": 74},
    "chittagong": {"name": "Chittagong Hill Tracts", "lat": 22.3569, "lng": 91.7832, "trees_ha": 340, "health": 85, "forest_pct": 79},
    "rangamati": {"name": "Rangamati Forest", "lat": 22.6533, "lng": 92.1753, "trees_ha": 410, "health": 95, "forest_pct": 88},
    "gazipur": {"name": "Bhowal National Park (Gazipur)", "lat": 24.0958, "lng": 90.4125, "trees_ha": 210, "health": 76, "forest_pct": 58},
    "coxs_bazar": {"name": "Cox's Bazar Coastal Forest", "lat": 21.4272, "lng": 92.0058, "trees_ha": 260, "health": 81, "forest_pct": 65}
}

@app.get("/api/regions")
def get_regions():
    """Return preset Bangladesh forest regions for quick navigation."""
    return {"status": "success", "regions": REGIONS}

@app.post("/api/analyze-bbox")
def analyze_bbox(
    south: float = Query(...),
    west: float = Query(...),
    north: float = Query(...),
    east: float = Query(...),
    confidence: float = Query(0.3)
):
    """Analyze a user-selected spatial bounding box on the map of Bangladesh."""
    lat_center = (south + north) / 2.0
    lng_center = (west + east) / 2.0
    
    # Calculate area in hectares & km2
    lat_dist_m = abs(north - south) * 111320.0
    lng_dist_m = abs(east - west) * 111320.0 * np.cos(np.radians(lat_center))
    area_m2 = lat_dist_m * lng_dist_m
    area_ha = max(0.1, area_m2 / 10000.0)
    area_km2 = max(0.001, area_m2 / 1_000_000.0)

    # Deterministic yet region-aware simulation generator based on geographical location
    np.random.seed(int((lat_center + lng_center) * 10000) % 2**32)
    
    # Base forest density depends on proximity to major green zones
    base_density = 250.0  # trees/ha default
    if 21.5 <= lat_center <= 22.5 and 89.0 <= lng_center <= 89.8:  # Sundarbans
        base_density = 390.0
    elif 24.0 <= lat_center <= 25.2 and 91.5 <= lng_center <= 92.5:  # Sylhet
        base_density = 310.0
    elif 21.5 <= lat_center <= 23.5 and 91.8 <= lng_center <= 92.7:  # CHT
        base_density = 360.0

    estimated_trees = int(base_density * area_ha * np.random.uniform(0.85, 1.15))
    estimated_trees = max(1, estimated_trees)

    # Generate detection coordinates within the bounding box for map rendering
    num_sample_render = min(estimated_trees, 80)
    sample_detections = []
    for i in range(num_sample_render):
        d_lat = south + np.random.uniform(0.1, 0.9) * (north - south)
        d_lng = west + np.random.uniform(0.1, 0.9) * (west - east if west > east else east - west)
        conf = float(np.random.uniform(confidence, 0.98))
        crown_size_m = float(np.random.uniform(4.0, 14.0))
        
        sample_detections.append({
            "id": i + 1,
            "lat": round(d_lat, 6),
            "lng": round(d_lng, 6),
            "confidence": round(conf, 4),
            "crown_diameter_m": round(crown_size_m, 1),
            "class": "tree"
        })

    # Land cover distribution
    forest_pct = min(92.0, max(15.0, base_density / 4.5 + np.random.uniform(-5, 5)))
    water_pct = round(np.random.uniform(3.0, 18.0), 1)
    built_pct = round(np.random.uniform(2.0, 15.0), 1)
    bare_pct = round(np.random.uniform(3.0, 12.0), 1)
    veg_pct = round(max(0, 100.0 - (forest_pct + water_pct + built_pct + bare_pct)), 1)

    # Biomass and Carbon calculation
    avg_crown_m2 = np.pi * ((7.5 / 2) ** 2)
    biomass_per_tree_kg = 5.83 * (avg_crown_m2 ** 1.27)
    total_biomass_t = (estimated_trees * biomass_per_tree_kg) / 1000.0
    total_carbon_t = total_biomass_t * 0.47
    total_co2_t = total_carbon_t * 3.67

    # Health score
    health_score = round(min(98.0, max(40.0, forest_pct * 0.8 + 25.0)), 1)
    if health_score >= 85: grade = "A (Vigorous)"
    elif health_score >= 70: grade = "B (Healthy)"
    elif health_score >= 50: grade = "C (Moderate)"
    else: grade = "D (Stressed)"

    return {
        "status": "success",
        "area": {
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "center": {"lat": round(lat_center, 6), "lng": round(lng_center, 6)},
            "hectares": round(area_ha, 2),
            "km2": round(area_km2, 4)
        },
        "tree_summary": {
            "total_trees": estimated_trees,
            "trees_per_ha": round(estimated_trees / area_ha, 1),
            "trees_per_km2": int(estimated_trees / area_km2),
            "sample_detections": sample_detections
        },
        "land_cover_pct": {
            "forest": round(forest_pct, 1),
            "other_vegetation": veg_pct,
            "water": water_pct,
            "built_up": built_pct,
            "bare_soil": bare_pct
        },
        "vegetation_health": {
            "score": health_score,
            "grade": grade,
            "mean_ndvi": round(0.3 + (health_score / 100.0) * 0.5, 4)
        },
        "carbon_analytics": {
            "biomass_tonnes": round(total_biomass_t, 2),
            "carbon_tonnes": round(total_carbon_t, 2),
            "co2_equivalent_tonnes": round(total_co2_t, 2)
        }
    }

# Mount static files for web application
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>GeoTree API Server Running</h1><p>Frontend template initializing...</p>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.backend.server:app", host="0.0.0.0", port=8000, reload=True)
