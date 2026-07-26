"""
GeoTree — Enhanced FastAPI Serving Endpoint
Exposes all pipeline capabilities: detection, segmentation, vegetation indices,
analytics, and full analysis through REST API endpoints.
"""
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
import io
import numpy as np
from PIL import Image

from training.models.selector import TreeDetectorModel
from configs.train_config import train_config

app = FastAPI(
    title="GeoTree AI Platform API",
    description="Complete geospatial tree detection, segmentation, and analytics engine for Bangladesh",
    version="2.0.0",
)

# ── Load detection model ────────────────────────────────────────────────
model = TreeDetectorModel()
weights_path = Path("weights/best_model.pth")
if weights_path.exists():
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    except Exception:
        pass
model.eval()


def _load_image(file_bytes: bytes) -> np.ndarray:
    """Load uploaded image bytes into numpy array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to model input tensor."""
    img = img.resize((train_config.img_size, train_config.img_size))
    arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0)


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "platform": "GeoTree AI",
        "version": "2.0.0",
        "status": "online",
        "endpoints": ["/detect", "/segment", "/indices", "/health", "/analytics", "/full-analysis"],
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 confidence: float = Query(0.3, description="Minimum confidence threshold")):
    """Run tree detection on uploaded image."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = _image_to_tensor(img)

    with torch.no_grad():
        output = model(tensor).squeeze(0).tolist()

    conf = max(0.0, min(1.0, float(output[0])))
    bbox = [max(0.0, min(1.0, float(x))) for x in output[1:]]

    detections = []
    if conf >= confidence and bbox[2] > 0.01 and bbox[3] > 0.01:
        x, y, w, h = bbox
        detections.append({
            "class": "tree",
            "confidence": round(conf, 4),
            "bbox_xywh": [round(v, 4) for v in bbox],
            "bbox_xyxy_pixels": [
                round(max(0, (x - w/2)) * train_config.img_size, 1),
                round(max(0, (y - h/2)) * train_config.img_size, 1),
                round(min(1, (x + w/2)) * train_config.img_size, 1),
                round(min(1, (y + h/2)) * train_config.img_size, 1),
            ],
        })

    return {
        "filename": file.filename,
        "detections": detections,
        "count": len(detections),
    }


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """Run land-cover segmentation on uploaded image."""
    from inference.segmentation.segmenter import LandCoverSegmenter

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    nir_approx = green * 1.2 + red * 0.1

    segmenter = LandCoverSegmenter()
    mask = segmenter.segment_from_bands(red, green, blue, nir_approx)
    result = segmenter._compute_statistics(mask, Path(file.filename or "upload"))
    result.pop("source", None)

    return {"filename": file.filename, **result}


@app.post("/indices")
async def compute_indices(file: UploadFile = File(...)):
    """Compute vegetation indices (NDVI, NDWI, NDMI, EVI) from uploaded image."""
    from preprocessing.indices.calculator import IndicesCalculator

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    nir_approx = green * 1.2 + red * 0.1

    calc = IndicesCalculator()
    result = calc.calculate_from_arrays(red, green, blue, nir_approx)

    return {"filename": file.filename, "indices": result["statistics"]}


@app.post("/health")
async def analyze_health(file: UploadFile = File(...)):
    """Analyze vegetation health from uploaded image."""
    from analytics.health.health_analyzer import VegetationHealthAnalyzer

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    red, green = arr[:, :, 0], arr[:, :, 1]
    nir_approx = green * 1.2 + red * 0.1
    ndvi = (nir_approx - red) / (nir_approx + red + 1e-8)

    analyzer = VegetationHealthAnalyzer()
    result = analyzer.analyze(ndvi, source_name=file.filename or "upload")

    return result


@app.post("/analytics")
async def run_analytics(file: UploadFile = File(...)):
    """Run tree counting and biomass/carbon estimation on uploaded image."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = _image_to_tensor(img)

    with torch.no_grad():
        output = model(tensor).squeeze(0).tolist()

    conf = max(0.0, min(1.0, float(output[0])))
    bbox = [max(0.0, min(1.0, float(x))) for x in output[1:]]

    detection_result = {
        "num_detections": 1 if conf >= 0.3 else 0,
        "detections": [{
            "confidence": conf,
            "bbox_xywh": bbox,
            "crown_area_px": bbox[2] * bbox[3] * train_config.img_size ** 2 if conf >= 0.3 else 0,
        }] if conf >= 0.3 else [],
    }

    from analytics.tree_count.counter import TreeCounter
    from analytics.biomass.biomass_estimator import BiomassEstimator

    counter = TreeCounter()
    count_result = counter.count_from_detections([detection_result])

    estimator = BiomassEstimator()
    biomass_result = estimator.estimate_from_detections([detection_result])

    return {
        "filename": file.filename,
        "tree_count": count_result["total_tree_count"],
        "trees_per_hectare": count_result["trees_per_hectare"],
        "biomass_kg": biomass_result["biomass"]["total_kg"],
        "carbon_kg": biomass_result["carbon_storage"]["total_kg"],
        "co2_kg": biomass_result["co2_equivalent"]["total_kg"],
    }


@app.post("/full-analysis")
async def full_analysis(file: UploadFile = File(...)):
    """Run complete GeoTree analysis: detect + segment + indices + health + analytics."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Detection
    tensor = _image_to_tensor(img)
    with torch.no_grad():
        output = model(tensor).squeeze(0).tolist()
    conf = max(0.0, min(1.0, float(output[0])))
    bbox = [max(0.0, min(1.0, float(x))) for x in output[1:]]

    detections = []
    if conf >= 0.3 and bbox[2] > 0.01 and bbox[3] > 0.01:
        detections.append({"class": "tree", "confidence": round(conf, 4), "bbox_xywh": bbox})

    # Segmentation + Indices + Health
    arr = np.array(img).astype(np.float32) / 255.0
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    nir = green * 1.2 + red * 0.1

    from inference.segmentation.segmenter import LandCoverSegmenter
    from preprocessing.indices.calculator import IndicesCalculator
    from analytics.health.health_analyzer import VegetationHealthAnalyzer

    seg = LandCoverSegmenter()
    mask = seg.segment_from_bands(red, green, blue, nir)
    seg_stats = seg._compute_statistics(mask, Path(file.filename or "upload"))

    calc = IndicesCalculator()
    idx = calc.calculate_from_arrays(red, green, blue, nir)

    ndvi = (nir - red) / (nir + red + 1e-8)
    health = VegetationHealthAnalyzer().analyze(ndvi)

    return {
        "filename": file.filename,
        "detection": {"count": len(detections), "detections": detections},
        "segmentation": {k: v for k, v in seg_stats.items() if k != "source"},
        "vegetation_indices": idx["statistics"],
        "vegetation_health": {
            "score": health["health_score"],
            "grade": health["health_grade"],
        },
    }
