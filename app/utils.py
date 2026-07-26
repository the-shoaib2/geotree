import json
from pathlib import Path
from typing import Dict, Any
from shapely.geometry import shape, MultiPolygon, Polygon

DISTRICT_BOUNDS = {
    "bandarban": {
        "type": "Polygon",
        "coordinates": [[
            [92.03, 21.19],
            [92.68, 21.19],
            [92.68, 22.37],
            [92.03, 22.37],
            [92.03, 21.19]
        ]]
    },
    "rangamati": {
        "type": "Polygon",
        "coordinates": [[
            [92.04, 22.27],
            [92.73, 22.27],
            [92.73, 23.38],
            [92.04, 23.38],
            [92.04, 22.27]
        ]]
    },
    "sylhet": {
        "type": "Polygon",
        "coordinates": [[
            [91.63, 24.59],
            [92.51, 24.59],
            [92.51, 25.19],
            [91.63, 25.19],
            [91.63, 24.59]
        ]]
    },
    "gazipur": {
        "type": "Polygon",
        "coordinates": [[
            [90.15, 23.88],
            [90.71, 23.88],
            [90.71, 24.34],
            [90.15, 24.34],
            [90.15, 23.88]
        ]]
    }
}

def load_geojson_geometry(geojson_path: str) -> Dict[str, Any]:
    """Reads a GeoJSON file and returns the geometry dict."""
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found at: {geojson_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if "features" in data:
        geom = data["features"][0]["geometry"]
    elif "geometry" in data:
        geom = data["geometry"]
    else:
        geom = data
        
    geom_shape = shape(geom)
    if not isinstance(geom_shape, (Polygon, MultiPolygon)):
        raise ValueError("GeoJSON must represent a Polygon or MultiPolygon geometry.")
        
    return geom

def get_district_geometry(district_name: str) -> Dict[str, Any]:
    """Gets the GeoJSON geometry for a specified district."""
    name_clean = district_name.strip().lower()
    if name_clean not in DISTRICT_BOUNDS:
        raise ValueError(f"Unknown district: {district_name}. Supported: {list(DISTRICT_BOUNDS.keys())}")
    return DISTRICT_BOUNDS[name_clean]

def geom_to_wkt(geom: dict) -> str:
    """Converts a GeoJSON geometry dict to a WKT string."""
    geom_type = geom["type"]
    if geom_type == "Polygon":
        rings = []
        for ring in geom["coordinates"]:
            coords_str = ", ".join(f"{lon} {lat}" for lon, lat in ring)
            rings.append(f"({coords_str})")
        return f"POLYGON({', '.join(rings)})"
    elif geom_type == "MultiPolygon":
        polys = []
        for poly in geom["coordinates"]:
            rings = []
            for ring in poly:
                coords_str = ", ".join(f"{lon} {lat}" for lon, lat in ring)
                rings.append(f"({coords_str})")
            polys.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON({', '.join(polys)})"
    else:
        raise ValueError(f"Unsupported geometry type for WKT conversion: {geom_type}")
