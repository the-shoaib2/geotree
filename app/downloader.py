import os
import time
import urllib.parse
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from app.config import config
from app.auth import auth_manager
from app.logger import logger
from app.database import db_manager
from app.checksum import calculate_sha256
from app.utils import DISTRICT_BOUNDS

class DownloadManager:
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries

    def download_product(self, product: Dict[str, Any], download_url: str, is_sentinel: bool = True) -> bool:
        """Downloads a dataset product (Sentinel-2 or public) with progress bar, resume support, and retry logic."""
        product_id = product["product_id"]
        dataset_type = product.get("dataset_type", "sentinel2")
        district = product.get("district")
        
        # Determine destination folder path
        base_dir = Path(config.download_directory)
        
        if download_url.startswith("sentinelhub://process"):
            # Parse parameters from custom URI
            parsed_url = urllib.parse.urlparse(download_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            dist_name = query_params.get("district", [district])[0].lower()
            date_str = query_params.get("date", [product["date"]])[0]
            
            dest_dir = base_dir / "sentinel2" / dist_name
            file_name = f"{dist_name}_{date_str}.tif"
            local_path = dest_dir / file_name
        else:
            dest_dir = base_dir / dataset_type.lower()
            url_part = download_url.split("/")[-1].split("?")[0]
            if not url_part or url_part == "content" or len(url_part) > 50:
                if dataset_type == "deepforest":
                    url_part = "deepforest_data.png"
                elif dataset_type == "zenodo":
                    url_part = "zenodo_dataset.pdf"
                elif dataset_type == "selvabox":
                    url_part = "selvabox_data.md"
                elif dataset_type == "global_forest_change":
                    url_part = "global_forest_change_bangladesh.tif"
                else:
                    url_part = f"{product_id}.bin"
            file_name = url_part
            local_path = dest_dir / file_name
            
        dest_dir.mkdir(parents=True, exist_ok=True)

        # If already completed in DB and file exists, skip
        if product.get("download_status") == "completed" and local_path.exists():
            logger.info(f"Product {product_id} ({dataset_type}) already downloaded successfully. Skipping.")
            return True

        logger.info(f"Starting download for {dataset_type} product {product_id} -> {local_path}...")
        db_manager.update_status(product_id, "downloading", local_path=str(local_path))
        
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            start_time = time.time()
            try:
                if download_url.startswith("sentinelhub://process"):
                    # Use Sentinel Hub Processing API
                    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
                    headers = auth_manager.get_auth_header()
                    headers["Content-Type"] = "application/json"
                    headers["Accept"] = "image/tiff"
                    
                    bbox = DISTRICT_BOUNDS.get(dist_name, {}).get("coordinates", [[[88.01, 20.59], [92.68, 20.59], [92.68, 26.63], [88.01, 26.63], [88.01, 20.59]]])[0]
                    # Convert to flat min_x, min_y, max_x, max_y bounding box
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    bbox_coords = [min(xs), min(ys), max(xs), max(ys)]
                    
                    payload = {
                        "input": {
                            "bounds": {
                                "properties": {
                                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                                },
                                "bbox": bbox_coords
                            },
                            "data": [
                                {
                                    "type": "sentinel-2-l2a",
                                    "dataFilter": {
                                        "timeRange": {
                                            "from": f"{date_str}T00:00:00Z",
                                            "to": f"{date_str}T23:59:59Z"
                                        },
                                        "maxCloudCoverage": 10
                                    }
                                }
                            ]
                        },
                        "output": {
                            "width": 512,
                            "height": 512,
                            "responses": [
                                {
                                    "identifier": "default",
                                    "format": {
                                        "type": "image/tiff"
                                    }
                                }
                            ]
                        },
                        "evalscript": """//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08"],
    output: { bands: 4 }
  };
}
function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02, sample.B08];
}"""
                    }
                    
                    response = requests.post(process_url, json=payload, headers=headers, timeout=60)
                    response.raise_for_status()
                    
                    # Save content directly
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                        
                    total_size = len(response.content)
                else:
                    # Public URL stream download
                    response = requests.get(download_url, stream=True, timeout=60, allow_redirects=True)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get("content-length", 0))
                    
                    progress_bar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {dataset_type}",
                        leave=True
                    )
                    
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                    progress_bar.close()
                    response.close()

                actual_size = local_path.stat().st_size
                end_time = time.time()
                duration = end_time - start_time
                download_speed_mb = (actual_size / (1024 * 1024)) / duration if duration > 0 else 0
                
                logger.info(f"Download complete: {local_path} ({actual_size} bytes) in {duration:.2f}s ({download_speed_mb:.2f} MB/s).")
                
                # Compute SHA256 checksum
                sha_hash = calculate_sha256(local_path)
                
                # Update SQLite DB
                db_manager.update_status(
                    product_id,
                    "completed",
                    size=actual_size,
                    download_time=f"{duration:.2f}s",
                    sha256_hash=sha_hash,
                    local_path=str(local_path)
                )
                return True

            except Exception as e:
                logger.error(f"Attempt {attempt}/{self.max_retries} failed for {product_id}: {e}")
                if attempt < self.max_retries:
                    time.sleep(5)
                else:
                    db_manager.update_status(product_id, "failed")
                    logger.error(f"All {self.max_retries} download attempts failed.")
                    return False
        
        return False
