import re
from typing import List, Dict, Any, Optional
from app.config import config
from app.logger import logger
from app.database import db_manager

def search_sentinel_images_for_district(
    district_name: str,
    date_str: str,
    cloud_cover: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Registers a Sentinel-2 image download entry for a specific district on a specific date.
    Since download is handled via Sentinel Hub Process API, we register a sentinelhub:// URI.
    """
    logger.info(f"Registering Sentinel-2 L2A search record for district '{district_name}' on date {date_str}...")
    
    product_id = f"s2_{district_name.lower()}_{date_str.replace('-', '')}"
    tile_name = f"S2_{district_name.upper()}"
    
    # We use a virtual protocol to signal the download manager to use Sentinel Hub Process API
    download_url = f"sentinelhub://process?district={district_name.lower()}&date={date_str}"
    
    product_record = {
        "product_id": product_id,
        "tile_name": tile_name,
        "date": date_str,
        "cloud_cover": 0.0,
        "size": 1048576,  # 1 MB estimated
        "download_status": "pending",
        "download_time": None,
        "sha256_hash": None,
        "local_path": None,
        "district": district_name.lower(),
        "dataset_type": "sentinel2"
    }
    
    # Save or update in database
    db_manager.add_or_update_product(product_record)
    product_record["download_url"] = download_url
    
    return [product_record]
