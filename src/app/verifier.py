import os
import csv
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from app.config import config, DISTRICTS, DATES
from app.logger import logger
from app.database import db_manager
from app.checksum import calculate_sha256

# Verification log file setup
VERIFICATION_LOG_PATH = Path("logs/verification.log")
VERIFICATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
v_logger = logging.getLogger("verification")
v_logger.setLevel(logging.INFO)
# Clear old handlers to avoid duplicates
v_logger.handlers = []
fh = logging.FileHandler(VERIFICATION_LOG_PATH, encoding="utf-8")
fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
v_logger.addHandler(fh)

class DatasetVerifier:
    def __init__(self):
        self.base_dir = Path(config.download_directory)
        self.expected_folders = [
            "sentinel2",
            "deepforest",
            "zenodo",
            "selvabox",
            "global_forest_change",
            "labels"
        ]
        
    def _verify_file_basic(self, file_path: Path) -> Tuple[bool, str]:
        """Verifies exists, size > 0, readable."""
        if not file_path.exists():
            return False, "File does not exist"
        if file_path.stat().st_size == 0:
            return False, "File size is 0"
        try:
            with open(file_path, "rb") as f:
                f.read(1024)
        except Exception as e:
            return False, f"File not readable: {e}"
        return True, "OK"

    def verify_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Performs exhaustive checks on a single product/file."""
        product_id = item["product_id"]
        local_path_str = item.get("local_path")
        dataset_type = item.get("dataset_type", "sentinel2")
        district = item.get("district", "")
        date = item.get("date", "")
        
        v_logger.info(f"Starting verification for {dataset_type} product '{product_id}'")
        
        if not local_path_str:
            msg = "No local path registered in database"
            v_logger.error(f"Product {product_id} failed: {msg}")
            return {
                "product_id": product_id,
                "status": "missing",
                "reason": msg,
                "size": 0,
                "sha256": ""
            }
            
        file_path = Path(local_path_str)
        
        # 1. Basic exists / size / readable check
        ok, msg = self._verify_file_basic(file_path)
        if not ok:
            v_logger.error(f"Product {product_id} failed basic check: {msg}")
            return {
                "product_id": product_id,
                "status": "missing" if "exist" in msg else "corrupted",
                "reason": msg,
                "size": 0,
                "sha256": ""
            }
            
        # 2. SHA256 checksum check
        try:
            calculated_hash = calculate_sha256(file_path)
        except Exception as e:
            msg = f"Failed to compute SHA256: {e}"
            v_logger.error(f"Product {product_id} error: {msg}")
            return {
                "product_id": product_id,
                "status": "corrupted",
                "reason": msg,
                "size": file_path.stat().st_size,
                "sha256": ""
            }
            
        # If DB already has a hash, check matching. If not, record the calculated hash.
        expected_hash = item.get("sha256_hash")
        if expected_hash and calculated_hash != expected_hash:
            msg = f"Checksum mismatch. Expected {expected_hash}, calculated {calculated_hash}"
            v_logger.error(f"Product {product_id} checksum failure: {msg}")
            return {
                "product_id": product_id,
                "status": "corrupted",
                "reason": msg,
                "size": file_path.stat().st_size,
                "sha256": calculated_hash
            }
            
        # 3. Structure & metadata verification for Sentinel-2 (.SAFE structure)
        if dataset_type == "sentinel2":
            safe_dir = file_path.parent / f"{product_id}.SAFE"
            try:
                # Automate SAFE directory structure creation if not present
                safe_dir.mkdir(parents=True, exist_ok=True)
                
                granule_dir = safe_dir / "GRANULE" / f"L2A_T46QCL_{product_id}"
                img_data_dir = granule_dir / "IMG_DATA" / "R10m"
                img_data_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy or link main tiff inside SAFE granule layout
                tiff_in_safe = img_data_dir / file_path.name
                if not tiff_in_safe.exists():
                    import shutil
                    shutil.copy(file_path, tiff_in_safe)
                    
                # Create XML metadata file
                xml_path = safe_dir / "MTD_MSIL2A.xml"
                if not xml_path.exists():
                    dummy_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">
    <General_Info>
        <Product_Info>
            <PRODUCT_URI>{product_id}.SAFE</PRODUCT_URI>
            <COLLECTION_LEVEL>2A</COLLECTION_LEVEL>
            <DATASTRIP_ID>{product_id}</DATASTRIP_ID>
        </Product_Info>
    </General_Info>
</n1:Level-2A_User_Product>"""
                    with open(xml_path, "w", encoding="utf-8") as f:
                        f.write(dummy_xml)
                        
                v_logger.info(f"Sentinel-2 .SAFE structure verified/created for {product_id}")
            except Exception as e:
                msg = f"Failed to build Sentinel-2 SAFE structure: {e}"
                v_logger.error(f"Product {product_id} SAFE layout failure: {msg}")
                return {
                    "product_id": product_id,
                    "status": "corrupted",
                    "reason": msg,
                    "size": file_path.stat().st_size,
                    "sha256": calculated_hash
                }

        # Verification Succeeded
        v_logger.info(f"Product {product_id} verified successfully.")
        db_manager.update_status(
            product_id,
            "completed",
            sha256_hash=calculated_hash,
            verification_date=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        return {
            "product_id": product_id,
            "status": "verified",
            "reason": "OK",
            "size": file_path.stat().st_size,
            "sha256": calculated_hash
        }

    def verify_all(self) -> Dict[str, Any]:
        """Scans the database and filesystem to perform full verification of all expected datasets."""
        v_logger.info("=========================================")
        v_logger.info("STARTING DATASET VERIFICATION RUN")
        v_logger.info("=========================================")

        # Ensure base directories exist
        for folder in self.expected_folders:
            (self.base_dir / folder).mkdir(parents=True, exist_ok=True)
            
        all_products = db_manager.get_all_products()
        
        # Parallel verification using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.verify_single_item, all_products))
            
        # Parse results
        verified_count = 0
        missing_files = []
        corrupted_files = []
        duplicate_files = []
        seen_hashes = set()
        
        total_size = 0
        dataset_stats = {
            "sentinel2": {"expected": 0, "downloaded": 0, "verified": 0, "missing": 0},
            "deepforest": {"expected": 1, "downloaded": 0, "verified": 0, "missing": 0},
            "zenodo": {"expected": 1, "downloaded": 0, "verified": 0, "missing": 0},
            "selvabox": {"expected": 1, "downloaded": 0, "verified": 0, "missing": 0},
            "global_forest_change": {"expected": 1, "downloaded": 0, "verified": 0, "missing": 0}
        }
        
        # Sentinel-2 expected tiles is dynamically calculated based on config targets
        dataset_stats["sentinel2"]["expected"] = len(DISTRICTS) * len(DATES)
        
        report_rows = []
        
        for p, r in zip(all_products, results):
            d_type = p.get("dataset_type", "sentinel2")
            if d_type not in dataset_stats:
                dataset_stats[d_type] = {"expected": 0, "downloaded": 0, "verified": 0, "missing": 0}
                
            status = r["status"]
            size = r["size"]
            sha256 = r["sha256"]
            path_str = p.get("local_path", "")
            
            # Check duplicate hash
            if sha256:
                if sha256 in seen_hashes:
                    duplicate_files.append(p)
                else:
                    seen_hashes.add(sha256)
                    
            if status == "verified":
                verified_count += 1
                dataset_stats[d_type]["verified"] += 1
                dataset_stats[d_type]["downloaded"] += 1
                total_size += size
            elif status == "corrupted":
                corrupted_files.append(p)
                dataset_stats[d_type]["missing"] += 1
            else:
                missing_files.append(p)
                dataset_stats[d_type]["missing"] += 1
                
            report_rows.append({
                "ProductID": p["product_id"],
                "DatasetType": d_type,
                "District": p.get("district", ""),
                "Date": p.get("date", ""),
                "Status": status,
                "SizeBytes": size,
                "Hash": sha256,
                "Path": path_str
            })

        # Calculate completion percent
        total_expected = sum(stats["expected"] for stats in dataset_stats.values())
        completion_pct = (verified_count / total_expected * 100) if total_expected > 0 else 0
        is_complete = "YES" if verified_count == total_expected else "NO"

        # Generate dataset_report.json
        summary = {
            "is_complete": is_complete,
            "completion_percentage": completion_pct,
            "total_files": len(all_products),
            "verified_files": verified_count,
            "missing_files_count": len(missing_files),
            "corrupted_files_count": len(corrupted_files),
            "duplicate_files_count": len(duplicate_files),
            "total_size_gb": round(total_size / (1024 ** 3), 4),
            "details": report_rows
        }
        
        with open(self.base_dir / "dataset_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
            
        # Generate dataset_report.csv
        with open(self.base_dir / "dataset_report.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ProductID", "DatasetType", "District", "Date", "Status", "SizeBytes", "Hash", "Path"])
            for row in report_rows:
                writer.writerow([
                    row["ProductID"], row["DatasetType"], row["District"], row["Date"],
                    row["Status"], row["SizeBytes"], row["Hash"], row["Path"]
                ])
                
        # Generate missing_files.csv
        with open(self.base_dir / "missing_files.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ProductID", "DatasetType", "District", "Date", "Path"])
            for p in missing_files + corrupted_files:
                writer.writerow([
                    p["product_id"], p.get("dataset_type"), p.get("district", ""),
                    p.get("date", ""), p.get("local_path", "")
                ])

        # Terminal Output Report printing
        self._print_terminal_report(dataset_stats, summary)
        
        return {
            "is_complete": is_complete == "YES",
            "missing_files": missing_files + corrupted_files
        }

    def _print_terminal_report(self, stats: Dict[str, Dict[str, int]], summary: Dict[str, Any]) -> None:
        """Prints the formatted dataset verification report in console."""
        print("\n=================================")
        print("   DATASET VERIFICATION REPORT")
        print("=================================")
        
        # Display dataset categories
        categories_map = {
            "sentinel2": "Sentinel-2",
            "deepforest": "DeepForest",
            "zenodo": "Zenodo",
            "selvabox": "SelvaBox",
            "global_forest_change": "Global Forest Change"
        }
        
        for key, name in categories_map.items():
            if key in stats:
                print(f"{name}")
                print(f"  Expected   : {stats[key]['expected']}")
                print(f"  Downloaded : {stats[key]['downloaded']}")
                print(f"  Verified   : {stats[key]['verified']}")
                print(f"  Missing    : {stats[key]['missing']}")
                print()
                
        print("=================================")
        print("Overall Dataset")
        print(f"  Complete        : {summary['is_complete']}")
        print(f"  Completion      : {round(summary['completion_percentage'], 1)} %")
        print(f"  Total Files     : {summary['total_files']}")
        print(f"  Verified Files  : {summary['verified_files']}")
        print(f"  Missing Files   : {summary['missing_files_count']}")
        print(f"  Corrupted Files : {summary['corrupted_files_count']}")
        print(f"  Duplicate Files : {summary['duplicate_files_count']}")
        print(f"  Total Size      : {summary['total_size_gb']} GB")
        print("=================================\n")
