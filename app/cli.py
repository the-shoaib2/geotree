import csv
import json
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

from app.config import config, DISTRICTS, DATES
from app.logger import logger
from app.auth import auth_manager
from app.search import search_sentinel_images_for_district
from app.downloader import DownloadManager
from app.database import db_manager
from app.checksum import calculate_sha256
from app.preparation import prepare_training_datasets
from app.verifier import DatasetVerifier

def execute_pipeline() -> None:
    """Executes the dataset collection pipeline: downloads, verifies, and repeats until 100% complete."""
    logger.info("=========================================")
    logger.info("STARTING TREE MONITORING DATASET PIPELINE")
    logger.info("=========================================")
    
    start_time_all = time.time()
    downloader = DownloadManager()
    verifier = DatasetVerifier()

    # Step 1: CDSE Authentication
    logger.info("\n--- STEP 1: Authenticating with CDSE ---")
    try:
        auth_manager.get_access_token()
    except Exception as e:
        logger.error(f"Failed to authenticate with CDSE: {e}")
        logger.warning("Continuing pipeline with Sentinel Hub fallback.")

    # Register additional public datasets in DB if not present
    additional_datasets = [
        {
            "product_id": "deepforest_sample",
            "tile_name": "DeepForest Sample",
            "date": "2026-07-26",
            "cloud_cover": 0.0,
            "size": 412520,
            "download_status": "pending",
            "download_url": config.deepforest_sample_url,
            "dataset_type": "deepforest"
        },
        {
            "product_id": "deepforest_zenodo",
            "tile_name": "DeepForest Zenodo",
            "date": "2026-07-26",
            "cloud_cover": 0.0,
            "size": 945773,
            "download_status": "pending",
            "download_url": config.deepforest_zenodo_url,
            "dataset_type": "zenodo"
        },
        {
            "product_id": "selvabox_partial",
            "tile_name": "SelvaBox Partial",
            "date": "2026-07-26",
            "cloud_cover": 0.0,
            "size": 12350,
            "download_url": config.selvabox_url,
            "download_status": "pending",
            "dataset_type": "selvabox"
        },
        {
            "product_id": "hansen_gfc_bangladesh",
            "tile_name": "Hansen GFC Bangladesh",
            "date": "2026-07-26",
            "cloud_cover": 0.0,
            "size": 685184395,
            "download_url": config.global_forest_change_url,
            "download_status": "pending",
            "dataset_type": "global_forest_change"
        }
    ]

    for item in additional_datasets:
        existing = db_manager.get_product(item["product_id"])
        if not existing:
            db_manager.add_or_update_product(item)

    # Main search, download, and verification loop
    max_loops = 5
    loop_count = 0
    
    while loop_count < max_loops:
        loop_count += 1
        logger.info(f"\n--- EXECUTION LOOP {loop_count}/{max_loops} ---")
        
        # Search & Download Sentinel-2 products
        logger.info("Searching Sentinel-2 imagery...")
        s2_products_to_download = []
        for district in DISTRICTS:
            for date in DATES:
                try:
                    products = search_sentinel_images_for_district(district, date)
                    s2_products_to_download.extend(products)
                except Exception as e:
                    logger.error(f"Failed searching S2 images for {district} on {date}: {e}")

        # Download Sentinel-2 products one by one
        for p in s2_products_to_download:
            # Check if already completed in DB
            db_record = db_manager.get_product(p["product_id"])
            if db_record and db_record["download_status"] == "completed":
                continue
            try:
                downloader.download_product(p, p["download_url"], is_sentinel=True)
            except Exception as e:
                logger.error(f"Failed S2 download for product {p['product_id']}: {e}")

        # Download public datasets
        for item in additional_datasets:
            db_record = db_manager.get_product(item["product_id"])
            if db_record and db_record["download_status"] == "completed":
                continue
            if item["download_url"]:
                try:
                    downloader.download_product(item, item["download_url"], is_sentinel=False)
                except Exception as e:
                    logger.error(f"Failed to download public dataset {item['product_id']}: {e}")

        # Run verification run
        logger.info("Running dataset integrity verification...")
        verif_result = verifier.verify_all()
        
        if verif_result["is_complete"]:
            logger.info("Dataset verification succeeded! 100% of files verified successfully.")
            break
        else:
            missing_ids = [m["product_id"] for m in verif_result["missing_files"]]
            logger.warning(f"Verification failed. {len(missing_ids)} files are missing/corrupted: {missing_ids}")
            # Reset missing file statuses to pending so they download again in the next loop
            for m in verif_result["missing_files"]:
                db_manager.update_status(m["product_id"], "pending")
    else:
        logger.critical(f"Dataset verification failed after {max_loops} attempts. AI training cannot start.")
        sys.exit(1)

    # Step 9: Generate metadata.json
    logger.info("\n--- STEP 9: Generating metadata.json ---")
    all_products = db_manager.get_all_products()
    metadata_path = Path(config.download_directory) / "metadata.json"
    metadata_list = []
    for p in all_products:
        metadata_list.append({
            "product_id": p["product_id"],
            "dataset_type": p["dataset_type"],
            "district": p["district"],
            "tile_name": p["tile_name"],
            "date": p["date"],
            "size_bytes": p["size"],
            "sha256_hash": p["sha256_hash"],
            "local_path": p["local_path"],
            "download_status": p["download_status"],
            "download_time": p["download_time"]
        })
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4)
    logger.info(f"Metadata file written to {metadata_path}")

    # Automatically prepare AI training folder structure
    prepare_training_datasets()

    duration_all = time.time() - start_time_all
    logger.info(f"Pipeline executed successfully in {duration_all:.2f} seconds.")

def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Tree Monitoring Dataset Collection & Verification Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Pipeline download/execution command
    subparsers.add_parser("download", help="Run the dataset pipeline (Download, Verify, Split, AI prepare)")
    
    # Standalone verification command
    subparsers.add_parser("verify", help="Scan folders, verify checksums, and generate reports")
    
    # List command
    subparsers.add_parser("list", help="List database products and download status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "download":
        try:
            execute_pipeline()
        except Exception as e:
            logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.command == "verify":
        try:
            verifier = DatasetVerifier()
            logger.info("Running standalone dataset verification...")
            verifier.verify_all()
        except Exception as e:
            logger.critical(f"Standalone verification failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.command == "list":
        try:
            products = db_manager.get_all_products()
            if not products:
                print("No products registered in the database.")
                return
            print(f"\nRegistered Datasets ({len(products)} total):")
            print(f"{'Product ID':<35} | {'Type':<12} | {'District':<10} | {'Status':<10} | {'Size (MB)':<10}")
            print("-" * 90)
            for p in products:
                size_mb = f"{p['size'] / (1024*1024):.2f}" if p['size'] else "N/A"
                print(f"{p['product_id']:<35} | {p['dataset_type']:<12} | {p.get('district') or 'N/A':<10} | {p['download_status']:<10} | {size_mb:<10}")
        except Exception as e:
            logger.error(f"List subcommand failed: {e}")
            sys.exit(1)
