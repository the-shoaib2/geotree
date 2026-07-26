import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_DIR = Path("database")
DB_PATH = DB_DIR / "sentinel.db"

class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS downloads (
                    product_id TEXT PRIMARY KEY,
                    tile_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    cloud_cover REAL,
                    size INTEGER,
                    download_status TEXT NOT NULL,
                    download_time TEXT,
                    sha256_hash TEXT,
                    local_path TEXT,
                    district TEXT,
                    dataset_type TEXT DEFAULT 'sentinel2'
                )
            """)
            conn.commit()

    def add_or_update_product(self, product: Dict[str, Any]) -> None:
        """Inserts a new product or updates an existing product record."""
        # Ensure default keys
        prod = {
            "product_id": product["product_id"],
            "tile_name": product.get("tile_name", "UNKNOWN"),
            "date": product.get("date", "UNKNOWN"),
            "cloud_cover": product.get("cloud_cover", 0.0),
            "size": product.get("size"),
            "download_status": product.get("download_status", "pending"),
            "download_time": product.get("download_time"),
            "sha256_hash": product.get("sha256_hash"),
            "local_path": product.get("local_path"),
            "district": product.get("district"),
            "dataset_type": product.get("dataset_type", "sentinel2")
        }
        
        query = """
            INSERT INTO downloads (
                product_id, tile_name, date, cloud_cover, size, 
                download_status, download_time, sha256_hash, local_path,
                district, dataset_type
            ) VALUES (
                :product_id, :tile_name, :date, :cloud_cover, :size, 
                :download_status, :download_time, :sha256_hash, :local_path,
                :district, :dataset_type
            )
            ON CONFLICT(product_id) DO UPDATE SET
                tile_name = excluded.tile_name,
                date = excluded.date,
                cloud_cover = excluded.cloud_cover,
                size = excluded.size,
                download_status = excluded.download_status,
                download_time = excluded.download_time,
                sha256_hash = excluded.sha256_hash,
                local_path = excluded.local_path,
                district = excluded.district,
                dataset_type = excluded.dataset_type
        """
        with self._get_connection() as conn:
            conn.execute(query, prod)
            conn.commit()

    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM downloads WHERE product_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(query, (product_id,)).fetchone()
            return dict(row) if row else None

    def get_all_products(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM downloads"
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    def update_status(self, product_id: str, status: str, **kwargs) -> None:
        """Updates the status and any other fields dynamically."""
        updates = ["download_status = ?"]
        params = [status]
        
        for key, val in kwargs.items():
            updates.append(f"{key} = ?")
            params.append(val)
            
        params.append(product_id)
        query = f"UPDATE downloads SET {', '.join(updates)} WHERE product_id = ?"
        
        with self._get_connection() as conn:
            conn.execute(query, tuple(params))
            conn.commit()

# Shared database manager instance
db_manager = DatabaseManager()
