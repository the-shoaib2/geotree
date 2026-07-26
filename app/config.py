import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.download_directory = "tree_dataset"
        self.max_parallel_downloads = 1
        self.cloud_cover = 10
        self.start_date = "2026-01-01"
        self.end_date = "2026-07-31"
        self.collection = "sentinel-2-l2a"
        self.processing_level = "LEVEL2A"
        
        self.deepforest_sample_url = ""
        self.deepforest_zenodo_url = ""
        self.selvabox_url = ""
        self.global_forest_change_url = ""
        
        self.load_yaml()
        
        # Load credentials from .env
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        
    def load_yaml(self):
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    self.download_directory = data.get("download_directory", self.download_directory)
                    self.max_parallel_downloads = int(data.get("max_parallel_downloads", self.max_parallel_downloads))
                    self.cloud_cover = float(data.get("cloud_cover", self.cloud_cover))
                    self.start_date = str(data.get("start_date", self.start_date))
                    self.end_date = str(data.get("end_date", self.end_date))
                    self.collection = str(data.get("collection", self.collection))
                    self.processing_level = str(data.get("processing_level", self.processing_level))
                    
                    self.deepforest_sample_url = data.get("deepforest_sample_url", self.deepforest_sample_url)
                    self.deepforest_zenodo_url = data.get("deepforest_zenodo_url", self.deepforest_zenodo_url)
                    self.selvabox_url = data.get("selvabox_url", self.selvabox_url)
                    self.global_forest_change_url = data.get("global_forest_change_url", self.global_forest_change_url)

    def validate(self):
        if not self.client_id or not self.client_secret:
            raise ValueError("CLIENT_ID and CLIENT_SECRET must be defined in the .env file.")
        
        # Ensure directories exist
        Path(self.download_directory).mkdir(parents=True, exist_ok=True)

# Shared config instance
config = Config()
try:
    config.validate()
except Exception as e:
    # We don't crash on import if variables are missing, but validate when running.
    pass
