import yaml
from pathlib import Path

class PreprocessingConfig:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.tile_size = 512
        self.overlap = 64
        self.cloud_threshold = 20
        self.output_format = "PNG"
        self.train_ratio = 0.8
        self.validation_ratio = 0.1
        self.test_ratio = 0.1
        self.crs = "EPSG:4326"
        self.augmentation = True
        self.batch_size = 16
        self.workers = 4
        
        self.load_config()

    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    self.tile_size = int(data.get("tile_size", self.tile_size))
                    self.overlap = int(data.get("overlap", self.overlap))
                    self.cloud_threshold = float(data.get("cloud_threshold", self.cloud_threshold))
                    self.output_format = str(data.get("output_format", self.output_format))
                    self.train_ratio = float(data.get("train_ratio", self.train_ratio))
                    self.validation_ratio = float(data.get("validation_ratio", self.validation_ratio))
                    self.test_ratio = float(data.get("test_ratio", self.test_ratio))
                    self.crs = str(data.get("crs", self.crs))
                    self.augmentation = bool(data.get("augmentation", self.augmentation))
                    self.batch_size = int(data.get("batch_size", self.batch_size))
                    self.workers = int(data.get("workers", self.workers))

p_config = PreprocessingConfig()
