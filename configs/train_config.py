import torch
from pathlib import Path

class TrainConfig:
    def __init__(self):
        self.epochs = 5
        self.img_size = 640
        self.weight_decay = 0.0005
        self.lr = 0.001
        self.batch_size = self._auto_select_batch_size()
        self.device = self._auto_detect_device()
        self.mixed_precision = self.device == "cuda"
        self.export_formats = ["onnx", "torchscript"]
        
        # Directories
        self.weights_dir = Path("weights")
        self.logs_dir = Path("logs")
        self.reports_dir = Path("reports")
        self.exports_dir = Path("exports")
        
        self._ensure_dirs()

    def _auto_detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _auto_select_batch_size(self) -> int:
        # Lower batch size on CPU/MPS, higher on CUDA
        device = self._auto_detect_device()
        if device == "cuda":
            return 16
        return 4

    def _ensure_dirs(self):
        for d in [self.weights_dir, self.logs_dir, self.reports_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)

train_config = TrainConfig()
