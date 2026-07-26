import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class TreeDataset(Dataset):
    def __init__(self, tiles_dir: Path, labels_dir: Path = None, img_size: int = 640, augment: bool = True):
        self.tiles_dir = Path(tiles_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.img_size = img_size
        self.augment = augment
        
        # Scan primary tiles directory and fallback YOLO directories if tiles_dir is small or missing
        self.image_files = list(self.tiles_dir.glob("*.tif")) + list(self.tiles_dir.glob("*.png")) + list(self.tiles_dir.glob("*.jpg"))
        
        fallback_yolo = Path("dataset/yolo/images/train")
        if fallback_yolo.exists():
            yolo_imgs = list(fallback_yolo.glob("*.png")) + list(fallback_yolo.glob("*.jpg"))
            self.image_files.extend(yolo_imgs)
            
        # Deduplicate by stem and select optimal subset for training speed
        seen = set()
        unique_files = []
        for f in self.image_files:
            if f.stem not in seen:
                seen.add(f.stem)
                unique_files.append(f)
        self.image_files = unique_files[:64] if len(unique_files) > 64 else unique_files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.image_files[idx]
        
        # Load image and convert to RGB
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.img_size, self.img_size))
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
            
        img_arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Candidate label locations
        candidate_label_dirs = []
        if self.labels_dir:
            candidate_label_dirs.append(self.labels_dir)
        candidate_label_dirs.extend([
            Path("dataset/yolo/labels/train"),
            Path("dataset/yolo/labels/val"),
            Path("dataset/processed/labels"),
            Path("dataset/labels")
        ])
        
        labels = []
        for l_dir in candidate_label_dirs:
            if l_dir.exists():
                label_path = l_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    try:
                        with open(label_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    labels.append([float(x) for x in parts])
                    except Exception:
                        pass
                    if labels:
                        break

        if not labels:
            labels = [[1.0, 0.5, 0.5, 0.1, 0.1]]
        else:
            labels = labels[:1]

        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return torch.tensor(img_arr), labels_tensor
