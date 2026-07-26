from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import torch
from PIL import Image
import io
import numpy as np

from training.models.selector import TreeDetectorModel
from configs.train_config import train_config

app = FastAPI(title="Bangladesh Tree AI serving endpoint")

# Load model weights
model = TreeDetectorModel()
weights_path = Path("weights/best_model.pth")
if weights_path.exists():
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    except Exception:
        pass
model.eval()

@app.get("/")
def read_root():
    return {"status": "online", "model": "TreeDetectorModel"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts image bytes, passes through model, outputs tree confidence and bounding boxes."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((train_config.img_size, train_config.img_size))
    
    img_arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_arr).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor).squeeze(0).tolist()
        
    # outputs: [confidence, x, y, w, h]
    confidence = float(outputs[0])
    bbox = [float(x) for x in outputs[1:]]
    
    # Clip confidence to [0, 1] for serving layer sanity
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        "tree_detected": confidence > 0.5,
        "confidence": confidence,
        "bbox": bbox
    }
