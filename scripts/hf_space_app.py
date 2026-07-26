"""
GeoTree Gradio Interactive Demo App for Hugging Face Spaces
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from huggingface/model import TreeDetectorModel if Path("huggingface/model.py").exists() else None

# Fallback import if directory structure varies
try:
    from training.models.selector import TreeDetectorModel
except Exception:
    pass

model = TreeDetectorModel()
weights_path = Path("weights/best_model.pth")
if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

def detect_trees(input_image, confidence_threshold=0.3):
    if input_image is None:
        return None, "Please upload a satellite image."

    img = Image.fromarray(input_image).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((640, 640))

    tensor = torch.tensor(np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor).squeeze(0).tolist()

    conf = float(torch.sigmoid(torch.tensor(output[0])))
    bbox = [float(x) for x in output[1:]]

    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox
    
    if conf >= confidence_threshold and w > 0.01 and h > 0.01:
        x1 = max(0, (x - w / 2)) * orig_w
        y1 = max(0, (y - h / 2)) * orig_h
        x2 = min(1, (x + w / 2)) * orig_w
        y2 = min(1, (y + h / 2)) * orig_h

        draw.rectangle([x1, y1, x2, y2], outline="#2ea44f", width=4)
        draw.text((x1, max(0, y1 - 15)), f"Tree: {conf * 100:.1f}%", fill="#2ea44f")

        # Carbon calculation simulation
        crown_area_m2 = (w * 640 * 10) * (h * 640 * 10)
        biomass_kg = 5.83 * (crown_area_m2 ** 1.27)
        carbon_kg = biomass_kg * 0.47

        summary = f"""### 🌳 GeoTree Analysis Results
- **Detection**: Tree Crown Detected
- **Confidence**: {conf * 100:.1f}%
- **Bounding Box**: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]
- **Est. Above Ground Biomass**: {biomass_kg:.2f} kg
- **Est. Carbon Storage**: {carbon_kg:.2f} kg
- **Model Name**: `geotree`
"""
    else:
        summary = "### 🌳 GeoTree Analysis Results\nNo tree detected above threshold."

    return np.array(img), summary

demo = gr.Interface(
    fn=detect_trees,
    inputs=[
        gr.Image(label="Upload Satellite / Aerial Image Tile"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.3, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(label="GeoTree AI Bounding Box Overlay"),
        gr.Markdown(label="GIS Analytics Output")
    ],
    title="🌳 GeoTree — Deep Learning Tree Crown Detector",
    description="Upload any satellite/aerial imagery tile to run real-time tree detection and biomass/carbon analytics using the `geotree` model."
)

if __name__ == "__main__":
    demo.launch()
