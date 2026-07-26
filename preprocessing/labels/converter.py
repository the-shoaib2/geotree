import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
from preprocessing.pipeline.helpers import logger

class LabelConverter:
    def to_yolo(self, bbox: List[float], img_w: int, img_h: int) -> str:
        """Converts [xmin, ymin, w, h] to YOLO normalized format: class x_center y_center width height."""
        xmin, ymin, w, h = bbox
        x_center = (xmin + w / 2.0) / img_w
        y_center = (ymin + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

    def to_pascal_voc(self, filename: str, bbox: List[float], img_w: int, img_h: int) -> str:
        """Converts to Pascal VOC XML structure."""
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = filename
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)
        ET.SubElement(size, "depth").text = "3"
        
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "tree"
        
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))
        
        return ET.tostring(root, encoding="utf-8").decode()

    def generate_coco_annotations(self, images: List[Dict[str, Any]], annotations: List[Dict[str, Any]]) -> str:
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "tree"}]
        }
        return json.dumps(coco_data, indent=4)
