import unittest
import torch
from pathlib import Path
from configs.train_config import train_config
from training.models.selector import TreeDetectorModel
from training.models.exporter import ModelExporter

class TestTreeAITraining(unittest.TestCase):
    def setUp(self):
        self.model = TreeDetectorModel()
        self.exporter = ModelExporter()

    def test_model_forward_pass(self):
        dummy_input = torch.randn(1, 3, 640, 640)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 5))

    def test_export_onnx(self):
        out_path = Path("exports/test_model.onnx")
        res = self.exporter.export_to_onnx(self.model, out_path)
        self.assertTrue(res)
        if out_path.exists():
            out_path.unlink()

if __name__ == "__main__":
    unittest.main()
