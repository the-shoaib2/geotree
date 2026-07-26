import unittest
from pathlib import Path
from preprocessing.config.config import p_config
from preprocessing.validation.verifier import DataVerifier
from preprocessing.tiling.tiler import ImageTiler
from preprocessing.labels.converter import LabelConverter

class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self.verifier = DataVerifier()
        self.tiler = ImageTiler(tile_size=512, overlap=64)
        self.label_conv = LabelConverter()

    def test_config_loading(self):
        self.assertEqual(p_config.tile_size, 512)
        self.assertEqual(p_config.overlap, 64)
        self.assertTrue(p_config.augmentation)

    def test_label_conversion_yolo(self):
        # [xmin, ymin, w, h] on 1000x1000 image
        yolo_str = self.label_conv.to_yolo([100, 100, 200, 200], 1000, 1000)
        self.assertEqual(yolo_str, "0 0.200000 0.200000 0.200000 0.200000")

    def test_label_conversion_pascal(self):
        xml_str = self.label_conv.to_pascal_voc("test.png", [100, 100, 200, 200], 1000, 1000)
        self.assertIn("test.png", xml_str)
        self.assertIn("<xmin>100</xmin>", xml_str)

if __name__ == "__main__":
    unittest.main()
