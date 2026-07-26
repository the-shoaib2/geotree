import torch
from pathlib import Path
from preprocessing.pipeline.helpers import logger
from configs.train_config import train_config

class ModelExporter:
    def export_to_onnx(self, model: torch.nn.Module, output_path: Path) -> bool:
        """Converts PyTorch model weights to ONNX format."""
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, train_config.img_size, train_config.img_size).to(device)
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=11
            )
            logger.info(f"Model exported successfully to ONNX: {output_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX export failure: {e}")
            return False
            
    def export_to_torchscript(self, model: torch.nn.Module, output_path: Path) -> bool:
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, train_config.img_size, train_config.img_size).to(device)
        try:
            traced_cell = torch.jit.trace(model, dummy_input)
            traced_cell.save(str(output_path))
            logger.info(f"Model saved to TorchScript: {output_path}")
            return True
        except Exception as e:
            logger.error(f"TorchScript export failure: {e}")
            return False
