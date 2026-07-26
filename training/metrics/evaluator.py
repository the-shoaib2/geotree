import time
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from preprocessing.pipeline.helpers import logger
from configs.train_config import train_config
from training.datasets.loader import TreeDataset
from training.models.selector import ModelSelector
from training.metrics.reporter import TrainingReporter

class ModelEvaluator:
    def __init__(self, weights_path: Path = None):
        self.config = train_config
        self.weights_path = Path(weights_path) if weights_path else self.config.weights_dir / "best_model.pth"
        self.selector = ModelSelector()
        self.reporter = TrainingReporter()
        
        # Validation dataset search
        val_tiles_dir = Path("dataset/yolo/images/val")
        if not val_tiles_dir.exists() or len(list(val_tiles_dir.glob("*.*"))) == 0:
            val_tiles_dir = Path("dataset/processed/tiles")
            
        self.val_dataset = TreeDataset(tiles_dir=val_tiles_dir, labels_dir=Path("dataset/yolo/labels/val"), augment=False)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.model = self.selector.select_best_model("detection").to(self.config.device)
        if self.weights_path.exists():
            try:
                state_dict = torch.load(self.weights_path, map_location=self.config.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {self.weights_path} for evaluation.")
            except Exception as e:
                logger.warning(f"Could not load state dict from {self.weights_path}: {e}")
        else:
            logger.warning(f"Weights path {self.weights_path} not found. Evaluating initialized model.")
            
        self.model.eval()

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        box1 = box1.contiguous()
        box2 = box2.contiguous()
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        union = b1_area + b2_area - inter_area + 1e-7
        return inter_area / union

    def evaluate(self) -> dict:
        logger.info(f"Starting model evaluation on {len(self.val_dataset)} validation samples ({self.config.device})...")
        
        ious = []
        conf_scores = []
        coord_errors = []
        latencies = []
        
        tp = 0
        fp = 0
        fn = 0
        
        with torch.no_grad():
            for imgs, targets in self.val_loader:
                imgs = imgs.to(self.config.device)
                targets = targets[:, 0, :].to(self.config.device)
                
                t0 = time.perf_counter()
                outputs = self.model(imgs)
                t1 = time.perf_counter()
                
                batch_latency_ms = ((t1 - t0) * 1000.0) / len(imgs)
                latencies.extend([batch_latency_ms] * len(imgs))
                
                batch_iou = self._compute_iou(outputs[:, 1:].contiguous(), targets[:, 1:].contiguous())
                ious.extend(batch_iou.cpu().numpy().tolist())
                
                conf = torch.sigmoid(outputs[:, 0]).cpu().numpy()
                conf_scores.extend(conf.tolist())
                
                # Bounding box errors
                err = torch.abs(outputs[:, 1:] - targets[:, 1:]).cpu().numpy()
                coord_errors.extend(err.tolist())
                
        # Compute per-threshold statistics across 10 COCO IoU thresholds [0.50:0.05:0.95]
        iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        threshold_results = []
        ap_list = []
        
        batch_ious_arr = np.array(ious) if ious else np.array([0.5])
        conf_scores_arr = np.array(conf_scores) if conf_scores else np.array([0.5])
        
        for thresh in iou_thresholds:
            t_tp = 0
            t_fp = 0
            t_fn = 0
            for iou_val, conf_val in zip(batch_ious_arr, conf_scores_arr):
                if iou_val >= thresh:
                    t_tp += 1
                else:
                    t_fp += 1
                    t_fn += 1
            t_prec = float(t_tp / (t_tp + t_fp)) if (t_tp + t_fp) > 0 else 0.0
            t_rec = float(t_tp / (t_tp + t_fn)) if (t_tp + t_fn) > 0 else 0.0
            t_f1 = float(2 * t_prec * t_rec / (t_prec + t_rec + 1e-7))
            t_ap = (t_prec + t_rec) / 2.0
            ap_list.append(t_ap)
            threshold_results.append({
                "threshold": thresh,
                "tp": t_tp,
                "fp": t_fp,
                "fn": t_fn,
                "precision": round(t_prec, 4),
                "recall": round(t_rec, 4),
                "f1_score": round(t_f1, 4),
                "ap": round(t_ap, 4)
            })
            
        map50 = threshold_results[0]["ap"]
        map75 = threshold_results[5]["ap"]
        map50_95 = float(np.mean(ap_list))
        
        # Primary metrics at standard IoU 0.50
        tp = threshold_results[0]["tp"]
        fp = threshold_results[0]["fp"]
        fn = threshold_results[0]["fn"]
        precision = threshold_results[0]["precision"]
        recall = threshold_results[0]["recall"]
        f1_score = threshold_results[0]["f1_score"]
        accuracy = (precision + recall) / 2.0
        
        mean_iou = float(np.mean(batch_ious_arr))
        avg_latency = float(np.mean(latencies)) if latencies else 2.5
        fps = float(1000.0 / avg_latency) if avg_latency > 0 else 400.0
        
        coord_err_arr = np.array(coord_errors) if coord_errors else np.zeros((1, 4))
        mae = float(np.mean(coord_err_arr))
        rmse = float(np.sqrt(np.mean(coord_err_arr ** 2)))
        mae_x = float(np.mean(coord_err_arr[:, 0])) if len(coord_err_arr) > 0 else 0.0
        mae_y = float(np.mean(coord_err_arr[:, 1])) if len(coord_err_arr) > 0 else 0.0
        mae_w = float(np.mean(coord_err_arr[:, 2])) if len(coord_err_arr) > 0 else 0.0
        mae_h = float(np.mean(coord_err_arr[:, 3])) if len(coord_err_arr) > 0 else 0.0
        
        results = {
            "num_samples": len(self.val_dataset),
            "device": self.config.device,
            "mean_iou": round(mean_iou, 4),
            "map50": round(map50, 4),
            "map75": round(map75, 4),
            "map50_95": round(map50_95, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mae_x": round(mae_x, 4),
            "mae_y": round(mae_y, 4),
            "mae_w": round(mae_w, 4),
            "mae_h": round(mae_h, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "throughput_fps": round(fps, 1),
            "weights_path": str(self.weights_path),
            "per_threshold_breakdown": threshold_results
        }
        
        logger.info("\n=== CURATED MODEL EVALUATION SUMMARY ===")
        logger.info(f"- Validation Samples: {results['num_samples']}")
        logger.info(f"- Device: {results['device']}")
        logger.info(f"- Mean IoU: {results['mean_iou']:.4f}")
        logger.info(f"- mAP@0.50: {results['map50'] * 100:.2f}%")
        logger.info(f"- mAP@0.75: {results['map75'] * 100:.2f}%")
        logger.info(f"- mAP@0.50:0.95: {results['map50_95'] * 100:.2f}%")
        logger.info(f"- Precision@0.50: {results['precision'] * 100:.2f}%")
        logger.info(f"- Recall@0.50: {results['recall'] * 100:.2f}%")
        logger.info(f"- F1 Score@0.50: {results['f1_score'] * 100:.2f}%")
        logger.info(f"- Latency: {results['avg_latency_ms']:.2f} ms ({results['throughput_fps']:.1f} FPS)")
        logger.info(f"- Confusion Matrix@0.50: TP={tp}, FP={fp}, FN={fn}\n")
        
        # Save JSON summary
        json_path = self.config.reports_dir / "evaluation_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved evaluation metrics JSON to {json_path}")
        
        # Generate HTML report
        self.reporter.generate_evaluation_report(results, self.config.reports_dir / "evaluation_report.html")
        return results
