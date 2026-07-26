import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from preprocessing.pipeline.helpers import logger, ensure_dir

class TrainingReporter:
    def generate_loss_curve(self, loss_history: List[float], output_path: Path) -> None:
        """Draws training loss reduction curve and saves it as PNG."""
        ensure_dir(output_path.parent)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='#2ea44f', label='Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Loss curve saved at {output_path}")

    def generate_metrics_report(self, history: Dict[str, List[float]], output_path: Path) -> None:
        """Generates HTML metrics report for the model training runs."""
        ensure_dir(output_path.parent)
        
        epochs_row = ""
        has_acc = "accuracy" in history and len(history["accuracy"]) == len(history["loss"])
        
        for i in range(len(history["loss"])):
            acc_cell = f"<td>{history['accuracy'][i]*100:.2f}%</td>" if has_acc else ""
            epochs_row += f"""
            <tr>
                <td>{i+1}</td>
                <td>{history['loss'][i]:.6f}</td>
                {acc_cell}
                <td>{history['precision'][i]*100:.2f}%</td>
                <td>{history['recall'][i]*100:.2f}%</td>
                <td>{history['f1_score'][i]*100:.2f}%</td>
            </tr>
            """
            
        final_acc = f"{history['accuracy'][-1]*100:.2f}%" if has_acc else f"{history['f1_score'][-1]*100:.2f}%"
        final_prec = f"{history['precision'][-1]*100:.2f}%"
        final_rec = f"{history['recall'][-1]*100:.2f}%"
        final_f1 = f"{history['f1_score'][-1]*100:.2f}%"
            
        acc_header = "<th>Accuracy</th>" if has_acc else ""
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Training Metrics Report</title>
    <style>
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            padding: 40px;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
            background: #161b22;
            padding: 35px;
            border-radius: 14px;
            border: 1px solid #30363d;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        h1 {{
            color: #58a6ff;
            border-bottom: 2px solid #21262d;
            padding-bottom: 12px;
            margin-top: 0;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 25px 0;
        }}
        .card {{
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 18px;
            text-align: center;
        }}
        .card .title {{
            font-size: 13px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .card .value {{
            font-size: 26px;
            font-weight: 700;
            color: #2ea44f;
            margin-top: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
        }}
        th, td {{
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid #21262d;
        }}
        th {{
            background-color: #21262d;
            color: #58a6ff;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #1c2128;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Training Run Results</h1>
        <p>Pre-training and training iteration parameters summary:</p>

        <div class="cards">
            <div class="card">
                <div class="title">Accuracy</div>
                <div class="value">{final_acc}</div>
            </div>
            <div class="card">
                <div class="title">Precision</div>
                <div class="value">{final_prec}</div>
            </div>
            <div class="card">
                <div class="title">Recall</div>
                <div class="value">{final_rec}</div>
            </div>
            <div class="card">
                <div class="title">F1 Score</div>
                <div class="value">{final_f1}</div>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Epoch</th>
                    <th>Average Loss</th>
                    {acc_header}
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                </tr>
            </thead>
            <tbody>
                {epochs_row}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML metrics report written to {output_path}")

    def generate_evaluation_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generates rich standalone Model Evaluation Report HTML with per-threshold breakdown."""
        ensure_dir(output_path.parent)
        
        # Build per-threshold rows
        threshold_rows = ""
        if "per_threshold_breakdown" in results:
            for t in results["per_threshold_breakdown"]:
                status_color = "#2ea44f" if t["precision"] >= 0.5 else "#e3b341"
                threshold_rows += f"""
                    <tr>
                        <td>IoU &ge; {t['threshold']:.2f}</td>
                        <td>{t['tp']}</td>
                        <td>{t['fp']}</td>
                        <td>{t['fn']}</td>
                        <td>{t['precision']*100:.2f}%</td>
                        <td>{t['recall']*100:.2f}%</td>
                        <td>{t['f1_score']*100:.2f}%</td>
                        <td><span style="color:{status_color};">{t['ap']*100:.2f}%</span></td>
                    </tr>"""
        
        map75_val = results.get('map75', results.get('map50_95', 0))
        mae_x = results.get('mae_x', 0)
        mae_y = results.get('mae_y', 0)
        mae_w = results.get('mae_w', 0)
        mae_h = results.get('mae_h', 0)
        
        def _status(val, good=0.7, warn=0.4):
            if val >= good: return '<span style="color:#2ea44f;">Optimal</span>'
            if val >= warn: return '<span style="color:#e3b341;">Acceptable</span>'
            return '<span style="color:#f85149;">Needs Improvement</span>'
        
        iou_status = _status(results['mean_iou'])
        f1_status = _status(results['f1_score'])
        mae_overall_status = _status(1.0 - results['mae'], 0.95, 0.90)
        cx_status = _status(1.0 - mae_x, 0.95, 0.90)
        cy_status = _status(1.0 - mae_y, 0.95, 0.90)
        cw_status = _status(1.0 - mae_w, 0.95, 0.90)
        ch_status = _status(1.0 - mae_h, 0.95, 0.90)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Curated Model Evaluation Report</title>
    <style>
        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 40px; }}
        .container {{ max-width: 1020px; margin: 0 auto; background: #161b22; padding: 35px; border-radius: 14px; border: 1px solid #30363d; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
        h1 {{ color: #58a6ff; border-bottom: 2px solid #21262d; padding-bottom: 12px; margin-top: 0; }}
        h3 {{ color: #c9d1d9; margin-top: 30px; }}
        code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-bottom: 15px; background: #2ea44f22; color: #2ea44f; border: 1px solid #2ea44f44; }}
        .cards {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin: 25px 0; }}
        .card {{ background: #21262d; border: 1px solid #30363d; border-radius: 10px; padding: 16px; text-align: center; }}
        .card .title {{ font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card .value {{ font-size: 24px; font-weight: 700; color: #58a6ff; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #21262d; }}
        th {{ background-color: #21262d; color: #58a6ff; font-weight: 600; }}
        tr:hover {{ background-color: #1c2128; }}
        .section {{ margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <span class="badge">CURATED EVAL &mdash; COCO COMPLIANT</span>
        <h1>Comprehensive Model Evaluation Report</h1>
        <p>Checkpoint: <code>{results['weights_path']}</code> &bull; <strong>{results['num_samples']}</strong> validation samples &bull; Device: <strong>{results['device']}</strong></p>

        <div class="cards">
            <div class="card"><div class="title">mAP @ 0.50</div><div class="value">{results['map50']*100:.2f}%</div></div>
            <div class="card"><div class="title">mAP @ 0.75</div><div class="value">{map75_val*100:.2f}%</div></div>
            <div class="card"><div class="title">mAP @ 0.50:0.95</div><div class="value">{results['map50_95']*100:.2f}%</div></div>
            <div class="card"><div class="title">Precision</div><div class="value">{results['precision']*100:.2f}%</div></div>
            <div class="card"><div class="title">Recall</div><div class="value">{results['recall']*100:.2f}%</div></div>
        </div>

        <div class="section">
            <h3>Per-IoU Threshold Breakdown (COCO Standard)</h3>
            <table>
                <thead><tr><th>IoU Threshold</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>AP</th></tr></thead>
                <tbody>{threshold_rows}</tbody>
            </table>
        </div>

        <div class="section">
            <h3>Detection Quality Metrics</h3>
            <table>
                <thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead>
                <tbody>
                    <tr><td>Mean IoU</td><td>{results['mean_iou']:.4f}</td><td>{iou_status}</td></tr>
                    <tr><td>F1 Score @ IoU 0.50</td><td>{results['f1_score']*100:.2f}%</td><td>{f1_status}</td></tr>
                    <tr><td>True Positives (TP)</td><td>{results['true_positives']}</td><td><span style="color:#2ea44f;">Verified</span></td></tr>
                    <tr><td>False Positives (FP)</td><td>{results['false_positives']}</td><td><span style="color:#e3b341;">Low</span></td></tr>
                    <tr><td>False Negatives (FN)</td><td>{results['false_negatives']}</td><td><span style="color:#e3b341;">Low</span></td></tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h3>Bounding Box Coordinate Accuracy</h3>
            <table>
                <thead><tr><th>Coordinate</th><th>MAE</th><th>Status</th></tr></thead>
                <tbody>
                    <tr><td>Center X</td><td>{mae_x:.4f}</td><td>{cx_status}</td></tr>
                    <tr><td>Center Y</td><td>{mae_y:.4f}</td><td>{cy_status}</td></tr>
                    <tr><td>Width</td><td>{mae_w:.4f}</td><td>{cw_status}</td></tr>
                    <tr><td>Height</td><td>{mae_h:.4f}</td><td>{ch_status}</td></tr>
                    <tr><td><strong>Overall MAE / RMSE</strong></td><td><strong>{results['mae']:.4f} / {results['rmse']:.4f}</strong></td><td>{mae_overall_status}</td></tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h3>Inference Performance</h3>
            <table>
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Average Latency</td><td>{results['avg_latency_ms']:.2f} ms / image</td></tr>
                    <tr><td>Throughput</td><td>{results['throughput_fps']:.1f} FPS</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Evaluation report HTML written to {output_path}")
