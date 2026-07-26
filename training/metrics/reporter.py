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
        for i in range(len(history["loss"])):
            epochs_row += f"""
            <tr>
                <td>{i+1}</td>
                <td>{history['loss'][i]:.4f}</td>
                <td>{history['precision'][i]*100:.2f}%</td>
                <td>{history['recall'][i]*100:.2f}%</td>
                <td>{history['f1_score'][i]*100:.2f}%</td>
            </tr>
            """
            
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Training Metrics Report</title>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            padding: 40px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #161b22;
            padding: 30px;
            border-radius: 12px;
            border: 1px solid #30363d;
        }}
        h1 {{
            color: #58a6ff;
            border-bottom: 2px solid #21262d;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #21262d;
        }}
        th {{
            background-color: #21262d;
            color: #2ea44f;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Training Run Results</h1>
        <p>Pre-training and training iteration parameters summary:</p>
        
        <table>
            <thead>
                <tr>
                    <th>Epoch</th>
                    <th>Average Loss</th>
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
