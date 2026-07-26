import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from preprocessing.pipeline.helpers import logger, ensure_dir

class ReportGenerator:
    def generate_html_report(self, summary: Dict[str, Any], output_path: Path) -> None:
        """Generates a premium visual HTML report detailing preprocessing run results."""
        ensure_dir(output_path.parent)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Bangladesh Tree AI Preprocessing Report</title>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #161b22;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2ea44f;
            border-bottom: 2px solid #21262d;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #21262d;
        }}
        th {{
            background-color: #21262d;
            color: #58a6ff;
        }}
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 25px;
        }}
        .card {{
            background: #21262d;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #2ea44f;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bangladesh Tree AI Preprocessing Run Report</h1>
        <p>This report summarizes the results of the geospatial preprocessing and tiling pipeline.</p>
        
        <div class="metric-cards">
            <div class="card">
                <div>Total Input Images</div>
                <div class="card-value">{summary['total_input_images']}</div>
            </div>
            <div class="card">
                <div>Generated Tiles</div>
                <div class="card-value">{summary['total_tiles']}</div>
            </div>
            <div class="card">
                <div>Augmented Tiles</div>
                <div class="card-value">{summary['total_augmented']}</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Stat Name</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Dataset Complete</td>
                    <td>{summary['is_complete']}</td>
                </tr>
                <tr>
                    <td>Dataset Total Size</td>
                    <td>{summary['total_size_mb']:.2f} MB</td>
                </tr>
                <tr>
                    <td>Processing Time</td>
                    <td>{summary['processing_time_sec']:.2f} seconds</td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Generated HTML Preprocessing Report at {output_path}")

    def generate_json_report(self, summary: Dict[str, Any], output_path: Path) -> None:
        ensure_dir(output_path.parent)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Generated JSON Preprocessing Report at {output_path}")
