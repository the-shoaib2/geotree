#!/usr/bin/env python3
"""
Model Evaluation CLI Script
"""
import sys
from pathlib import Path
from training.metrics.evaluator import ModelEvaluator

def main():
    weights_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("weights/best_model.pth")
    evaluator = ModelEvaluator(weights_path=weights_path)
    results = evaluator.evaluate()
    print("\nModel evaluation completed successfully.")
    print(f"- mAP@0.5: {results['map50']*100:.2f}%")
    print(f"- Precision: {results['precision']*100:.2f}%")
    print(f"- Recall: {results['recall']*100:.2f}%")
    print(f"- F1 Score: {results['f1_score']*100:.2f}%")
    print(f"- Report HTML: reports/evaluation_report.html")

if __name__ == "__main__":
    main()
