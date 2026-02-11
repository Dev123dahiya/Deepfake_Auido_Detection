from pathlib import Path
import pandas as pd


def generate_testing_report(test_metrics, save_dir="outputs"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    status = (
        "EXCELLENT"
        if test_metrics["accuracy"] > 0.95
        else "GOOD"
        if test_metrics["accuracy"] > 0.85
        else "FAIR"
        if test_metrics["accuracy"] > 0.75
        else "NEEDS IMPROVEMENT"
    )

    report = f"""
============================================================
DEEPFAKE DETECTION - TEST REPORT
============================================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Accuracy:      {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)
Precision:     {test_metrics['precision']:.4f}
Recall:        {test_metrics['recall']:.4f}
F1-Score:      {test_metrics['f1']:.4f}
ROC-AUC Score: {test_metrics['roc_auc']:.4f}

Overall Rating: {status}
============================================================
""".strip()

    print(report)
    report_file = save_path / "testing_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_file}")
    return report_file

