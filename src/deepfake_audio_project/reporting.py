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
Samples:       {test_metrics.get('n_samples', 'NA')}

Accuracy:      {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)
Accuracy 95% CI: [{test_metrics.get('accuracy_ci_95', (0.0, 0.0))[0]:.4f}, {test_metrics.get('accuracy_ci_95', (0.0, 0.0))[1]:.4f}]
Balanced Acc:  {test_metrics.get('balanced_accuracy', 0.0):.4f}
Precision:     {test_metrics['precision']:.4f}
Recall:        {test_metrics['recall']:.4f}
F1-Score:      {test_metrics['f1']:.4f}
F1 95% CI:     [{test_metrics.get('f1_ci_95_bootstrap', (0.0, 0.0))[0]:.4f}, {test_metrics.get('f1_ci_95_bootstrap', (0.0, 0.0))[1]:.4f}]
ROC-AUC Score: {test_metrics['roc_auc']:.4f}
PR-AUC Score:  {test_metrics.get('pr_auc', 0.0):.4f}
MCC:           {test_metrics.get('mcc', 0.0):.4f}
Brier Score:   {test_metrics.get('brier_score', 0.0):.4f}
Log Loss:      {test_metrics.get('log_loss', 0.0):.4f}
ECE:           {test_metrics.get('ece', 0.0):.4f}

Threshold (default):     {test_metrics.get('threshold_default', 0.5):.3f}
Threshold (optimal F1):  {test_metrics.get('threshold_optimal_f1', 0.5):.3f}
F1 @ optimal threshold:  {test_metrics.get('f1_at_optimal_threshold', 0.0):.4f}

Overall Rating: {status}
============================================================
""".strip()

    robustness = test_metrics.get("robustness")
    if robustness:
        robustness_lines = [
            "",
            "ROBUSTNESS (ATTACK SIMULATION):",
            f"Samples tested: {robustness.get('n_samples', 0)}",
            f"Clean accuracy: {robustness.get('clean_accuracy', 0.0):.4f}",
            f"Noisy accuracy: {robustness.get('noisy_accuracy', 0.0):.4f} (drop {robustness.get('noisy_drop_vs_clean', 0.0):.4f})",
            f"Pitch-shift accuracy: {robustness.get('pitch_shift_accuracy', 0.0):.4f} (drop {robustness.get('pitch_shift_drop_vs_clean', 0.0):.4f})",
            f"Speed-up accuracy: {robustness.get('speed_up_accuracy', 0.0):.4f} (drop {robustness.get('speed_up_drop_vs_clean', 0.0):.4f})",
            f"Compressed accuracy: {robustness.get('compressed_accuracy', 0.0):.4f} (drop {robustness.get('compressed_drop_vs_clean', 0.0):.4f})",
        ]
        report = report + "\n" + "\n".join(robustness_lines)

    print(report)
    report_file = save_path / "testing_report.txt"
    report_file.write_text(report, encoding="utf-8")
    if test_metrics.get("classification_report"):
        class_report_file = save_path / "classification_report.txt"
        class_report_file.write_text(test_metrics["classification_report"], encoding="utf-8")
        print(f"Classification report saved to: {class_report_file}")
    print(f"Report saved to: {report_file}")
    return report_file
