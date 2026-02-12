import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from .dataset import load_dataset


def evaluate_model(model, X_test, y_test, class_names=("Real", "Fake")):
    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    return accuracy, y_pred, predictions


def predict_single_audio(model, preprocessor, audio_path, class_names=("Real", "Fake"), use_enhanced=True):
    audio = preprocessor.load_audio(audio_path)
    if audio is None:
        return None, None
    features = preprocessor.create_combined_features(audio, use_enhanced=use_enhanced)
    prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)
    predicted_idx = prediction.argmax()
    return class_names[predicted_idx], prediction[0][predicted_idx]


def _accuracy_wilson_ci(y_true, y_pred, z=1.96):
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    p = (y_true == y_pred).mean()
    denom = 1 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_true_b = y_true[idx]
        y_pred_b = y_pred[idx]
        scores.append(metric_fn(y_true_b, y_pred_b))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def _expected_calibration_error(y_true, y_prob, n_bins=10):
    y_conf = np.maximum(y_prob, 1 - y_prob)
    y_hat = (y_prob >= 0.5).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_conf > left) & (y_conf <= right)
        if not np.any(mask):
            continue
        acc_bin = np.mean(y_hat[mask] == y_true[mask])
        conf_bin = np.mean(y_conf[mask])
        ece += np.abs(acc_bin - conf_bin) * np.mean(mask)
    return float(ece)


def _optimal_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_values = (2 * precision * recall) / (precision + recall + 1e-8)
    if len(thresholds) == 0:
        return 0.5, float(f1_values.max())
    best_idx = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_idx]), float(f1_values[:-1][best_idx])


def test_on_test_set(model, preprocessor, dataset_path, use_enhanced=True, show_plots=True):
    X_test_full, y_test_full = load_dataset(
        dataset_path,
        preprocessor,
        max_files_per_class=None,
        use_enhanced_features=use_enhanced,
    )
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_test_full)
    y_categorical = to_categorical(y_encoded, num_classes=2)

    predictions = model.predict(X_test_full, verbose=1)
    y_pred = predictions.argmax(axis=1)
    y_true = y_categorical.argmax(axis=1)
    y_prob = predictions[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    threshold_opt, best_f1_opt = _optimal_f1_threshold(y_true, y_prob)
    y_pred_opt = (y_prob >= threshold_opt).astype(int)

    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred_default),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_default),
        "precision": precision_score(y_true, y_pred_default),
        "recall": recall_score(y_true, y_pred_default),
        "f1": f1_score(y_true, y_pred_default),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred_default),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, predictions),
        "ece": _expected_calibration_error(y_true, y_prob),
        "accuracy_ci_95": _accuracy_wilson_ci(y_true, y_pred_default),
        "f1_ci_95_bootstrap": _bootstrap_ci(y_true, y_pred_default, f1_score),
        "threshold_default": 0.5,
        "threshold_optimal_f1": threshold_opt,
        "f1_at_optimal_threshold": best_f1_opt,
        "f1_optimal_threshold_ci_95_bootstrap": _bootstrap_ci(y_true, y_pred_opt, f1_score),
        "classification_report": classification_report(y_true, y_pred_default, target_names=["Real", "Fake"]),
        "predictions": predictions,
        "y_true": y_true,
        "y_pred": y_pred_default,
    }

    print(metrics["classification_report"])
    print(
        f"Accuracy (95% CI): {metrics['accuracy']:.4f} "
        f"[{metrics['accuracy_ci_95'][0]:.4f}, {metrics['accuracy_ci_95'][1]:.4f}]"
    )
    print(
        f"F1 (95% bootstrap CI): {metrics['f1']:.4f} "
        f"[{metrics['f1_ci_95_bootstrap'][0]:.4f}, {metrics['f1_ci_95_bootstrap'][1]:.4f}]"
    )

    if show_plots:
        cm = confusion_matrix(y_true, y_pred_default)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.title("Confusion Matrix - Test Set")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {metrics['roc_auc']:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, lw=2, color="purple", label=f"PR-AUC = {metrics['pr_auc']:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.show()
    return metrics


def _safe_normalize(audio):
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak > 1.0:
        return audio / peak
    return audio


def _predict_from_audio(model, preprocessor, audio, use_enhanced=True):
    features = preprocessor.create_combined_features(audio, use_enhanced=use_enhanced)
    pred = model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
    return int(np.argmax(pred))


def evaluate_robustness_on_dataset(model, preprocessor, dataset_path, use_enhanced=True, max_samples=200):
    audio_extensions = (".wav", ".mp3", ".flac", ".m4a")
    samples = []

    for class_name, y_val in [("real", 0), ("fake", 1)]:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for name in os.listdir(class_path):
            if name.lower().endswith(audio_extensions):
                samples.append((os.path.join(class_path, name), y_val))

    if not samples:
        return {"error": "No audio files found for robustness evaluation."}

    samples = samples[: max_samples if max_samples else len(samples)]
    results = {"clean": [], "noisy": [], "pitch_shift": [], "speed_up": [], "compressed": []}
    y_true = []

    for file_path, y_val in samples:
        audio = preprocessor.load_audio(file_path)
        if audio is None:
            continue

        y_true.append(y_val)
        clean_audio = _safe_normalize(audio)
        noisy = _safe_normalize(clean_audio + np.random.normal(0, 0.005, len(clean_audio)))
        pitch_shift = _safe_normalize(librosa.effects.pitch_shift(clean_audio, sr=preprocessor.sample_rate, n_steps=2))
        speed_up = librosa.effects.time_stretch(clean_audio, rate=1.05)
        speed_up = _safe_normalize(
            np.pad(speed_up, (0, max(0, len(clean_audio) - len(speed_up))))[: len(clean_audio)]
        )
        compressed = _safe_normalize(librosa.effects.preemphasis(clean_audio) + np.random.normal(0, 0.002, len(clean_audio)))

        attack_map = {
            "clean": clean_audio,
            "noisy": noisy,
            "pitch_shift": pitch_shift,
            "speed_up": speed_up,
            "compressed": compressed,
        }
        for attack_name, attacked_audio in attack_map.items():
            y_hat = _predict_from_audio(model, preprocessor, attacked_audio, use_enhanced=use_enhanced)
            results[attack_name].append(y_hat)

    y_true = np.array(y_true, dtype=int)
    summary = {"n_samples": int(len(y_true))}
    for attack_name, y_hat in results.items():
        if len(y_hat) == 0:
            summary[f"{attack_name}_accuracy"] = 0.0
        else:
            summary[f"{attack_name}_accuracy"] = float(accuracy_score(y_true[: len(y_hat)], np.array(y_hat, dtype=int)))

    clean_acc = summary.get("clean_accuracy", 0.0)
    for attack_name in ["noisy", "pitch_shift", "speed_up", "compressed"]:
        summary[f"{attack_name}_drop_vs_clean"] = clean_acc - summary.get(f"{attack_name}_accuracy", 0.0)
    return summary
