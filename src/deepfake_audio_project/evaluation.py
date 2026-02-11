import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
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


def test_on_test_set(model, preprocessor, dataset_path, use_enhanced=True):
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

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "predictions": predictions,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    cm = confusion_matrix(y_true, y_pred)
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
    return metrics
