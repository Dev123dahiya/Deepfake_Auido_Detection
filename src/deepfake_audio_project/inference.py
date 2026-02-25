import os

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .forensics_api import DeepfakeForensicsIntegration
from .security import AuditLogger, DriftMonitor, RiskPolicy, is_ood_prediction


def _predict_single_audio_label_conf(model, preprocessor, audio_path, class_names=("Real", "Fake"), use_enhanced=True):
    audio = preprocessor.load_audio(audio_path)
    if audio is None:
        return None, None
    features = preprocessor.create_combined_features(audio, use_enhanced=use_enhanced)
    prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)
    predicted_idx = int(prediction.argmax())
    return class_names[predicted_idx], float(prediction[0][predicted_idx])


def predict_with_ensemble(model, preprocessor, audio_path, use_api_ensemble=False, elevenlabs_key=None, resemble_key=None):
    predicted_class, confidence = _predict_single_audio_label_conf(
        model, preprocessor, audio_path, class_names=("Real", "Fake"), use_enhanced=True
    )
    if predicted_class is None:
        return None
    if not use_api_ensemble:
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "note": "API ensemble disabled. Set use_api_ensemble=True to enable.",
        }

    forensics = DeepfakeForensicsIntegration(elevenlabs_key=elevenlabs_key, resemble_key=resemble_key)
    return forensics.ensemble_with_external_apis(audio_path, predicted_class, confidence)


def test_single_audio(model, preprocessor, audio_path, use_enhanced=True):
    if not os.path.exists(audio_path):
        print(f"Error: File not found - {audio_path}")
        return None
    audio = preprocessor.load_audio(audio_path)
    if audio is None:
        print("Error: Could not load audio file")
        return None

    features = preprocessor.create_combined_features(audio, use_enhanced=use_enhanced)
    prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)
    predicted_idx = prediction.argmax()
    predicted_class = "Fake" if predicted_idx == 1 else "Real"
    confidence = prediction[0][predicted_idx]
    return {
        "file": os.path.basename(audio_path),
        "prediction": predicted_class,
        "confidence": confidence,
        "real_prob": prediction[0][0],
        "fake_prob": prediction[0][1],
    }


def secure_predict_single_audio(
    model,
    preprocessor,
    audio_path,
    use_enhanced=True,
    ood_confidence_threshold=0.60,
    ood_entropy_threshold=0.68,
    low_risk_threshold=0.30,
    high_risk_threshold=0.75,
    audit_log_path="outputs/security_audit.jsonl",
    drift_monitor=None,
):
    base_result = test_single_audio(model, preprocessor, audio_path, use_enhanced=use_enhanced)
    if not base_result:
        return None

    probs = np.array([base_result["real_prob"], base_result["fake_prob"]], dtype=float)
    ood_info = is_ood_prediction(
        probs,
        confidence_threshold=ood_confidence_threshold,
        entropy_threshold=ood_entropy_threshold,
    )
    risk_policy = RiskPolicy(low_risk_threshold=low_risk_threshold, high_risk_threshold=high_risk_threshold)
    decision = risk_policy.classify(base_result["fake_prob"], is_ood=ood_info["is_ood"])

    if drift_monitor is None:
        drift_monitor = DriftMonitor()
    drift_info = drift_monitor.update(base_result["fake_prob"])

    result = {
        **base_result,
        "ood": ood_info,
        "security_decision": decision,
        "drift": drift_info,
    }

    AuditLogger(log_path=audit_log_path).log(
        {
            "file": audio_path,
            "prediction": base_result["prediction"],
            "confidence": base_result["confidence"],
            "real_prob": base_result["real_prob"],
            "fake_prob": base_result["fake_prob"],
            "ood": ood_info,
            "security_decision": decision,
            "drift": drift_info,
        }
    )
    return result


def test_batch_files(model, preprocessor, file_paths, use_enhanced=True):
    results = []
    for audio_path in file_paths:
        result = test_single_audio(model, preprocessor, audio_path, use_enhanced=use_enhanced)
        if result:
            results.append(result)
    return pd.DataFrame(results) if results else None


def test_folder(model, preprocessor, folder_path, max_files=None, use_enhanced=True, save_csv_path=None):
    audio_extensions = (".wav", ".mp3", ".flac", ".m4a")
    audio_files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(audio_extensions)
    ]
    if max_files:
        audio_files = audio_files[:max_files]
    results_df = test_batch_files(model, preprocessor, audio_files, use_enhanced=use_enhanced)
    if results_df is not None and save_csv_path:
        results_df.to_csv(save_csv_path, index=False)
    return results_df


def visualize_prediction(preprocessor, audio_path, model, use_enhanced=True):
    audio = preprocessor.load_audio(audio_path)
    if audio is None:
        print("Error loading audio")
        return None, None

    mel_spec = preprocessor.extract_melspectrogram(audio)
    mfcc = preprocessor.extract_mfcc(audio)
    laplacian = preprocessor.extract_laplacian_features(mel_spec)

    features = preprocessor.create_combined_features(audio, use_enhanced=use_enhanced)
    prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)

    predicted_class = "Fake" if prediction[0][1] > 0.5 else "Real"
    confidence = prediction[0][prediction.argmax()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    librosa.display.waveshow(audio, sr=preprocessor.sample_rate, ax=axes[0, 0])
    axes[0, 0].set_title(f"Waveform - {os.path.basename(audio_path)}")

    img1 = librosa.display.specshow(
        mel_spec,
        sr=preprocessor.sample_rate,
        hop_length=preprocessor.hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[0, 1],
        cmap="viridis",
    )
    axes[0, 1].set_title("Mel Spectrogram")
    plt.colorbar(img1, ax=axes[0, 1], format="%+2.0f dB")

    img2 = librosa.display.specshow(
        mfcc,
        sr=preprocessor.sample_rate,
        hop_length=preprocessor.hop_length,
        x_axis="time",
        ax=axes[1, 0],
        cmap="coolwarm",
    )
    axes[1, 0].set_title("MFCC Features")
    plt.colorbar(img2, ax=axes[1, 0])

    img3 = axes[1, 1].imshow(laplacian, aspect="auto", origin="lower", cmap="RdBu_r")
    axes[1, 1].set_title("Laplacian Features")
    plt.colorbar(img3, ax=axes[1, 1])

    color = "red" if predicted_class == "Fake" else "green"
    fig.suptitle(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})", color=color)
    plt.tight_layout()
    plt.show()
    return predicted_class, confidence
