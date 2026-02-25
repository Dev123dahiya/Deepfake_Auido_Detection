import os
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.deepfake_audio_project.inference import secure_predict_single_audio
from src.deepfake_audio_project.model_io import create_default_preprocessor, load_trained_model


st.set_page_config(page_title="Deepfake Audio Detector", page_icon=":studio_microphone:", layout="wide")
st.title("Deepfake Audio Detector")
st.caption("Analyze uploaded or recorded audio with local model inference, confidence, and forensic feature plots.")


@st.cache_resource
def get_model_and_preprocessor(model_path: str):
    model = load_trained_model(model_path)
    preprocessor = create_default_preprocessor()
    return model, preprocessor


def save_temp_audio(file_name: str, audio_bytes: bytes) -> str:
    suffix = Path(file_name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def render_analysis_plots(preprocessor, audio: np.ndarray):
    mel_spec = preprocessor.extract_melspectrogram(audio)
    mfcc = preprocessor.extract_mfcc(audio)
    laplacian = preprocessor.extract_laplacian_features(mel_spec)
    spectral = preprocessor.extract_spectral_features(audio)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    librosa.display.waveshow(audio, sr=preprocessor.sample_rate, ax=axes[0, 0], color="#2563eb")
    axes[0, 0].set_title("Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    img_mel = librosa.display.specshow(
        mel_spec,
        sr=preprocessor.sample_rate,
        hop_length=preprocessor.hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Mel Spectrogram")
    fig.colorbar(img_mel, ax=axes[0, 1], format="%+2.0f dB")

    img_mfcc = librosa.display.specshow(
        mfcc,
        sr=preprocessor.sample_rate,
        hop_length=preprocessor.hop_length,
        x_axis="time",
        cmap="viridis",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("MFCC")
    fig.colorbar(img_mfcc, ax=axes[1, 0])

    img_lap = axes[1, 1].imshow(laplacian, aspect="auto", origin="lower", cmap="coolwarm")
    axes[1, 1].set_title("Laplacian Features")
    axes[1, 1].set_xlabel("Frames")
    axes[1, 1].set_ylabel("Mel bins")
    fig.colorbar(img_lap, ax=axes[1, 1])
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    rolloff = spectral["spectral_rolloff"][0]
    zcr = spectral["zero_crossing_rate"][0]
    frame_axis = np.arange(len(rolloff))
    summary_fig, summary_ax = plt.subplots(figsize=(12, 3.5))
    summary_ax.plot(frame_axis, rolloff, label="Spectral Rolloff", color="#dc2626")
    summary_ax.plot(frame_axis, zcr * np.max(rolloff), label="ZCR (scaled)", color="#059669")
    summary_ax.set_title("Temporal Forensic Signals")
    summary_ax.set_xlabel("Frame")
    summary_ax.set_ylabel("Value")
    summary_ax.legend(loc="upper right")
    summary_ax.grid(alpha=0.2)
    summary_fig.tight_layout()
    st.pyplot(summary_fig, clear_figure=True)

    return {
        "audio_seconds": float(len(audio) / preprocessor.sample_rate),
        "mean_abs_amplitude": float(np.mean(np.abs(audio))),
        "mean_spectral_rolloff": float(np.mean(rolloff)),
        "mean_zero_crossing_rate": float(np.mean(zcr)),
    }


with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model path (.h5)", value="outputs/deepfake_detector_enhanced_final.h5")
    use_basic = st.checkbox("Use basic features only", value=False)
    st.header("Risk thresholds")
    ood_confidence_threshold = st.slider("OOD confidence threshold", 0.0, 1.0, 0.60, 0.01)
    ood_entropy_threshold = st.slider("OOD entropy threshold", 0.0, 1.0, 0.68, 0.01)
    low_risk_threshold = st.slider("Low risk threshold", 0.0, 1.0, 0.30, 0.01)
    high_risk_threshold = st.slider("High risk threshold", 0.0, 1.0, 0.75, 0.01)


input_mode = st.radio("Choose input source", ["Upload audio file", "Record live audio"], horizontal=True)

audio_bytes = None
audio_name = "recorded_audio.wav"

if input_mode == "Upload audio file":
    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        audio_name = uploaded_file.name
else:
    if hasattr(st, "audio_input"):
        recorded_file = st.audio_input("Record audio")
        if recorded_file is not None:
            audio_bytes = recorded_file.getvalue()
            audio_name = "live_recording.wav"
    else:
        st.warning("This Streamlit version does not support live recording. Use file upload.")

if audio_bytes:
    st.audio(audio_bytes)

if st.button("Analyze Audio", type="primary"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if audio_bytes is None:
        st.warning("Provide an audio file from upload or live recording.")
        st.stop()

    temp_audio_path = save_temp_audio(audio_name, audio_bytes)
    try:
        model, preprocessor = get_model_and_preprocessor(model_path)
        result = secure_predict_single_audio(
            model=model,
            preprocessor=preprocessor,
            audio_path=temp_audio_path,
            use_enhanced=not use_basic,
            ood_confidence_threshold=ood_confidence_threshold,
            ood_entropy_threshold=ood_entropy_threshold,
            low_risk_threshold=low_risk_threshold,
            high_risk_threshold=high_risk_threshold,
            audit_log_path="outputs/security_audit.jsonl",
        )

        if not result:
            st.error("Prediction failed. Check audio format and model compatibility.")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction", result["prediction"])
        col2.metric("Confidence", f'{float(result["confidence"]) * 100:.2f}%')
        col3.metric("Fake Probability", f'{float(result["fake_prob"]) * 100:.2f}%')
        col4.metric("Real Probability", f'{float(result["real_prob"]) * 100:.2f}%')

        probs_df = pd.DataFrame(
            {"Class": ["Real", "Fake"], "Probability": [float(result["real_prob"]), float(result["fake_prob"])]}
        ).set_index("Class")
        st.subheader("Probability Distribution")
        st.bar_chart(probs_df)

        security_col1, security_col2, security_col3 = st.columns(3)
        security_col1.metric("Security Decision", result["security_decision"]["action"])
        security_col2.metric("Risk Level", result["security_decision"]["risk_level"])
        security_col3.metric("OOD Flag", "Yes" if result["ood"]["is_ood"] else "No")

        st.subheader("Forensic Analysis Graphs")
        audio_arr = preprocessor.load_audio(temp_audio_path)
        if audio_arr is None:
            st.error("Could not decode audio for analysis plots.")
        else:
            summary_stats = render_analysis_plots(preprocessor, audio_arr)
            st.subheader("Analysis Summary")
            st.json(summary_stats)

        with st.expander("Full Prediction JSON"):
            st.json(result)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
