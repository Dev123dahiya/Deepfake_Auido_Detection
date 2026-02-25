import os
import tempfile
from pathlib import Path

import streamlit as st

from src.deepfake_audio_project.inference import test_single_audio
from src.deepfake_audio_project.model_io import create_default_preprocessor, load_trained_model


st.set_page_config(page_title="Deepfake Audio Detector", page_icon=":studio_microphone:", layout="centered")
st.title("Deepfake Audio Detector")
st.write("Upload a trained `.h5` model (optional) and an audio file to classify it as Real or Fake.")


@st.cache_resource
def _load_model_and_preprocessor(model_path: str):
    model = load_trained_model(model_path)
    preprocessor = create_default_preprocessor()
    return model, preprocessor


def _save_uploaded_file(uploaded_file, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


default_model_path = os.getenv("MODEL_PATH", "outputs/deepfake_detector_enhanced_final.h5")
model_path = None

with st.sidebar:
    st.header("Model Settings")
    st.caption(f"Default MODEL_PATH: `{default_model_path}`")
    uploaded_model = st.file_uploader("Upload model (.h5)", type=["h5"])

if uploaded_model is not None:
    model_path = _save_uploaded_file(uploaded_model, ".h5")
elif Path(default_model_path).exists():
    model_path = default_model_path

if not model_path:
    st.error("No model found. Upload a `.h5` model from the sidebar or set `MODEL_PATH` to an existing file.")
    st.stop()

try:
    model, preprocessor = _load_model_and_preprocessor(model_path)
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a"])
use_basic = st.checkbox("Use basic features (disable enhanced features)", value=False)

if st.button("Predict", type="primary"):
    if audio_file is None:
        st.warning("Please upload an audio file.")
        st.stop()

    temp_audio_path = _save_uploaded_file(audio_file, Path(audio_file.name).suffix or ".wav")
    try:
        result = test_single_audio(model, preprocessor, temp_audio_path, use_enhanced=not use_basic)
    finally:
        try:
            os.remove(temp_audio_path)
        except OSError:
            pass

    if not result:
        st.error("Prediction failed. Please try another audio file.")
    else:
        st.subheader("Result")
        st.metric("Prediction", result["prediction"])
        st.metric("Confidence", f'{float(result["confidence"]) * 100:.2f}%')
        st.write(
            {
                "real_prob": float(result["real_prob"]),
                "fake_prob": float(result["fake_prob"]),
                "file": result["file"],
            }
        )
