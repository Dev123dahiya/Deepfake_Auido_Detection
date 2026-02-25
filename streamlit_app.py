import os
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="Deepfake Audio Detector", page_icon=":studio_microphone:", layout="centered")
st.title("Deepfake Audio Detector")
st.write("Upload an audio file and run prediction through your deployed API service.")

default_api_url = os.getenv("BACKEND_API_URL", "").strip()

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input("Backend API URL", value=default_api_url, placeholder="https://your-api.onrender.com")
    use_basic = st.checkbox("Use basic features", value=False)

if not api_url:
    st.info("Set Backend API URL in sidebar. Example: `https://your-api.onrender.com`")
    st.stop()

audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a"])

if st.button("Predict", type="primary"):
    if audio_file is None:
        st.warning("Please upload an audio file.")
        st.stop()

    base_url = api_url.rstrip("/")
    endpoint = f"{base_url}/predict"
    params = {"use_basic_features": str(use_basic).lower()}

    suffix = Path(audio_file.name).suffix or ".wav"
    mime = "audio/wav" if suffix.lower() == ".wav" else "application/octet-stream"
    files = {"file": (audio_file.name, audio_file.getvalue(), mime)}

    try:
        response = requests.post(endpoint, params=params, files=files, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    try:
        result = response.json()
    except ValueError:
        st.error(f"Backend returned non-JSON response: {response.text[:500]}")
        st.stop()

    st.subheader("Result")
    st.metric("Prediction", str(result.get("prediction", "N/A")))

    confidence = result.get("confidence")
    if isinstance(confidence, (float, int)):
        st.metric("Confidence", f"{float(confidence) * 100:.2f}%")

    st.json(result)
