import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.deepfake_audio_project.inference import secure_predict_single_audio, test_single_audio
from src.deepfake_audio_project.model_io import (
    calculate_model_checksum,
    create_default_preprocessor,
    load_trained_model,
    verify_model_checksum,
)
from src.deepfake_audio_project.security import DriftMonitor


MODEL_PATH = os.getenv("MODEL_PATH", "outputs/deepfake_detector_enhanced_final.h5")
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "outputs/security_audit.jsonl")
MODEL_URL = os.getenv("MODEL_URL")
ALLOW_MISSING_MODEL = os.getenv("ALLOW_MISSING_MODEL", "false").lower() == "true"

app = FastAPI(title="Deepfake Audio Detector API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
PREPROCESSOR = None
DRIFT_MONITOR = DriftMonitor(window_size=200, alert_std=3.0, min_history=30)


def _ensure_model_loaded():
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Set MODEL_PATH env var to a valid trained .h5 file and restart server."
            ),
        )


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload.file.read())
        return temp_file.name


def _download_model_if_needed(model_path: Path):
    if model_path.exists() or not MODEL_URL:
        return
    import requests

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading model from MODEL_URL -> {model_path}")
    with requests.get(MODEL_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(model_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
    print("[INFO] Model download complete.")


@app.on_event("startup")
def startup_event():
    global MODEL, PREPROCESSOR
    model_path = Path(MODEL_PATH)
    _download_model_if_needed(model_path)
    if not model_path.exists():
        if ALLOW_MISSING_MODEL:
            print(f"[WARN] MODEL_PATH not found: {model_path}. Starting without model.")
            return
        raise RuntimeError(
            f"MODEL_PATH does not exist: {model_path}. "
            "Set MODEL_PATH to a trained .h5 model file before starting the API, "
            "or set MODEL_URL to auto-download it."
        )
    MODEL = load_trained_model(str(model_path))
    PREPROCESSOR = create_default_preprocessor()
    print(f"[INFO] Model loaded from: {model_path}")


@app.get("/")
def root():
    return {"service": "Deepfake Audio Detector API", "status": "running"}


@app.get("/health")
def health():
    model_exists = Path(MODEL_PATH).exists()
    return {
        "ok": MODEL is not None and PREPROCESSOR is not None,
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "model_exists_on_disk": model_exists,
    }


@app.get("/checksum")
def checksum():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {MODEL_PATH}")
    return {"model_path": MODEL_PATH, "sha256": calculate_model_checksum(str(model_path))}


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    use_basic_features: bool = Query(False, description="Use basic features instead of enhanced"),
):
    _ensure_model_loaded()
    temp_path = _save_upload_to_temp(file)
    try:
        result = test_single_audio(MODEL, PREPROCESSOR, temp_path, use_enhanced=not use_basic_features)
        if not result:
            raise HTTPException(status_code=400, detail="Failed to process uploaded audio.")
        return result
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.post("/secure-predict")
def secure_predict(
    file: UploadFile = File(...),
    use_basic_features: bool = Query(False),
    expected_sha256: Optional[str] = Query(None),
    ood_confidence_threshold: float = Query(0.60),
    ood_entropy_threshold: float = Query(0.68),
    low_risk_threshold: float = Query(0.30),
    high_risk_threshold: float = Query(0.75),
):
    _ensure_model_loaded()
    if expected_sha256:
        ok, actual = verify_model_checksum(MODEL_PATH, expected_sha256)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"Model checksum mismatch. expected={expected_sha256} actual={actual}",
            )

    temp_path = _save_upload_to_temp(file)
    try:
        result = secure_predict_single_audio(
            MODEL,
            PREPROCESSOR,
            temp_path,
            use_enhanced=not use_basic_features,
            ood_confidence_threshold=ood_confidence_threshold,
            ood_entropy_threshold=ood_entropy_threshold,
            low_risk_threshold=low_risk_threshold,
            high_risk_threshold=high_risk_threshold,
            audit_log_path=AUDIT_LOG_PATH,
            drift_monitor=DRIFT_MONITOR,
        )
        if not result:
            raise HTTPException(status_code=400, detail="Failed to process uploaded audio.")
        return result
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
