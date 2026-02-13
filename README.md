# Dee-fake Audio Detector

Deepfake Audio Detection project developed by **Khushi Bishnoi** to classify audio as **Real** or **Fake** using deep learning, advanced audio forensics features, and optional LLM-assisted research analysis.

## Work Done

I designed and implemented an end-to-end pipeline that includes:

- Audio preprocessing and feature engineering for spoof detection.
- A hybrid deep model: **CNN + BiLSTM + Attention**.
- Training, evaluation, and single-file inference pipeline.
- Optional integration hooks for external forensics APIs.
- Refactoring from notebook-style code into a clean modular codebase.
- GitHub-ready project structure with deployable files.

## Techniques Used

- **Core audio features**: Mel Spectrogram, MFCC
- **Enhanced forensic features**: Laplacian features on spectrograms, LFCC, spectral contrast, spectral rolloff, zero-crossing rate, chroma
- **Modeling**:
  - 2D CNN blocks for spatial feature extraction
  - Bidirectional LSTM for temporal context
  - Custom Attention layer for important-frame focus
- **Training strategy**:
  - Adam optimizer
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint
- **Evaluation**:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
  - PR-AUC, Balanced Accuracy, MCC, Brier Score, Log Loss, ECE
  - Confidence intervals (Wilson + bootstrap)
  - Threshold optimization for best F1
  - Confusion matrix and classification report
- **LLM-assisted layer (optional)**:
  - Generates research-style critique and improvement suggestions from computed metrics
  - Useful for paper writing and discussion section drafting

## Project Structure

```text
important_files/
  main.py
  README.md
  requirements.txt
  scripts/
    download_asvspoof2021.ps1
  src/
    deepfake_audio_project/
      config.py
      dataset.py
      preprocessing.py
      modeling.py
      training.py
      evaluation.py
      inference.py
      forensics_api.py
      model_io.py
      pipeline.py
      reporting.py
```

## Setup

```powershell
pip install -r requirements.txt
```

## Dataset Format

Expected dataset structure:

```text
<dataset_path>/
  real/
    *.wav / *.mp3 / *.flac / *.m4a
  fake/
    *.wav / *.mp3 / *.flac / *.m4a
```

## Run

### Train

```powershell
python main.py train --dataset "C:\path\to\dataset" --output outputs --max-files 500 --epochs 30
```

### Predict on one audio file

```powershell
python main.py predict --model outputs\deepfake_detector_enhanced_final.h5 --audio "C:\path\to\file.wav"
```

### Advanced Evaluation (journal-style metrics)

```powershell
python main.py evaluate --model outputs\deepfake_detector_enhanced_final.h5 --dataset "C:\path\to\dataset" --output outputs
```

### Robustness Evaluation (cybersecurity)

```powershell
python main.py evaluate --model outputs\deepfake_detector_enhanced_final.h5 --dataset "C:\path\to\dataset" --output outputs --robustness-check --max-robustness-samples 200
```

### Optional LLM Insights

Set your API key first:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Then run:

```powershell
python main.py evaluate --model outputs\deepfake_detector_enhanced_final.h5 --dataset "C:\path\to\dataset" --output outputs --llm-insights
```

### Security-Aware Prediction (OOD + risk policy + audit log)

```powershell
python main.py secure-predict --model outputs\deepfake_detector_enhanced_final.h5 --audio "C:\path\to\file.wav" --audit-log outputs\security_audit.jsonl
```

### Model Checksum (integrity)

```powershell
python main.py checksum --model outputs\deepfake_detector_enhanced_final.h5
```

## Live Server Deployment

### 1. Run API locally

```powershell
pip install -r requirements.txt
$env:MODEL_PATH="outputs\deepfake_detector_enhanced_final.h5"
uvicorn serve_api:app --host 0.0.0.0 --port 8000
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

### 2. API Endpoints

- `GET /health`
- `GET /checksum`
- `POST /predict` (multipart file upload)
- `POST /secure-predict` (multipart file upload + security thresholds)

### 3. Deploy on Render (recommended)

Repository already includes deployment files:

- `Dockerfile`
- `render.yaml`
- `Procfile`
- `runtime.txt`

Steps:

1. Push latest code to GitHub.
2. In Render, create a new Web Service from your repo.
3. Render auto-detects Docker setup.
4. Set environment variable:
   - `MODEL_PATH=outputs/deepfake_detector_enhanced_final.h5`
   - Optional (recommended for cloud): `MODEL_URL=<direct_download_link_to_your_h5_model>`
5. Deploy.

Important:
- You must ensure trained model file exists at `MODEL_PATH` inside the deployed app.
- If model file is missing but `MODEL_URL` is set, API will auto-download model at startup.
- If model file is not present, API startup will fail by design.

## Notes

- This project supports both **basic** and **enhanced** feature modes.
- Use `--basic` in CLI commands to run with basic features only.
- Trained model files (`.h5`) are generated in the `outputs/` directory.
- LLM does not compute base metrics directly; it analyzes computed results and proposes research improvements.
- Security commands add OOD detection, risk-tiered decisions, checksum verification support, drift monitoring output, and audit logging.
