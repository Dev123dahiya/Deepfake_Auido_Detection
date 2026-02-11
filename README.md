# Dee-fake Audio Detector

Deepfake Audio Detection project developed by **Khushi Bishnoi** to classify audio as **Real** or **Fake** using deep learning and advanced audio forensics features.

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
  - Confusion matrix and classification report

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

## Notes

- This project supports both **basic** and **enhanced** feature modes.
- Use `--basic` in CLI commands to run with basic features only.
- Trained model files (`.h5`) are generated in the `outputs/` directory.
