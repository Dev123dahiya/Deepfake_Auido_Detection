# Refactored Project Layout

This folder now contains:

- Original notebook split by cells (`notebook_cells/`)
- Meaningfully named copies of those cell files in each section folder
- A merged, modular Python package in `src/deepfake_audio_project/`
- A command-line runner: `main.py`

## Run examples

```powershell
# Train
python main.py train --dataset "C:\path\to\dataset" --output outputs --max-files 500 --epochs 30

# Predict one file
python main.py predict --model outputs\deepfake_detector_enhanced_final.h5 --audio "C:\path\to\file.wav"
```
