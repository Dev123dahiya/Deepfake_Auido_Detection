import os
import random
import numpy as np


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


def load_dataset(dataset_path, preprocessor, max_files_per_class=None, use_enhanced_features=True):
    features = []
    labels = []

    for class_name in ["real", "fake"]:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist")
            continue

        audio_files = [f for f in os.listdir(class_path) if f.lower().endswith(AUDIO_EXTENSIONS)]
        random.Random(42).shuffle(audio_files)
        if max_files_per_class:
            audio_files = audio_files[:max_files_per_class]

        print(f"Loading {len(audio_files)} files from {class_name} class...")
        for idx, filename in enumerate(audio_files, start=1):
            file_path = os.path.join(class_path, filename)
            audio = preprocessor.load_audio(file_path)
            if audio is not None:
                feature = preprocessor.create_combined_features(audio, use_enhanced=use_enhanced_features)
                features.append(feature)
                labels.append(class_name)
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(audio_files)} files in {class_name}")

    print(f"Total files processed: {len(features)}")
    return np.array(features), np.array(labels)
