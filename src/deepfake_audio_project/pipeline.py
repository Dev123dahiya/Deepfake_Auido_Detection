from pathlib import Path

import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from .config import TrainingConfig
from .dataset import load_dataset
from .evaluation import evaluate_model
from .modeling import create_cnn_attention_model
from .preprocessing import AudioPreprocessor
from .training import plot_training_history, train_model


def main_training_pipeline(config: TrainingConfig):
    print("=== DEEPFAKE AUDIO DETECTION TRAINING ===")
    print(f"Using enhanced features: {config.use_enhanced_features}")

    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    preprocessor = AudioPreprocessor()
    dataset_path = config.dataset_path
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    X, y = load_dataset(
        dataset_path,
        preprocessor,
        max_files_per_class=config.max_files_per_class,
        use_enhanced_features=config.use_enhanced_features,
    )
    if len(X) == 0:
        raise RuntimeError("No data loaded. Check dataset structure.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=2)
    class_counts = Counter(y.tolist())
    if any(count < 3 for count in class_counts.values()):
        raise ValueError(
            f"Too few readable samples per class after loading: {dict(class_counts)}. "
            "Increase --max-files or fix corrupted audio files."
        )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, random_state=config.random_seed, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=config.random_seed,
        stratify=y_temp.argmax(axis=1),
    )

    model = create_cnn_attention_model(X_train.shape[1:], num_classes=2)
    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
    )
    plot_training_history(history)
    evaluate_model(model, X_test, y_test, class_names=("Real", "Fake"))

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_suffix = "enhanced" if config.use_enhanced_features else "basic"
    final_model_path = output_path / f"deepfake_detector_{model_suffix}_final.h5"
    model.save(str(final_model_path))
    print(f"Model saved to: {final_model_path}")
    return model, preprocessor, label_encoder, history
