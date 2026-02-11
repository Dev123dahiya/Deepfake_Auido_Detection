from tensorflow.keras.models import load_model

from .modeling import AttentionLayer
from .preprocessing import AudioPreprocessor


def load_trained_model(model_path):
    custom_objects = {"AttentionLayer": AttentionLayer}
    return load_model(model_path, custom_objects=custom_objects)


def create_default_preprocessor():
    return AudioPreprocessor(sample_rate=16000, duration=3, n_mels=128, hop_length=512, n_fft=2048)

