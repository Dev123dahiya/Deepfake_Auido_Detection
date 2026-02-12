from tensorflow.keras.models import load_model

from .modeling import AttentionLayer
from .preprocessing import AudioPreprocessor
from .security import sha256_file, verify_file_checksum


def load_trained_model(model_path):
    custom_objects = {"AttentionLayer": AttentionLayer}
    return load_model(model_path, custom_objects=custom_objects)


def create_default_preprocessor():
    return AudioPreprocessor(sample_rate=16000, duration=3, n_mels=128, hop_length=512, n_fft=2048)


def calculate_model_checksum(model_path):
    return sha256_file(model_path)


def verify_model_checksum(model_path, expected_sha256):
    return verify_file_checksum(model_path, expected_sha256)
