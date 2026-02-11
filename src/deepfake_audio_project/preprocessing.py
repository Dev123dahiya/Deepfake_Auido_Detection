import cv2
import librosa
import numpy as np


class AudioPreprocessor:
    def __init__(self, sample_rate=16000, duration=3, n_mels=128, hop_length=512, n_fft=2048):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_len = int(sample_rate * duration)

    def load_audio(self, file_path):
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            if len(audio) < self.max_len:
                audio = np.pad(audio, (0, self.max_len - len(audio)))
            else:
                audio = audio[: self.max_len]
            return audio
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            return None

    def extract_melspectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    def extract_mfcc(self, audio, n_mfcc=13):
        return librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )

    def extract_laplacian_features(self, mel_spec):
        mel_spec_normalized = cv2.normalize(mel_spec, None, 0, 255, cv2.NORM_MINMAX)
        mel_spec_uint8 = mel_spec_normalized.astype(np.uint8)
        laplacian = cv2.Laplacian(mel_spec_uint8, cv2.CV_64F)
        laplacian_normalized = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-8)
        return (laplacian_normalized * 2) - 1

    def extract_lfcc(self, audio, n_lfcc=20):
        power = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)) ** 2
        return librosa.feature.mfcc(S=librosa.power_to_db(power), n_mfcc=n_lfcc, dct_type=2)

    def extract_spectral_features(self, audio):
        return {
            "spectral_contrast": librosa.feature.spectral_contrast(
                y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            ),
            "spectral_rolloff": librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            ),
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length),
            "chroma": librosa.feature.chroma_stft(
                y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            ),
        }

    def create_combined_features(self, audio, use_enhanced=True):
        if not use_enhanced:
            mel_spec = self.extract_melspectrogram(audio)
            mfcc = self.extract_mfcc(audio)
            min_time_steps = min(mel_spec.shape[1], mfcc.shape[1])
            return np.vstack([mel_spec[:, :min_time_steps], mfcc[:, :min_time_steps]])

        mel_spec = self.extract_melspectrogram(audio)
        mfcc = self.extract_mfcc(audio)
        lfcc = self.extract_lfcc(audio)
        spectral_features = self.extract_spectral_features(audio)
        laplacian = self.extract_laplacian_features(mel_spec)

        min_time_steps = min(
            mel_spec.shape[1],
            mfcc.shape[1],
            lfcc.shape[1],
            spectral_features["spectral_contrast"].shape[1],
        )

        mel_spec = mel_spec[:, :min_time_steps]
        mfcc = mfcc[:, :min_time_steps]
        lfcc = lfcc[:, :min_time_steps]
        laplacian = laplacian[:, :min_time_steps]
        spectral_contrast = spectral_features["spectral_contrast"][:, :min_time_steps]
        spectral_rolloff = spectral_features["spectral_rolloff"][:, :min_time_steps]
        zcr = spectral_features["zero_crossing_rate"][:, :min_time_steps]
        chroma = spectral_features["chroma"][:, :min_time_steps]

        return np.vstack([mel_spec, mfcc, lfcc, laplacian, spectral_contrast, spectral_rolloff, zcr, chroma])

