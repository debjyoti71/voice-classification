import os
import librosa
import numpy as np
from scipy.stats import skew


class AudioFeatureExtractor:
    def __init__(self, file_path, label):
        self.file_path = file_path
        self.label = label
        self.sr = 22050
        self.y, self.sr = librosa.load(self.file_path, sr=self.sr)
        self.data = []

    def add_noise(self, y, noise_level=0.02):
        noise = np.random.normal(0, noise_level, y.shape)
        return y + noise

    def add_echo(self, y, delay=0.2, decay=0.5):
        delay_samples = int(self.sr * delay)
        echo_signal = np.zeros_like(y)
        echo_signal[delay_samples:] = y[:-delay_samples] * decay
        return y + echo_signal

    def increase_loudness(self, y, gain_db=10):
        gain = 10 ** (gain_db / 20)
        return y * gain

    def high_pass_filter(self, y, cutoff=3000):
        return librosa.effects.preemphasis(y, coef=0.97)

    def extract_features(self, y_mod, condition):
        try:
            f0, _, _ = librosa.pyin(y_mod, fmin=50, fmax=500)
        except:
            f0 = None

        features = {
            "Mean Pitch (F0)": np.nanmean(f0) if f0 is not None else None,
            "Zero Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y_mod)),
            "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=y_mod, sr=self.sr)),
            "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y_mod, sr=self.sr)),
            "Spectral Flatness": np.mean(librosa.feature.spectral_flatness(y=y_mod)),
            "Spectral Roll-off": np.mean(librosa.feature.spectral_rolloff(y=y_mod, sr=self.sr)),
            "Root Mean Square Energy": np.mean(librosa.feature.rms(y=y_mod)),
            "Skewness": skew(y_mod) if len(y_mod) > 0 else None,
        }

        mfccs = librosa.feature.mfcc(y=y_mod, sr=self.sr, n_mfcc=13)
        for i in range(13):
            features[f"MFCC {i + 1}"] = np.mean(mfccs[i])

        # Additional chroma and mel features with exception handling
        try:
            features["Chroma STFT"] = np.mean(librosa.feature.chroma_stft(y=y_mod, sr=self.sr))
        except:
            features["Chroma STFT"] = None

        try:
            features["Chroma CQT"] = np.mean(librosa.feature.chroma_cqt(y=y_mod, sr=self.sr))
        except:
            features["Chroma CQT"] = None

        try:
            features["Mel Spectrogram"] = np.mean(librosa.feature.melspectrogram(y=y_mod, sr=self.sr))
        except:
            features["Mel Spectrogram"] = None

        features["Label"] = self.label
        features["Condition"] = condition

        print(f"Extracted features for {condition}: {features}")
        return features

    def process_all_conditions(self):
        error_conditions = {
            "Original": self.y,
            "Noise": self.add_noise(self.y),
            "Echo": self.add_echo(self.y),
            "High dB": self.increase_loudness(self.y),
            "High Frequency": self.high_pass_filter(self.y),
        }

        for condition, y_mod in error_conditions.items():
            features = self.extract_features(y_mod, condition)
            self.data.append(features)

        return self.data


# Example usage:
# extractor = AudioFeatureExtractor("sample.wav", "Speaker1")
# feature_data = extractor.process_all_conditions()
