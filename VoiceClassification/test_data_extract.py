import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew

class AudioFeatureExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sr = 22050
        self.data = []

        # Load audio once
        self.y, _ = librosa.load(self.file_path, sr=self.sr)

    # Error Condition Methods
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

    def _safe_feature(self, func, y_mod):
        try:
            return np.mean(func(y=y_mod, sr=self.sr))
        except Exception:
            return None

    def extract_formants(self, label):
        # Apply error transformations
        error_conditions = {
            "Original": self.y,
            "Noise": self.add_noise(self.y),
            "Echo": self.add_echo(self.y),
            "High dB": self.increase_loudness(self.y),
            "High Frequency": self.high_pass_filter(self.y),
        }

        for condition, y_mod in error_conditions.items():
            f0, voiced_flag, voiced_probs = librosa.pyin(y_mod, fmin=50, fmax=500)

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
                features[f"MFCC {i+1}"] = np.mean(mfccs[i])

            features["Chroma STFT"] = self._safe_feature(librosa.feature.chroma_stft, y_mod)
            features["Chroma CQT"] = self._safe_feature(librosa.feature.chroma_cqt, y_mod)
            features["Mel Spectrogram"] = self._safe_feature(librosa.feature.melspectrogram, y_mod)

            features["Label"] = label
            features["Condition"] = condition

            self.data.append(features)
            print(f"✅ Extracted features for condition: {condition}")    

        return self.data


# Example usage
if __name__ == "__main__":
    file_path = "predictvoice.wav"
    label = "unknown"

    print("🔍 Extracting audio features with all error conditions...")
    afe = AudioFeatureExtractor(file_path)
    features = afe.extract_formants(label)
    df = pd.DataFrame(features)
    df.to_csv("data/audio_features_for_predict.csv", index=False)
    print("📁 Features saved to audio_features_for_predict.csv")
    print(df)
