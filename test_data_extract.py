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

    def extract_formants(self, label):
        y, sr = librosa.load(self.file_path, sr=self.sr)

        error_conditions = {
            "Original": y,
        }

        for condition, y_mod in error_conditions.items():
            f0, voiced_flag, voiced_probs = librosa.pyin(y_mod, fmin=50, fmax=500)

            features = {
                "Mean Pitch (F0)": np.nanmean(f0) if f0 is not None else None,
                "Zero Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y_mod)),
                "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=y_mod, sr=sr)),
                "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y_mod, sr=sr)),
                "Spectral Flatness": np.mean(librosa.feature.spectral_flatness(y=y_mod)),
                "Spectral Roll-off": np.mean(librosa.feature.spectral_rolloff(y=y_mod, sr=sr)),
                "Root Mean Square Energy": np.mean(librosa.feature.rms(y=y_mod)),
                "Skewness": skew(y_mod) if len(y_mod) > 0 else None,
            }

            mfccs = librosa.feature.mfcc(y=y_mod, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"MFCC {i+1}"] = np.mean(mfccs[i])

            # Chroma and Mel Spectrogram features with error handling
            features["Chroma STFT"] = self._safe_feature(librosa.feature.chroma_stft, y_mod)
            features["Chroma CQT"] = self._safe_feature(librosa.feature.chroma_cqt, y_mod)
            features["Mel Spectrogram"] = self._safe_feature(librosa.feature.melspectrogram, y_mod)

            features["Label"] = label
            features["Condition"] = condition

            self.data.append(features)
            print(f"Extracted features for {condition}: {features}")

        return self.data

    def _safe_feature(self, func, y_mod):
        try:
            return np.mean(func(y=y_mod, sr=self.sr))
        except Exception:
            return None

# Example usage
if __name__ == "__main__":
    file_path = "predictvoice.wav"
    label = "unknown"

    print("🔍 Extracting audio features...")
    afe = AudioFeatureExtractor(file_path)
    features = afe.extract_formants(label)
    features = pd.DataFrame(features)
    features.to_csv("audio_features.csv", index=False)
    print(features)