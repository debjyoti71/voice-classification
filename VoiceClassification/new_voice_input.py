import os
import wave
import pyaudio
import pandas as pd
import speech_recognition as sr
from VoiceClassification.train_data_extract import AudioFeatureExtractor

VOICE_DATA_DIR = "voice_samples"
SPLIT_SENTENCES = [
    "the quick brown", "fox jumps over", "the lazy dog",
    "and the quick brown", "fox jumps high", "over the lazy dog", "in the morning"
]

class VoiceSampleCollector:
    print("🔊 Voice Sample Collector Initialized")
    def __init__(self, name):
        self.name = name.strip().lower()
        self.user_dir = os.path.join(VOICE_DATA_DIR, self.name)
        self.recognizer = sr.Recognizer()
        self.all_features = []

    def record_audio(self, output_file, duration=5, rate=22050, channels=1):
        print("🎙️ Recording audio...")
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=1024)
        frames = [stream.read(1024) for _ in range(int(rate / 1024 * duration))]
        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        print(f"✅ Recording saved to {output_file}")
        return output_file

    def convert_audio_to_text(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.RequestError):
                return None

    def confirm_overwrite(self):
        if os.path.exists(self.user_dir) and os.listdir(self.user_dir):
            confirm = input(f"Samples already exist for '{self.name}'. Overwrite? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Operation cancelled.")
                return False
        os.makedirs(self.user_dir, exist_ok=True)
        return True

    def collect_samples(self):
        i = 0
        while i < len(SPLIT_SENTENCES):
            part = SPLIT_SENTENCES[i]
            print(f"\nPart {i + 1}: {part}")
            file_path = os.path.join(self.user_dir, f"sample{i + 1}.wav")
            self.record_audio(file_path)

            transcribed_text = self.convert_audio_to_text(file_path)
            if transcribed_text and transcribed_text.strip() == part.strip():
                print("🟢 Match confirmed.")
                i += 1
            else:
                print("🔴 Mismatch in sample. Please try again.")

    def extract_features(self):
        for file in os.listdir(self.user_dir):
            file_path = os.path.join(self.user_dir, file)
            extractor = AudioFeatureExtractor(file_path, self.name)
            features = extractor.process_all_conditions()
            self.all_features.extend(features)

    def save_to_csv(self, csv_file="data/audio_features_with_errors.csv"):
        new_df = pd.DataFrame(self.all_features)

        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            new_df = pd.concat([existing_df, new_df], ignore_index=True)

        new_df.to_csv(csv_file, index=False)
        print(f"📁 Features successfully saved to '{csv_file}'")

    def run(self):
        if not self.confirm_overwrite():
            return

        self.collect_samples()
        self.extract_features()
        self.save_to_csv()
        print("🎉 All voice samples collected and verified.")


if __name__ == "__main__":
    name_input = input("Enter name: ")
    collector = VoiceSampleCollector(name_input)
    collector.run()
