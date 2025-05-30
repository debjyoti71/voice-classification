import os
import pyaudio
import wave
import pandas as pd
import speech_recognition as sr
from test_data_extract import AudioFeatureExtractor
from predict import VoicePredict

class VoiceAuthenticator:
    def __init__(self, sentence="open the door", file_path="predictvoice.wav", duration=5, rate=22050, channels=1):
        self.sentence = sentence.lower()
        self.file_path = file_path
        self.duration = duration
        self.rate = rate
        self.channels = channels
        self.name = "unknown"

    def record_audio(self):
        print("🎙️ Recording audio...")
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=self.channels,
                            rate=self.rate, input=True, frames_per_buffer=1024)

        frames = [stream.read(1024) for _ in range(int(self.rate / 1024 * self.duration))]

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(self.file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))

        print(f"✅ Recording saved to {self.file_path}")

    def convert_audio_to_text(self):
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.file_path) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.RequestError):
                return None

    def process_new_input(self):
        self.record_audio()
        transcribed_text = self.convert_audio_to_text()
        if transcribed_text:
            print(f"🗣️ You said: {transcribed_text}")
            if transcribed_text.strip() == self.sentence:
                print("✅ Sentence matched. Proceeding to feature extraction...")
                label = "unknown"
                print("Extracting features...")
                afe = AudioFeatureExtractor(self.file_path)
                feature = afe.extract_formants(label)

                # Save extracted features
                df = pd.DataFrame(feature)
                df.to_csv("audio_features_for_predict.csv", index=False)
                print("📁 Features saved to 'audio_features_for_predict.csv'")
                predictor = VoicePredict(df)
                label_to_predict = "debjyoti"
                predicted_df = predictor.predict(label_to_predict)
            else:
                print("❌ Sentence mismatch. Please try again.")
                return None
        else:
            print("⚠️ Could not understand the audio. Please try again.")
            return None


# Example usage: 
if __name__ == "__main__":
    va = VoiceAuthenticator()
    va.process_new_input()
