import pandas as pd
import speech_recognition as sr
import joblib
import os
from sklearn.preprocessing import StandardScaler

class VoicePredict:
    def __init__(self, dataframe, sentence):
        self.df = dataframe
        self.sentence = sentence.lower()
        self.file_path = "predictvoice.wav"

    def convert_audio_to_text(self):
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.file_path) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.RequestError):
                return None    

    def predict(self, label):
        model_path = f"models/{label}_voice_model.pkl"
        scaler_path = f"models/{label}_scaler.pkl"

        print(f"🔍 Loading model from {model_path} and scaler from {scaler_path} for label '{label}'...")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"❌ Model or scaler for label '{label}' not found.")
            return None, False

        # Load trained model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Prepare features
        features_df = self.df.drop(columns=[col for col in ['Label', 'Condition'] if col in self.df.columns])
        features_scaled = scaler.transform(features_df)

        # Predict
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        confidence_scores = probabilities.max(axis=1)

        # Append prediction results
        self.df['Predicted_Label'] = predictions
        self.df['Confidence'] = confidence_scores

        print("\n🔍 Prediction Results with Confidence:")
        print(self.df[['Predicted_Label', 'Confidence']])

        # Get label with highest average confidence
        summary = self.df.groupby('Predicted_Label')['Confidence'].mean()
        top_label = summary.idxmax()
        top_conf = summary.max()

        transcribed_text = self.convert_audio_to_text()
        sentence_match = False

        if transcribed_text:
            print(f"🗣️ You said: \"{transcribed_text}\"")
            if transcribed_text.strip() == self.sentence.strip():
                sentence_match = True
                print(f"\n🎉 Welcome, {top_label.capitalize()}! (Avg confidence: {top_conf:.2f})")
            else:
                print(f"\n🛑 Text mismatch. Expected: \"{self.sentence.strip()}\"")
        else:
            print("\n❗ Could not transcribe voice to text.")

        return self.df, sentence_match



# Usage Example
if __name__ == "__main__":
    label_to_predict = "debjyoti"  # Replace with the actual label you want to predict
    csv_path = f'temp/{label_to_predict}_features.csv'
    sentence = "hello world"  # Expected sentence to match
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna().reset_index(drop=True)
        predictor = VoicePredict(df, sentence)
        predicted_df, sentence_matched = predictor.predict(label_to_predict)
        if sentence_matched:
            print("✅ Sentence matched correctly.")
        else:
            print("❌ Sentence did not match.")
    else:
        print(f"❌ Feature CSV not found at: {csv_path}")
