import pandas as pd
import joblib
import os

class VoicePredict:
    def __init__(self, dataframe):
        self.df = dataframe

    def predict(self, label):
        model_path = f"models/{label}_voice_model.pkl"
        print(f"🔍 Loading model from {model_path} for label '{label}'...")
        if not os.path.exists(model_path):
            print(f"❌ Model for label '{label}' not found at {model_path}")
            return

        # Load the trained model
        model = joblib.load(model_path)

        # Drop non-feature columns if present
        features_df = self.df.drop(columns=[col for col in ['Label', 'Condition'] if col in self.df.columns])

        # Make predictions with probabilities
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)

        # Get the confidence (max probability) for each prediction
        confidence_scores = probabilities.max(axis=1)

        # Append results to the DataFrame
        self.df['Predicted_Label'] = predictions
        self.df['Confidence'] = confidence_scores

        print("\n🔍 Prediction Results with Confidence:")
        print(self.df[['Predicted_Label', 'Confidence']])
        return self.df


# Usage Example
if __name__ == "__main__":
    csv_path = 'audio_features_for_predict.csv'
    df = pd.read_csv(csv_path)
    
    predictor = VoicePredict(df)
    label_to_predict = "debjyoti"
    predicted_df = predictor.predict(label_to_predict)
