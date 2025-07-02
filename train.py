import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

csv_path = 'data/audio_features_with_errors.csv'
df = pd.read_csv(csv_path)
print("Original Data:")
print(df.head(5))


class VoiceTrainer:
    def __init__(self, dataframe):
        self.df = dataframe

    def filtered_df(self, label):
        self.df['Label'] = self.df['Label'].apply(lambda x: x if x == label else 'unknown')
        return self.df
    
    def train_model(self, df,label):
        X = df.drop(columns=['Label', 'Condition'])
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{label}_voice_model.pkl")
        print(f"\nModel saved to models/{label}_voice_model.pkl")

        return model

    def run(self, target_Label):
        print(f"🔧 Training model for Label: '{target_Label}' (others will be 'unknown')\n")
        filtered_df = self.filtered_df(target_Label)
        self.train_model(filtered_df, target_Label)
        

if __name__ == "__main__":
    Label_to_train = "deb"
    trainer = VoiceTrainer(df)
    trainer.run(Label_to_train)
