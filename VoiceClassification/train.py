import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
csv_path = 'data/audio_features_with_errors.csv'
df = pd.read_csv(csv_path)


class VoiceTrainer:
    print("🔧 Voice Trainer Initialized")

    def __init__(self, dataframe):
        self.df = dataframe

    def filtered_df(self, label):
        # Keep 'label' samples
        target_df = self.df[self.df['Label'] == label]

        # Mark others as 'unknown'
        unknown_df = self.df[self.df['Label'] != label].copy()
        unknown_df['Label'] = 'unknown'

        # Sample equal number of 'unknown' entries
        unknown_df_sampled = unknown_df.sample(n=len(target_df), random_state=42)

        # Combine and shuffle
        balanced_df = pd.concat([target_df, unknown_df_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        return balanced_df

    def train_model(self, df, label):
        print(df.head(20))

        X = df.drop(columns=['Label', 'Condition'])
        y = df['Label']

        # 🔄 Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # model = RandomForestClassifier(n_estimators=100, random_state=42)
        # model = LinearRegression()  # Uncomment for regression
        # model = GaussianNB()  # Uncomment for Naive Bayes # better
        # model = SVC(probability=True, random_state=42)  # Uncomment for
        model = DecisionTreeClassifier(random_state=42)  # Uncomment for Decision Tree # best
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=model.classes_))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        # ROC Curve
        if label in model.classes_:
            label_index = list(model.classes_).index(label)
            y_true_binary = (y_test == label).astype(int)
            y_score = y_proba[:, label_index]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)

        # Save model and scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{label}_voice_model.pkl")
        joblib.dump(scaler, f"models/{label}_scaler.pkl")
        print(f"\n✅ Model and scaler saved to models/{label}_voice_model.pkl & scaler.pkl")

        return model, scaler

    def run(self, target_Label):
        print(f"🔧 Training model for Label: '{target_Label}' (others will be 'unknown')\n")
        filtered_df = self.filtered_df(target_Label)
        # Drop rows with any NaNs
        filtered_df = filtered_df.dropna().reset_index(drop=True)


        # 🎯 Label distribution
        label_counts = filtered_df['Label'].value_counts()
        print("\n🎯 Filtered Label Distribution:\n", label_counts)

        # 📏 Cosine Similarity Score
        target_features = filtered_df[filtered_df['Label'] == target_Label].drop(columns=['Label', 'Condition'])
        unknown_features = filtered_df[filtered_df['Label'] == 'unknown'].drop(columns=['Label', 'Condition'])

        scaler = StandardScaler()
        target_scaled = scaler.fit_transform(target_features)
        unknown_scaled = scaler.transform(unknown_features)

        target_mean = np.mean(target_scaled, axis=0).reshape(1, -1)
        unknown_mean = np.mean(unknown_scaled, axis=0).reshape(1, -1)


        similarity_score = cosine_similarity(target_mean, unknown_mean)[0][0]
        print(f"\n📏 Cosine Similarity (avg '{target_Label}' vs 'unknown'): {similarity_score:.4f}")

        self.train_model(filtered_df, target_Label)


# 🔽 Run script
if __name__ == "__main__":
    Label_to_train = "avra"  # Replace with the actual label you want to train
    trainer = VoiceTrainer(df)
    trainer.run(Label_to_train)
