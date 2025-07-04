from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from predict import VoicePredict
from test_data_extract import AudioFeatureExtractor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username'].strip().lower()
        print(f"📝 Form submitted: username = {username}")
        return redirect(url_for('record_page', username=username))
    print("📄 Rendering index.html")
    return render_template('index.html')


@app.route('/record/<username>')
def record_page(username):
    print(f"🎙️ Navigated to /record for username: {username}")
    return render_template('record.html', username=username)


@app.route('/upload_audio/<username>', methods=['POST'])
def upload_audio(username):
    print(f"⬆️ Uploading audio for user: {username}")
    if 'audio_data' not in request.files:
        print("❌ audio_data not found in request.files")
        return "❌ No audio uploaded", 400

    wav_path = 'predictvoice.wav'
    try:
        audio_file = request.files['audio_data']
        audio_file.save(wav_path)
        print(f"✅ Audio saved as: {wav_path}")
    except Exception as e:
        print(f"❌ Failed to save audio: {e}")
        return f"❌ Failed to save audio: {e}", 500

    try:
        print("🔍 Extracting audio features...")
        afe = AudioFeatureExtractor(wav_path)
        features = afe.extract_formants(label=username)
        print("✅ Feature extraction complete")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return f"❌ Feature extraction failed: {e}", 500

    try:
        df = pd.DataFrame(features)
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        feature_path = os.path.join(temp_dir, f"{username}_features.csv")
        df.to_csv(feature_path, index=False)
        print(f"💾 Features saved at: {feature_path}")
    except Exception as e:
        print(f"❌ Saving features failed: {e}")
        return f"❌ Saving features failed: {e}", 500

    print("➡️ Redirecting to /predict route...")
    return redirect(url_for('predict_result', username=username))


@app.route('/predict/<username>')
def predict_result(username):
    print(f"🧠 Running prediction for: {username}")
    csv_path = f"temp/{username}_features.csv"
    

    if not os.path.exists(csv_path):
        print("❌ Feature CSV not found")
        return "❌ Feature file not found. Please re-record.", 404

    try:
        df = pd.read_csv(csv_path)
        df = df.dropna().reset_index(drop=True)
        
        print(f"📄 Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return f"❌ Failed to read feature file: {e}", 500
    
    numeric_df = df.select_dtypes(include=['number'])

    mean_features = numeric_df.mean().to_frame().T  # .T to get a single-row DataFrame

    mean_features['Label'] = username
    mean_features['Condition'] = 'mean'

    cols = list(numeric_df.columns) + ['Label', 'Condition']
    mean_df = mean_features[cols]

    sentence = "hello world"
    try:
        predictor = VoicePredict(df, sentence)
        result_df,sentence_matched = predictor.predict(username)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return f"❌ Prediction failed: {e}", 500

    if result_df is None:
        print("❌ Prediction returned None (model not found)")
        return render_template('predict_result.html', username=username, result="❌ match not found.", match=False)

    top_label = result_df['Predicted_Label'].mode()[0]
    top_conf = result_df['Confidence'].mean()
    match = (top_label == username)

    print("🔍 Prediction Results:")
    print(result_df)
    print(f"🎯 Predicted: {top_label}, Confidence: {top_conf:.2f}, Match: {match}")

    return render_template(
        'predict_result.html',
        username=username,
        predicted_label=top_label,
        confidence=round(top_conf * 100, 2),
        match=match,
        sentence_matched=sentence_matched,
    )


if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    print("🚀 Starting Flask server...")
    app.run(debug=True)
