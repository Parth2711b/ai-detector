import os
import librosa
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_DIR = "data"
MODEL_PATH = "models/voice_detector.pkl"

SAMPLE_RATE = 16000
N_MFCC = 40

def extract_features(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

X, y = [], []

print("📦 Loading dataset...")

for label_name, label_value in [("human", 0), ("ai", 1)]:
    base_path = os.path.join(DATASET_DIR, label_name)

    for lang in os.listdir(base_path):
        lang_path = os.path.join(base_path, lang)

        for file in tqdm(os.listdir(lang_path), desc=f"{label_name}-{lang}"):
            if not file.lower().endswith(".mp3"):
                continue

            file_path = os.path.join(lang_path, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label_value)
            except Exception as e:
                print(f"Skipping {file}: {e}")

X = np.array(X)
y = np.array(y)

print(f"✅ Samples loaded: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("🧠 Training...")
model.fit(X_train, y_train)

print("📊 Evaluation")
print(classification_report(y_test, model.predict(X_test)))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"💾 Model saved → {MODEL_PATH}")
