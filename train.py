import os
import io
import librosa
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
DATASET_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "voice_detector.pkl")

def extract_features(path):
    """
    Extracts 53 features:
    - 40 MFCCs (Timbre/Tone)
    - 7 Spectral Contrast (Texture/Sharpness)
    - 6 Tonnetz (Harmonic Relations)
    """
    try:
        # Load audio at 16kHz
        y, sr = librosa.load(path, sr=16000)
        
        # 1. MFCC (40 features)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # 2. Spectral Contrast (7 features) - Great for catching AI 'flatness'
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        
        # 3. Tonnetz (6 features) - Checks harmonic patterns
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        
        # Total = 53 features
        return np.hstack([mfcc, contrast, tonnetz])
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

X, y = [], []

print("🚀 Starting Advanced Feature Extraction (53 Features)...")

# Label Mapping: 0 = HUMAN, 1 = AI_GENERATED
for label_name, label_value in [("human", 0), ("ai", 1)]:
    base_path = os.path.join(DATASET_DIR, label_name)
    if not os.path.exists(base_path):
        print(f"⚠️ Warning: {base_path} not found. Skipping...")
        continue
    
    # Iterate through language folders (hindi, malayalam, etc.)
    for lang in os.listdir(base_path):
        lang_path = os.path.join(base_path, lang)
        if not os.path.isdir(lang_path):
            continue
            
        files = [f for f in os.listdir(lang_path) if f.lower().endswith(".mp3")]
        for file in tqdm(files, desc=f"Loading {label_name.upper()}-{lang.upper()}"):
            file_path = os.path.join(lang_path, file)
            feat = extract_features(file_path)
            if feat is not None:
                X.append(feat)
                y.append(label_value)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("❌ Error: No data found. Please check your 'data' folder structure.")
    exit()

print(f"✅ Extracted features for {len(X)} samples.")

# Split data: 80% Train, 20% Test (with Stratify to keep AI/Human ratio balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize optimized Random Forest
print("🧠 Training Optimized Random Forest (500 estimators)...")
model = RandomForestClassifier(
    n_estimators=500, 
    max_depth=25, 
    random_state=42, 
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save the trained model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)

# Final Evaluation
print("\n📊 Training Complete. Evaluation Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["HUMAN", "AI_GENERATED"]))
print(f"✅ Model saved at: {MODEL_PATH}")