import io
import base64
import joblib
import numpy as np
import librosa
import json
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

# --- CONFIGURATION ---
app = FastAPI(title="Voice Detection API")
API_KEY = "guvi"
MODEL_PATH = "models/voice_detector.pkl"

# --- PYDANTIC MODEL (This creates the input box in /docs) ---
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# --- LOAD MODEL ---
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Advanced 53-Feature Model Loaded")
except Exception as e:
    print(f"⚠️ Model not found: {e}")
    model = None

# --- FEATURE EXTRACTION ---
def extract_audio_features(audio_bytes):
    audio_data = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_data, sr=16000)
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
    
    return np.hstack([mfcc, contrast, tonnetz]).reshape(1, -1)

# --- API ENDPOINT ---
@app.post("/api/voice-detection")
async def voice_detection(
    request: Request, 
    data_hint: VoiceDetectionRequest, # This line brings back the Input Box
    x_api_key: str = Header(None)
):
    # 1. API Key Check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 2. Get the RAW body (This avoids the strict 422 error)
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8")
        
        # 3. Manual parse with strict=False to ignore hidden "control characters"
        data = json.loads(body_str, strict=False)
        
        audio_b64 = data.get("audioBase64", "")
        
        # 4. Deep Clean the Base64 string
        clean_b64 = "".join(audio_b64.split())
        
        # 5. Decode
        audio_bytes = base64.b64decode(clean_b64)

        if model:
            features = extract_audio_features(audio_bytes)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            classification = "AI_GENERATED" if prediction == 1 else "HUMAN"
            confidence = round(float(probabilities[prediction]), 2)
            
            return {
                "status": "success",
                "language": data.get("language"),
                "classification": classification,
                "confidenceScore": confidence,
                "explanation": f"Signal analysis suggests {classification} voice."
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)