from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import requests
import tempfile
import numpy as np
import traceback
from features import extract_features

app = FastAPI()

# Load model
model = joblib.load("baseline_model.pkl")

# ✅ Request schema
class PredictRequest(BaseModel):
    audio_url: str

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        audio_url = request.audio_url

        # Download audio
        response = requests.get(audio_url, timeout=15)
        response.raise_for_status()

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        # Feature extraction
        features = extract_features(temp_path)
        features = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "explanation": "Baseline MFCC + RandomForest classifier"
        }

    except Exception as e:
        print("ERROR during prediction:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


