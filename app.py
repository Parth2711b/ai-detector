from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64

app = FastAPI()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_KEY = "guvi"
SUPPORTED_FORMATS = {"mp3", "wav"}

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class VoiceRequest(BaseModel):
    audioFormat: str
    audioBase64: str

# -------------------------------------------------
# Placeholder detection logic
# (Replace with real ML inference later)
# -------------------------------------------------
def analyze_audio(audio_bytes: bytes):
    """
    Dummy logic.
    Replace with feature extraction + trained model.
    """
    size = len(audio_bytes)

    if size % 2 == 0:
        return "AI_GENERATED", 0.90, "Synthetic speech patterns detected"
    else:
        return "HUMAN", 0.87, "Natural speech variations detected"

# -------------------------------------------------
# API endpoint
# -------------------------------------------------
@app.post("/api/voice-detection")
def voice_detection(
    payload: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Validate audio format
    fmt = payload.audioFormat.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Only mp3 and wav supported")

    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Analyze audio
    classification, confidence, explanation = analyze_audio(audio_bytes)

    # Success response
    return {
        "status": "success",
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
