from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64

app = FastAPI()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_KEY = "guvi"
SUPPORTED_FORMATS = {"mp3"}
SUPPORTED_LANGUAGES = {"tamil", "english", "hindi", "malayalam", "telugu"}

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class VoiceRequest(BaseModel):
    language: str
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
        return "AI_GENERATED", 0.90, "Unnatural pitch stability and robotic speech patterns detected"
    else:
        return "HUMAN", 0.87, "Natural pitch variation and human speech patterns detected"

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
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )

    # Validate language
    lang = payload.language.lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Unsupported language"}
        )

    # Validate audio format (MP3 only)
    fmt = payload.audioFormat.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Only mp3 format is supported"}
        )

    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Invalid Base64 audio"}
        )

    # Analyze audio
    classification, confidence, explanation = analyze_audio(audio_bytes)

    # Success response (exact spec)
    return {
        "status": "success",
        "language": payload.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
