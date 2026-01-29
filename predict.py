import sys
import joblib
import numpy as np
from features import extract_features

MODEL_PATH = "baseline_model.pkl"

# -------------------------------------------------
# Load model ONCE (important for API usage)
# -------------------------------------------------
model = joblib.load(MODEL_PATH)


def predict_from_bytes(audio_bytes: bytes):
    """
    Predict AI vs HUMAN from raw audio bytes.
    """

    # Extract features
    features = extract_features(audio_bytes)

    # Reshape for sklearn (1 sample, n features)
    features = features.reshape(1, -1)

    # Predict label
    prediction = model.predict(features)[0]

    # Predict confidence (if supported)
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(features)[0]))
        return prediction, confidence

    return prediction, None


# -------------------------------------------------
# CLI support (for local testing only)
# -------------------------------------------------
def predict_from_file(file_path: str):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return predict_from_bytes(audio_bytes)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    label, confidence = predict_from_file(file_path)

    print("Prediction:", label)
    if confidence is not None:
        print("Confidence:", round(confidence, 3))