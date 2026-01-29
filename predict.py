import sys
import joblib
import numpy as np
from features import extract_features

MODEL_PATH = "baseline_model.pkl"


def predict(file_path: str):
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Extract features (same as training)
    features = extract_features(file_path)

    # Reshape for sklearn (1 sample, n features)
    features = features.reshape(1, -1)

    # Predict label
    prediction = model.predict(features)[0]

    # Predict confidence (if supported)
    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(features)[0])
        return prediction, confidence

    return prediction, None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    label, confidence = predict(file_path)

    print("Prediction:", label)
    if confidence is not None:
        print("Confidence:", round(confidence, 3))
