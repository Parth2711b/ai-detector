import os
import numpy as np
import joblib
from features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Lists to store data
X = []
y = []

DATA_DIR = "data"
LABELS = ["human", "ai"]

print("Loading data...")

for label in LABELS:
    folder_path = os.path.join(DATA_DIR, label)

    if not os.path.exists(folder_path):
        raise Exception(f"Folder not found: {folder_path}")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Skipped {file_name}: {e}")

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluation
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Training accuracy:", round(train_acc, 3))
print("Test accuracy:", round(test_acc, 3))

# Save model
joblib.dump(model, "baseline_model.pkl")
print("Model saved as baseline_model.pkl")
