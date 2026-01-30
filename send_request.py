import requests
import base64
import json
import os

# --- CONFIGURATION ---
URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "guvi" 
# Using raw string 'r' to handle Windows paths correctly
AUDIO_FILE_PATH = r"D:\COEP\ai_malayalam_00003.mp3"

def test_voice_detection():
    # 1. Check if the file exists locally
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"❌ Error: File '{AUDIO_FILE_PATH}' was not found.")
        return

    try:
        # 2. Encode Audio to Base64
        print(f"📦 Encoding '{AUDIO_FILE_PATH}' to Base64...")
        with open(AUDIO_FILE_PATH, "rb") as audio_file:
            # Read bytes -> encode to base64 -> convert to string
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')

        # 3. Prepare Payload (As per Competition Spec)
        payload = {
            "language": "Hindi",      # Options: Tamil / English / Hindi / Malayalam / Telugu
            "audioFormat": "mp3",
            "audioBase64": encoded_string
        }

        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }

        # 4. Send the POST Request
        print(f"🚀 Sending API request to {URL}...")
        response = requests.post(URL, json=payload, headers=headers)

        # 5. Display the Results
        if response.status_code == 200:
            print("✅ Success! API Response received:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"⚠️ API Error (Status Code: {response.status_code})")
            print(response.text)

    except Exception as e:
        print(f"🔥 An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_voice_detection()