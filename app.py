from flask import Flask, render_template, request
import requests
import os
import base64
import json

app = Flask(__name__)

# Hugging Face Router API
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/models/laion/CLIP-ViT-B-32-multilingual-v1"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def query_hf_model(image_bytes):
    """Send base64-encoded image to Hugging Face Router API and return JSON."""
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = json.dumps({"inputs": b64_image})
        response = requests.post(HF_API_URL, headers=headers, data=payload, timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        image_file = request.files.get("image")
        if not image_file:
            result = {"error": "No file uploaded."}
        else:
            result = query_hf_model(image_file.read())
    return render_template("index.html", result=result)

# No app.run here; Gunicorn will start the app on Render
