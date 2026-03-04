from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# Get Hugging Face token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/laion/CLIP-ViT-B-32-multilingual-v1"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_model(image_bytes):
    """Send image bytes to Hugging Face API and return JSON."""
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            files={"file": image_bytes},
            timeout=30  # timeout so request doesn't hang forever
        )
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

# Remove app.run for Render; Gunicorn will start the app
