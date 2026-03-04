from flask import Flask, render_template, request
import requests
import os
import base64

app = Flask(__name__)

# ❗ Set this in Render Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Model that actually outputs real vs ai
HF_API_URL = "https://api-inference.huggingface.co/models/manogna/pallete-ai-image-detection"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query_hf_model(image_bytes):
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "inputs": {
            "image": img_b64
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
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
            image_bytes = image_file.read()
            result = query_hf_model(image_bytes)

    return render_template("index.html", result=result)
