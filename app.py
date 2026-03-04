from flask import Flask, render_template, request
import requests
import os
import base64

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def detect_ai(image_bytes):
    response = requests.post(
        HF_API_URL,
        headers=headers,
        data=image_bytes,
        timeout=30
    )
    return response.json()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if image_file:
            image_bytes = image_file.read()
            result = detect_ai(image_bytes)
            image_data = base64.b64encode(image_bytes).decode("utf-8")

    return render_template("index.html", result=result, image_data=image_data)
