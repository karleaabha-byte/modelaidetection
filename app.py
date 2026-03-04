from flask import Flask, render_template, request
import requests
import base64
import os

app = Flask(__name__)

# Set your Hugging Face token as environment variable on Render
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/laion/CLIP-ViT-B-32-multilingual-v1"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_model(image_bytes):
    response = requests.post(
        HF_API_URL,
        headers=headers,
        files={"file": image_bytes}
    )
    return response.json()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "image" not in request.files:
            result = {"error": "No file uploaded."}
        else:
            image_file = request.files["image"]
            result = query_hf_model(image_file.read())
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
