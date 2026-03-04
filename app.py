from flask import Flask, render_template, request
import requests
import os
import base64

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def query_hf_model(image_bytes):
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image_bytes,
            timeout=30
        )

        if response.status_code != 200:
            return {
                "error": f"HF Error {response.status_code}",
                "details": response.text
            }

        return response.json()

    except Exception as e:
        return {"error": str(e)}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if not image_file:
            result = {"error": "No file uploaded."}
        else:
            image_bytes = image_file.read()
            result = query_hf_model(image_bytes)
            image_data = base64.b64encode(image_bytes).decode("utf-8")

    return render_template("index.html", result=result, image_data=image_data)
