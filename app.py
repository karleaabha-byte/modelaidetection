from flask import Flask, render_template, request
import requests
import os
import base64

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def query_hf_model(image_bytes, labels):
    try:
        payload = {
            "inputs": {
                "image": base64.b64encode(image_bytes).decode("utf-8"),
                "candidate_labels": labels
            }
        }

        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
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
        labels_input = request.form.get("labels")

        if not image_file:
            result = {"error": "No file uploaded."}
        elif not labels_input:
            result = {"error": "Please enter labels."}
        else:
            image_bytes = image_file.read()
            labels = [label.strip() for label in labels_input.split(",")]

            result = query_hf_model(image_bytes, labels)
            image_data = base64.b64encode(image_bytes).decode("utf-8")

    return render_template("index.html", result=result, image_data=image_data)
