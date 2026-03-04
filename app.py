from flask import Flask, render_template, request
import requests
import os
import base64

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

HF_API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/octet-stream"
}

def detect_ai(image_bytes):
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image_bytes,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if image_file and allowed_file(image_file.filename):
            image_bytes = image_file.read()
            result = detect_ai(image_bytes)
            image_data = base64.b64encode(image_bytes).decode("utf-8")
        else:
            result = {"error": "Invalid file type"}

    return render_template("index.html", result=result, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
