from flask import Flask, render_template, request
import onnxruntime as ort
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load ONNX model once at startup
session = ort.InferenceSession("model/mobilenetv2-7.onnx")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)

    # Normalize to 0-1
    image /= 255.0

    # Change shape from HWC to CHW
    image = image.transpose(2, 0, 1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if image_file and allowed_file(image_file.filename):
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            input_tensor = preprocess(image)

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_tensor})

            scores = outputs[0][0]
            predicted_class = int(np.argmax(scores))
            confidence = float(np.max(scores)) * 100

            result = {
                "label": f"Class #{predicted_class}",
                "confidence": f"{confidence:.2f}%"
            }

            image_data = base64.b64encode(image_bytes).decode("utf-8")

        else:
            result = {"error": "Invalid file type."}

    return render_template("index.html", result=result, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
