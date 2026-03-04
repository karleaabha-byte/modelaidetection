import os
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# -------------------------
# Environment Variables
# -------------------------
# HF_TOKEN for faster authenticated download
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# -------------------------
# Flask Setup
# -------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai_images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# -------------------------
# Database Model
# -------------------------
class ImageUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    uploaded_at = db.Column(db.DateTime, server_default=db.func.now())
    score_model_a = db.Column(db.Float)
    score_model_b = db.Column(db.Float)
    score_model_c = db.Column(db.Float)
    final_score = db.Column(db.Float)

# -------------------------
# Load smaller CLIP model once
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch16"  # smaller model ~330MB
print("Loading CLIP model, please wait...")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print("Model loaded successfully!")

labels = ["a real photograph", "an AI-generated image"]

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Save file
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # Open image
            image = Image.open(image_path).convert("RGB")

            # CLIP detection
            inputs = processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            score_model_a = round(probs[0][0].item() * 100, 2)
            score_model_b = round(probs[0][1].item() * 100, 2)

            # Frequency heuristic for Model C
            img_array = np.array(image)
            score_model_c = round(min(np.var(img_array) / 10000, 1.0) * 100, 2)

            final_score = round((score_model_a + score_model_b + score_model_c) / 3, 2)

            result = {
                "score_model_a": score_model_a,
                "score_model_b": score_model_b,
                "score_model_c": score_model_c,
                "final_score": final_score
            }

            # Save to DB
            img_record = ImageUpload(
                filename=file.filename,
                score_model_a=score_model_a,
                score_model_b=score_model_b,
                score_model_c=score_model_c,
                final_score=final_score
            )
            db.session.add(img_record)
            db.session.commit()

    return render_template("index.html", result=result, image_path=image_path)

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
