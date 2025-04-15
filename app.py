from flask import Flask, request, render_template
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2

app = Flask(__name__)

# Load ONNX model
model_path = "eye_disease_model.onnx"  # Replace with your actual ONNX model
session = ort.InferenceSession(model_path)

# Define class labels
LABELS = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

# Image Preprocessing Function
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # Resize for model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image is uploaded
        if "file" not in request.files:
            return render_template("index.html", result="No file uploaded")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", result="No file selected")

        # Save and process image
        file_path = "static/uploads/" + file.filename
        file.save(file_path)
        img_array = preprocess_image(file_path)

        # Perform prediction
        outputs = session.run(None, {"input": img_array})
        prediction = np.argmax(outputs[0])
        result = LABELS[prediction]

        return render_template("index.html", result=result, image_path=file_path)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
