import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX model
model_path = "../models/eye_disease_model.onnx"  # Ensure the path is correct
session = ort.InferenceSession(model_path)

# Define class labels (make sure they match your dataset's training order)
class_labels = ["Diabetic Retinopathy", "Glaucoma", "Cataract", "Normal"]

# Preprocess function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).numpy()
    return image

# Load test image
image_path = "../test_image.jpg"  # Ensure this path points to a valid image file
input_image = preprocess_image(image_path)

# Get the input name dynamically (useful for different ONNX models)
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_image})
predicted_class_index = np.argmax(outputs[0])

# Convert class index to disease name
predicted_disease = class_labels[predicted_class_index]

# Print the final output
print(f"🩺 Predicted Disease: {predicted_disease}")
