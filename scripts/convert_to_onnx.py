import torch
from train_model import EyeDiseaseCNN  # Import trained model class

# Load trained PyTorch model
model = EyeDiseaseCNN()
model.load_state_dict(torch.load("../models/eye_disease_model.pth"))
model.eval()

# Dummy input for ONNX conversion
dummy_input = torch.randn(1, 3, 224, 224)

# Convert to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "../models/eye_disease_model.onnx",  # Save as ONNX
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("✅ Model converted to ONNX and saved as 'eye_disease_model.onnx'")
