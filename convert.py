#convert.py
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import onnx
import os

# Load the Hugging Face model
model_name = "google/pix2struct-docvqa-base"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample inputs (you might need to adjust this based on the actual model's input requirements)
question = "What is in the image?"
image_path = "path/to/your/image.jpg"

# Perform any necessary preprocessing

# Export the model to ONNX
onnx_model_path = os.path.join("model", "pix2struct-docvqa-base.onnx")
dummy_input = (torch.zeros(1, 3, 224, 224), torch.zeros(1, 1, dtype=torch.int64))  # Adjust shape based on model inputs
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)

print("Model converted to ONNX format.")
