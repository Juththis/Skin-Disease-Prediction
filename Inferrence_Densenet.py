# infer_with_picker.py

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
from tkinter import Tk, filedialog


# File selection

Tk().withdraw()  # hide root window
img_path = filedialog.askopenfilename(title="Select Image")

if not img_path:
    raise ValueError("No image selected")

print("Selected:", img_path)


# Preprocessing

def preprocess_image(image):
    cropped = image[25:425, 25:575]
    resized = cv2.resize(cropped, (224, 224))

    normalized = resized / 255.0
    normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return normalized.astype(np.float32)


# Load model

checkpoint = torch.load("skin_model.pth", map_location="cpu", weights_only=True)

classes = checkpoint["classes"]
num_classes = len(classes)

model = densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Load and preprocess image

image = cv2.imread(img_path)

if image is None:
    raise ValueError("Failed to load image")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess_image(image)

image = np.transpose(image, (2, 0, 1))  # HWC → CHW
image = torch.tensor(image).unsqueeze(0)


# Prediction

with torch.no_grad():
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)
    percentages = probs * 100


# Output

print("\nPrediction probabilities:\n")

for i, cls in enumerate(classes):
    print(f"{cls}: {percentages[0][i]:.2f}%")

_, pred = torch.max(probs, 1)
print(f"\nFinal Prediction: {classes[pred.item()]}")